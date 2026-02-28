"""
问题生成工具 - 供Agent调用
"""

from typing import Type, Optional, List, Dict
import json
import os
from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

from src.utils.location_service import get_realtime_context
from src.llm.http_client_pool import get_siliconflow_chat_openai


class QuestionGenerationToolArgs(BaseModel):
    """问题生成工具参数"""
    dimension_name: str = Field(..., description="当前评估的维度名称，如'定向力'、'记忆力'等")
    dimension_description: str = Field(default="", description="维度的描述，如'时间/地点定向'")
    knowledge_context: str = Field(
        default="", 
        description="检索到的相关医学知识，用于指导问题生成"
    )
    patient_age: Optional[int] = Field(default=None, description="患者年龄")
    patient_education: Optional[int] = Field(default=None, description="患者教育年限")
    patient_name: Optional[str] = Field(default=None, description="患者姓名，用于称呼患者")
    patient_gender: Optional[str] = Field(default=None, description="患者性别：男/女")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        default=None, 
        description="最近的对话历史（JSON格式列表）"
    )
    generated_questions: Optional[List[str]] = Field(
        default=None, 
        description="已生成的问题列表（用于去重）"
    )
    patient_emotion: Optional[str] = Field(default=None, description="患者当前情绪")
    task_instruction: Optional[str] = Field(default=None, description="本轮任务指令（内部用，不向对方暴露）")
    persona_hooks: Optional[List[str]] = Field(default=None, description="个性化钩子（兴趣/习惯/刚聊到的点）")
    must_include: Optional[List[str]] = Field(default=None, description="生成问题必须包含的关键词/数字")
    avoid_questions: Optional[List[str]] = Field(default=None, description="需要避免的已问过问题（防止重复）")


class QuestionGenerationTool(BaseTool):
    """
    问题生成工具
    
    基于当前评估维度、医学知识和患者信息，生成合适的评估问题。
    """
    
    name: str = "generate_question"
    description: str = (
        "生成用于评估患者认知功能的问题。"
        "基于当前维度、检索到的医学知识、患者画像和对话历史，生成温和、专业、易懂的问题。"
        "适用场景：需要询问患者新问题时使用。"
    )
    
    args_schema: Type[BaseModel] = QuestionGenerationToolArgs
    
    _llm: ChatOpenAI = PrivateAttr()
    _fast_llm: Optional[ChatOpenAI] = PrivateAttr(default=None)
    _default_model: str = PrivateAttr(default="Qwen/Qwen2.5-72B-Instruct")
    _fast_model: Optional[str] = PrivateAttr(default=None)
    _fast_dimensions: set = PrivateAttr(default_factory=set)
    
    def __init__(
        self,
        use_local: bool = False,  # 新增参数
        llm_instance = None,      # 允许直接传入LLM实例
        **kwargs
    ):
        super().__init__(**kwargs)

        if llm_instance is not None:
            self._llm = llm_instance
            self._fast_llm = None
            self._default_model = "custom_llm_instance"
            self._fast_model = None
            self._fast_dimensions = set()
            print("[QuestionGenTool] 🚀 使用外部注入 LLM 实例")
            return

        self._default_model = os.getenv("QUESTION_GEN_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        self._fast_model = os.getenv("QUESTION_GEN_FAST_MODEL", "Qwen/Qwen2.5-7B-Instruct")

        try:
            llm_temperature = float(os.getenv("QUESTION_GEN_TEMPERATURE", "0.7"))
        except ValueError:
            llm_temperature = 0.7
        try:
            llm_max_tokens = int(os.getenv("QUESTION_GEN_MAX_TOKENS", "160"))
        except ValueError:
            llm_max_tokens = 160
        try:
            llm_timeout = float(os.getenv("QUESTION_GEN_TIMEOUT", "20"))
        except ValueError:
            llm_timeout = 20.0

        fast_dims_raw = os.getenv("QUESTION_GEN_FAST_DIMENSIONS", "闲聊")
        self._fast_dimensions = {d.strip() for d in fast_dims_raw.split(",") if d.strip()}

        self._llm = get_siliconflow_chat_openai(
            model=self._default_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens,
            timeout=llm_timeout,
        )

        if self._fast_model and self._fast_model != self._default_model:
            self._fast_llm = get_siliconflow_chat_openai(
                model=self._fast_model,
                temperature=max(0.3, llm_temperature - 0.1),
                max_tokens=llm_max_tokens,
                timeout=llm_timeout,
            )
        else:
            self._fast_llm = None

        fast_model_info = self._fast_model if self._fast_llm else "disabled"
        print(
            f"[QuestionGenTool] 🚀 主模型={self._default_model} | "
            f"快速模型={fast_model_info} | 快速维度={sorted(self._fast_dimensions)}"
        )

    def _select_llm(self, dimension_name: str) -> tuple[ChatOpenAI, str]:
        """按维度选择模型：核心评估走质量模型，闲聊走快速模型。"""
        normalized_dimension = (dimension_name or "").strip()
        if self._fast_llm and normalized_dimension in self._fast_dimensions:
            return self._fast_llm, self._fast_model or self._default_model
        return self._llm, self._default_model

    def _looks_like_question(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        markers = ("吗", "呢", "么", "什么", "怎么", "哪里", "哪个", "多少", "为何", "为啥", "是否")
        return any(m in t for m in markers) or ("？" in t) or ("?" in t)

    def _keep_single_question(self, text: str) -> str:
        """压缩为单个主问句，减少“连环双问”带来的冗余。"""
        import re
        q = (text or "").strip()
        if not q:
            return q
        q = q.replace("?", "？")
        segments = [seg.strip("，。；;!！~～ ") for seg in re.split(r"[？]+", q) if seg.strip("，。；;!！~～ ")]
        if not segments:
            return ""
        chosen = segments[0]
        if len(segments) > 1 and not self._looks_like_question(chosen):
            for seg in segments[1:]:
                if self._looks_like_question(seg):
                    chosen = seg
                    break
        if self._looks_like_question(chosen):
            return f"{chosen}？"
        return chosen

    def _normalize_for_similarity(self, text: str) -> str:
        import re
        t = (text or "").strip().lower()
        t = re.sub(r"\s+", "", t)
        t = re.sub(r"[，。！？、:：;；\"'“”‘’（）()【】\[\]~～\-—]", "", t)
        return t

    def _sanitize_ack(self, ack: str, q: str = "") -> str:
        """清理 ack 中残留问句，并避免和 q 语义重复。"""
        import re
        from difflib import SequenceMatcher

        a = (ack or "").strip()
        if not a:
            return ""

        a = a.replace("?", "？")
        if "？" in a:
            a = a.split("？", 1)[0].strip()

        parts = [p.strip("，,。!！~～ ") for p in re.split(r"[。!！~～]", a) if p.strip("，,。!！~～ ")]
        kept_parts = []
        for p in parts:
            if self._looks_like_question(p):
                break
            kept_parts.append(p)

        if kept_parts:
            a = "。".join(kept_parts).strip("，,。 ")
        elif self._looks_like_question(a):
            a = ""
        else:
            a = a.strip("，,。 ")

        if a and q:
            na = self._normalize_for_similarity(a)
            nq = self._normalize_for_similarity(q)
            if na and nq:
                ratio = SequenceMatcher(None, na, nq).ratio()
                if na in nq or nq in na or ratio >= 0.82:
                    a = ""

        if a and not a.endswith(("。", "！", "!", "~", "～")):
            a += "。"
        return a

    def _build_ack_fallback(
        self,
        patient_name: Optional[str] = None,
        patient_gender: Optional[str] = None,
        patient_age: Optional[int] = None,
        user_answer: Optional[str] = None,
    ) -> str:
        """当 ack 被清空时，补一条简短的非问句回应，保留过渡感。"""
        import re

        snippet = ""
        if user_answer:
            candidate = re.split(r"[，。！？?]", user_answer.strip())[0].strip()
            if 3 <= len(candidate) <= 18:
                snippet = candidate

        if snippet:
            base = f"您刚说{snippet}，我听着挺好。"
        else:
            base = "听您这么说我也挺高兴的。"

        if patient_name:
            if patient_gender == '男':
                suffix = '爷爷' if (patient_age and patient_age >= 60) else '叔叔'
            else:
                suffix = '奶奶' if (patient_age and patient_age >= 60) else '阿姨'
            return f"{patient_name}{suffix}，{base}"
        return base

    def _extract_topic_hint(self, task_instruction: Optional[str], bridge_hint: Optional[str]) -> str:
        import re
        task = task_instruction or ""
        m = re.search(r"「([^」]{1,12})」", task)
        if m:
            return m.group(1).strip()
        if bridge_hint:
            return bridge_hint.split("→")[-1].strip()
        return ""

    def _is_too_open_ended(self, q: str) -> bool:
        text = (q or "").strip()
        if not text:
            return False
        risky_patterns = [
            r"新鲜事",
            r"分享一下",
            r"最近.*(怎么样|咋样|如何)",
            r"最近有.*(吗|么|呢|？|\?)",
            r"讲讲",
            r"说说看",
        ]
        import re
        return any(re.search(p, text) for p in risky_patterns)

    def _rewrite_open_question(
        self, q: str, topic_hint: str, dimension_name: str
    ) -> str:
        """仅在命中空泛问法时触发，把问题改成更具体、老人更易回答的问法。"""
        topic = (topic_hint or "").strip()
        if "爱好" in topic or "兴趣" in topic:
            return "您平时更喜欢散散步，还是在家看看电视呀？"
        if "生活" in topic or "日常" in topic:
            return "您今天白天在家一般做点啥呀？"
        if "心情" in topic:
            return "您今天心情挺不错的，是不是和家里人聊了会儿天呀？"
        if "回忆" in topic:
            return "您年轻的时候平时最爱做什么呀？"
        if "天气" in topic:
            return "今天天气还行，您今天有出去走走吗？"
        if dimension_name == "闲聊":
            return "您今天在家是看看电视，还是听听歌呀？"
        return q
    
    def _run(
        self,
        dimension_name: str,
        dimension_description: str = "",
        knowledge_context: str = "",
        patient_age: Optional[int] = None,
        patient_education: Optional[int] = None,
        patient_name: Optional[str] = None,
        patient_gender: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,  # 🔥 类型变更：支持结构化历史
        generated_questions: Optional[List[str]] = None,  # 🔥 新增：已生成问题列表
        patient_emotion: Optional[str] = None,
        task_instruction: Optional[str] = None,
        persona_hooks: Optional[List[str]] = None,
        must_include: Optional[List[str]] = None,
        avoid_questions: Optional[List[str]] = None,
        bridge_hint: Optional[str] = None,  # 🔥 新增：自然过渡提示
    ) -> str:
        """
        生成问题
        
        Returns:
            JSON格式：{"success": true, "question": "生成的问题"}
        """
        import time
        from src.utils.tool_logger import ToolLogger
        
        self._logger = ToolLogger("QuestionGenTool")
        self._logger.start(
            维度=dimension_name,
            任务指令=task_instruction[:30] if task_instruction else "N/A"
        )
        _start_time = time.time()
        
        system_prompt = (
            "你是对方的晚辈或老邻居，在轻松聊天。\n"
            "你不是AI、不是医生、不是采访者。\n\n"
            "严格只输出 JSON：{\"ack\":\"...\",\"q\":\"...\"}\n\n"
            "规则：\n"
            "1) ack 先回应对方刚说的话，再自然过渡到下一个话题；ack 不要问号。\n"
            "2) q 是下一句要问的问题，必须口语化、具体、可直接回答。\n"
            "3) 必须执行任务指令；若有 must_include，q 必须包含。\n"
            "4) 若有 bridge_hint，q 必须直接命中该目标话题。\n"
            "5) 对认知减退老人，一次只问一件事，优先日常具体问题。\n"
            "6) 避免空泛问法，如“最近有什么新鲜事分享一下”。\n"
            "7) 可用轻量选择式问法（如“您更喜欢A还是B”），但不要连环追问。\n"
            "8) 不要输出任何解释、前后缀或 Markdown。"
        )
        
        user_prompt_parts = []
        
        # 聊天话题引导（内部使用，不暴露给用户）
        topic_hints = {
            '定向力': '围绕时间、地点做自然提问（如星期、日期、所在区域）。',
            '即时记忆': '让对方记住并复述简短信息（词语/数字）。',
            '注意力与计算': '用生活化小算术或连续计算测试注意力。',
            '延迟回忆': '回问之前提过的信息，观察回忆情况。',
            '语言': '围绕命名、复述、理解指令进行简短提问。',
            '构图(临摹)': '用简单图形临摹相关引导语。'
        }
        hint = topic_hints.get(dimension_name, '随便聊聊')
        user_prompt_parts.append(f"聊天方向提示：{hint}")

        if task_instruction:
            user_prompt_parts.append(f"\n【本轮要做的事（只给你看）】：{task_instruction}")

        if persona_hooks:
            hooks_text = '、'.join([h for h in persona_hooks if h])
            if hooks_text:
                user_prompt_parts.append(f"\n【个性化钩子】尽量在一句话里自然提到其中至少一个：{hooks_text}")

        if must_include:
            must_text = '、'.join([m for m in must_include if m])
            if must_text:
                user_prompt_parts.append(f"\n【必须包含】生成的问题里必须出现这些关键词/数字：{must_text}")
        
        # 🆕 合并非核心话题去重逻辑，统统加入 avoid_questions
        all_avoid = []
        if avoid_questions:
            all_avoid.extend(avoid_questions)
        if generated_questions:
            all_avoid.extend(generated_questions)
        
        if all_avoid:
            import re
            core_questions = []
            for q in all_avoid[-8:]:
                match = re.search(r'[，。]?([^，。]*[吗呢啊？?])$', q)
                if match:
                    core_questions.append(match.group(1).strip())
                else:
                    core_questions.append((q.split('，')[-1] if '，' in q else q).strip())
            if core_questions:
                avoid_text = '；'.join(core_questions[:6])
                user_prompt_parts.append(
                    f"\n【禁止重复】以下问法已问过，避免同义复问：{avoid_text}"
                )
        
        # 🔥 新增：自然过渡提示 - 融入 ack
        if bridge_hint:
            target_topic = bridge_hint.split("→")[-1].strip() if "→" in bridge_hint else bridge_hint
            user_prompt_parts.append(
                f"\n【话题约束】目标话题：{target_topic}。q 必须直接问这个话题，不能偏题。"
            )
        
        # 添加称呼信息（根据性别和年龄生成正确称呼）
        if patient_name:
            # 根据性别确定称呼后缀
            if patient_gender == '男':
                suffix = '爷爷' if (patient_age and patient_age >= 60) else '叔叔'
            else:
                suffix = '奶奶' if (patient_age and patient_age >= 60) else '阿姨'
            full_name = f"{patient_name}{suffix}"
            user_prompt_parts.append(f"\n对方叫{patient_name}，【必须】统一称呼'{full_name}'，不要只叫'{patient_name}'或其他变体")
        
        # 如果是定向力维度，注入当前真实时间、地点和天气信息
        if "定向" in dimension_name or "orientation" in dimension_description.lower():
            # 获取完整实时上下文（位置、时间、天气）
            context = get_realtime_context()
            time_info = context['time']
            location = context['location']
            weather = context['weather']
            
            realtime_info = (
                f"\n【当前真实信息，聊天时可以用到】\n"
                f"今天是{time_info['year']}年{time_info['month']}月{time_info['day']}日，星期{time_info['weekday']}，{time_info['season']}\n"
                f"地点：{location.get('city', '')} {location.get('district', '')}\n"
                f"天气：{weather.get('temperature', '')} {weather.get('weather', '')}\n"
            )
            user_prompt_parts.append(realtime_info)
        
        if patient_age:
            user_prompt_parts.append(f"对方{patient_age}岁")
        
        if patient_emotion and patient_emotion != 'neutral':
            emotion_map = {'happy': '心情不错', 'sad': '有点低落', 'angry': '有点烦躁', 'fear': '有点紧张'}
            user_prompt_parts.append(f"对方现在{emotion_map.get(patient_emotion, '心情一般')}")
        
        last_user_msg = ""
        if conversation_history:
            history_list = conversation_history if isinstance(conversation_history, list) else []
            
            prev_ai_responses = []  # 收集之前AI的回复，用于防重复
            
            if history_list and len(history_list) > 0:
                # 倒序查找最后一句用户的话
                for msg in reversed(history_list):
                    if msg.get('role') == 'user':
                        last_user_msg = msg.get('content', '')
                        break
                
                # 收集AI之前的回复（用于风格防重复）
                for msg in history_list:
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if content and len(content) > 5:
                            prev_ai_responses.append(content[:60])
                
                # 最近2轮对话，控制 token
                recent = history_list[-4:]
                context_desc = "\n【最近聊天记录】\n"
                for msg in recent:
                    role_zh = "你" if msg.get("role") == "assistant" else "对方"
                    content = msg.get("content", "")
                    context_desc += f"{role_zh}: {content}\n"
                user_prompt_parts.append(context_desc)
            
            if last_user_msg:
                print(f"  ├── 🗣️  用户上轮回答: \"{last_user_msg}\"")
                user_prompt_parts.append(f"🔔 对方刚才说：「{last_user_msg}」")
                
                # 🔥 风格防重复：告诉LLM之前用了什么风格
                if prev_ai_responses:
                    # 提取之前回复的开头模式，让LLM避开
                    prev_starts = []
                    for resp in prev_ai_responses[-3:]:  # 最近3条AI回复
                        # 取回复的前15个字作为风格标记
                        start = resp[:15].rstrip('，。！')
                        if start:
                            prev_starts.append(start)
                    if prev_starts:
                        starts_text = '」「'.join(prev_starts)
                        user_prompt_parts.append(
                            f"\n🚫【防重复】你之前的回复开头是：「{starts_text}」\n"
                            f"这次**必须**用完全不同的开头、不同的回应方式、不同的过渡手法！"
                        )
            else:
                user_prompt_parts.append(f"\n最近对话：\n{str(conversation_history)[-300:]}")
        else:
            print(f"[QuestionGenTool] ⚠️ 没有对话历史传入！")
        
        user_prompt_parts.append("\n直接输出你的回复（必须先回应再提问）：")
        
        user_prompt = "\n".join(user_prompt_parts)
        
        try:
            active_llm, active_model = self._select_llm(dimension_name)
            if active_model != self._default_model:
                print(f"[QuestionGenTool] ⚡ 快速模型路径: {active_model} (维度={dimension_name})")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            try:
                response = active_llm.invoke(messages)
            except Exception as invoke_error:
                # 主模型失败时回退到快速模型，避免一次失败拖慢整轮对话
                if active_llm is self._llm and self._fast_llm is not None:
                    print(f"[QuestionGenTool] ⚠️ 主模型调用失败，回退快速模型: {invoke_error}")
                    active_model = self._fast_model or self._default_model
                    response = self._fast_llm.invoke(messages)
                else:
                    raise
            
            # 兼容处理：本地模型返回str，ChatOpenAI返回AIMessage
            if hasattr(response, "content"):
                raw_output = response.content
            else:
                raw_output = str(response)
            
            raw_output = raw_output.strip()
            # print(f"  ├── 📝 LLM 原始输出: {raw_output[:50]}...")
            
            # 🆕 解析 JSON 结构化输出
            import re
            import json as json_module
            
            question = None  # 最终结果
            
            # 清洗 JSON：去掉 markdown 代码块标记
            cleaned_output = raw_output
            if "```" in cleaned_output:
                cleaned_output = re.sub(r"```json?\s*", "", cleaned_output)
                cleaned_output = cleaned_output.replace("```", "").strip()
            
            # 尝试解析 JSON
            try:
                # 尝试直接解析
                parsed = json_module.loads(cleaned_output)
                ack = parsed.get("ack", "").strip()
                q = parsed.get("q", "").strip()
                q = self._keep_single_question(q)
                topic_hint = self._extract_topic_hint(task_instruction, bridge_hint)
                if self._is_too_open_ended(q):
                    q = self._rewrite_open_question(q, topic_hint, dimension_name)
                ack = self._sanitize_ack(ack, q)
                if not ack and q:
                    ack = self._build_ack_fallback(patient_name, patient_gender, patient_age, last_user_msg)
                
                print(f"[QuestionGenTool] ✅ JSON解析成功: ack='{ack}', q='{q[:30]}...'")
                
                # 🆕 校验：ack 不应该过分长（放宽到80字，允许更丰富的回应）
                if ack and len(ack) > 80:
                    # 尝试在句号/感叹号处截断，保持语义完整
                    for i in range(60, len(ack)):
                        if ack[i] in '。！~':
                            ack = ack[:i+1]
                            break
                    else:
                        ack = ack[:70]
                
                # 拼接 acknowledgment 和 question
                if ack and q:
                    # 根据 ack 结尾决定连接符
                    if ack.endswith(("！", "!", "~", "～", "。", ".")):
                        question = f"{ack}{q}"
                    else:
                        question = f"{ack}，{q}"
                elif q:
                    question = q
                elif ack:
                    question = ack
                    
            except json_module.JSONDecodeError:
                print(f"[QuestionGenTool] ⚠️ JSON解析失败，尝试提取...")
                # 尝试用正则提取
                ack_match = re.search(r'"ack"\s*:\s*"([^"]*)"', cleaned_output)
                q_match = re.search(r'"q"\s*:\s*"([^"]*)"', cleaned_output)
                
                if ack_match and q_match:
                    ack = ack_match.group(1).strip()
                    q = q_match.group(1).strip()
                    q = self._keep_single_question(q)
                    topic_hint = self._extract_topic_hint(task_instruction, bridge_hint)
                    if self._is_too_open_ended(q):
                        q = self._rewrite_open_question(q, topic_hint, dimension_name)
                    ack = self._sanitize_ack(ack, q)
                    if not ack and q:
                        ack = self._build_ack_fallback(patient_name, patient_gender, patient_age, last_user_msg)
                    print(f"[QuestionGenTool] ✅ 正则提取成功: ack='{ack}', q='{q[:30]}...'")
                    if ack and q:
                        question = f"{ack}，{q}" if not ack.endswith(("！", "!", "~", "～")) else f"{ack}{q}"
                    else:
                        question = q or ack
                else:
                    # JSON 完全失败，回退到原始输出
                    print(f"[QuestionGenTool] ⚠️ 回退到原始输出")
                    question = raw_output.strip().strip('"').strip("'")
            
            # 如果还是没有问题，使用回退
            if not question or len(question) < 3:
                question = raw_output.strip().strip('"').strip("'")
            
            # 最终清洗
            question = re.sub(r"^(医生|护士|我)[:：]\s*", "", question).strip()
            
            # 清理问题末尾
            if question and not question.endswith(("？", "?", "。", ".", "！", "!", "~", "～")):
                question += "？"
            
            result = json.dumps({
                "success": True,
                "question": question,
                "dimension": dimension_name
            }, ensure_ascii=False, indent=2)
            
            return self._finalize_result(question, dimension_name, _start_time)
            
        except Exception as e:
            _elapsed = time.time() - _start_time
            print(f"❌ [QuestionGenTool] 问题生成失败 (耗时: {_elapsed:.2f}秒): {e}\n")
            return json.dumps({
                "success": False,
                "error": str(e),
                "fallback_question": f"请您描述一下您的{dimension_name}情况。"
            }, ensure_ascii=False, indent=2)
    
        

    

            
    def _finalize_result(self, question, dimension_name, start_time):
        """统一处理最终结果：清理符号、日志输出、构建JSON"""
        import re
        import json as json_module
        import time
        
        # 最终清洗
        question = re.sub(r"^(医生|护士|我)[:：]\s*", "", question).strip()
        
        # 🔥 清理 Markdown 符号 ** 和多余逗号
        question = question.replace("**", "")  # 移除 **
        question = re.sub(r"，+", "，", question)  # 多个逗号合并为一个
        question = re.sub(r"。，", "，", question)  # 。，变为，
        question = re.sub(r"，(?=[？?。!！~～])", "", question)  # 逗号后面直接跟标点时删除逗号
        # 只保留一个主问句，避免“两个连续问句”听感过于密集
        question = self._keep_single_question(question) if ("？" in question or "?" in question) else question
        
        # 清理问题末尾
        if question and not question.endswith(("？", "?", "。", ".", "！", "!", "~", "～")):
            question += "？"
        
        result = json_module.dumps({
            "success": True,
            "question": question,
            "dimension": dimension_name
        }, ensure_ascii=False, indent=2)
        
        if hasattr(self, '_logger'):
            self._logger.end(生成问题=question[:50])
        return result

    def generate_natural_transition(
        self,
        user_answer: str,
        dimension_name: str,
        patient_name: Optional[str] = None,
        patient_gender: Optional[str] = None,
        patient_age: Optional[int] = None,
        chat_history: Optional[List[Dict]] = None,
        current_emotion: str = 'neutral',
    ) -> str:
        """
        生成自然过渡回应（回应老人的话 + 自然引出下一个话题）
        
        Args:
            user_answer: 用户的回答
            dimension_name: 当前评估维度
            patient_name: 患者姓名
            patient_gender: 患者性别
            patient_age: 患者年龄
            chat_history: 对话历史
            current_emotion: 当前情绪
            
        Returns:
            JSON格式：{"success": true, "transition": "过渡回应"}
        """
        import time
        import random
        _start = time.time()
        
        # 生成称呼
        if patient_name:
            if patient_gender == '男':
                suffix = '爷爷' if (patient_age and patient_age >= 60) else '叔叔'
            else:
                suffix = '奶奶' if (patient_age and patient_age >= 60) else '阿姨'
            greeting = f"{patient_name}{suffix}"
        else:
            greeting = ""
        
        try:
            # 提取最近对话
            recent_context = ""
            if chat_history and len(chat_history) >= 2:
                recent_turns = []
                for msg in chat_history[-4:]:
                    role = "你" if msg.get('role') == 'assistant' else "对方"
                    content = msg.get('content', '')[:50]
                    if content:
                        recent_turns.append(f"{role}：{content}")
                recent_context = "\n".join(recent_turns)
            
            system_prompt = (
                "你是对方的家人或老朋友，正在陪他们聊天。\n"
                "现在需要生成一个自然的过渡回应，让对话更像朋友聊天。\n\n"
                "要求：\n"
                "1. **必须先具体回应对方刚才说的话！**（比如对方说吃了饺子，你要评两句饺子）\n"
                "2. 然后再用一句话自然地引出下一个话题或问题\n"
                "3. 语气亲切、口语化，像朋友聊天\n"
                "4. 控制在40-60字之间\n"
                "5. **禁止**生硬转折（如'说到这个'、'提到这个'）\n"
                "6. 不要带任何前缀（如'医生:'）\n"
            )
            
            user_prompt = f"""最近对话：
{recent_context}

对方刚才说："{user_answer}"
当前情绪：{current_emotion}
下一个话题方向：{dimension_name}

请生成一个自然的过渡回应（先具体回应对方说的"{user_answer}"，再自然过渡到"{dimension_name}"）："""
            
            transition_llm, transition_model = self._select_llm(dimension_name)
            if transition_model != self._default_model:
                print(f"[QuestionGenTool] ⚡ 自然过渡使用快速模型: {transition_model}")

            response = transition_llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            # 提取文本
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            result = result.strip().strip('"').strip("'")
            
            # 清理前缀
            import re
            result = re.sub(r"^(医生|护士|我)[:：]\s*", "", result)
            
            # 添加称呼
            if greeting and not result.startswith(patient_name):
                result = f"{greeting}，{result}"
            
            print(f"[QuestionGenTool] ✅ 自然过渡生成完成 (耗时: {(time.time()-_start)*1000:.0f}ms)")
            
            return json.dumps({
                "success": True,
                "transition": result
            }, ensure_ascii=False)
            
        except Exception as e:
            print(f"[QuestionGenTool] ⚠️ 自然过渡生成失败: {e}")
            # 备用方案
            responses = [
                "嗯，您说得对。对了，我想问您...",
                "理解理解。那咱们聊点别的...",
                "是这样啊。诶，我突然想到...",
            ]
            fallback = random.choice(responses)
            if greeting:
                fallback = f"{greeting}，{fallback}"
            
            return json.dumps({
                "success": True,
                "transition": fallback
            }, ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
