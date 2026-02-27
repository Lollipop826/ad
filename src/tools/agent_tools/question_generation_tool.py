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
    
    def __init__(
        self,
        use_local: bool = False,  # 新增参数
        llm_instance = None,      # 允许直接传入LLM实例
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 🔥 共享 ChatOpenAI + 共享 httpx 连接池（sync/async）
        print("[QuestionGenTool] 🚀 复用共享 ChatOpenAI（Qwen2.5-72B）")
        self._llm = get_siliconflow_chat_openai(
            model="Qwen/Qwen2.5-72B-Instruct",
            temperature=0.7,
            max_tokens=200,
            timeout=30,
        )
    
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
            "你是对方的晚辈或老邻居，正在陪老人唠嗑。你不是AI，不是医生，不是采访者。\n"
            "你是一个真实的人，有自己的感受、记忆和观点。\n\n"
            
            "🔔 **输出格式**：必须严格 JSON：\n"
            '{"ack": "回应 + 过渡", "q": "下一个问题"}\n\n'
            
            "═══════════════════════════════════════\n"
            "  📌 ack 字段 = 具体回应 + 过渡语句\n"
            "═══════════════════════════════════════\n"
            "ack 由两部分组成，共20-50字，**禁止问号**：\n\n"
            
            "【第一部分：具体回应】（必须有！）\n"
            "针对对方刚说的内容做出真实反应，不能空洞敷衍。\n"
            "回应方式要多样（每次换一种！不要总用同一种模式）：\n"
            "- **共鸣**：分享类似感受 →'可不是嘛，我今早出门也冻得直缩脖子。'\n"
            "- **评价**：带态度的看法 →'那多好啊，老伙计们凑一块儿最热闹了。'\n"
            "- **联想**：由此想到的事 →'饺子好啊，我最馋韭菜鸡蛋馅儿的。'\n"
            "- **感叹**：真情实感 →'七十年，一晃就过去了。'\n\n"
            
            "【第二部分：过渡语句】（关键！把前后话题连起来）\n"
            "用一句**生活常识或自然联想**把当前话题引向下一个话题。\n"
            "这句话要像真人脑子里自然冒出的念头，不是硬凑的连接词。\n\n"
            
            "✅ **好的过渡方式**（每次用不同的！）：\n"
            "① 生活常理型：用一句大家都认同的道理自然衔接\n"
            "   - 从'打麻将'→'地点'：'这大冷天的，还是找个暖和的地方打才舒服。'\n"
            "   - 从'心情好'→'天气'：'人心情好啊，跟这天儿也有关系。'\n"
            "② 个人经历型：用自己的经历带出新话题\n"
            "   - 从'吃饺子'→'计算'：'我上回买面粉还算了半天账，脑子不够使。'\n"
            "   - 从'看电视'→'星期'：'我就记着上礼拜看了个好节目。'\n"
            "③ 自然联想型：顺着对方的话自然想到相关的事\n"
            "   - 从'老朋友'→'孩子'：'老朋友聊天准少不了聊孩子们的事儿。'\n"
            "   - 从'过年'→'日期'：'这过了年啊，日子就过得快了。'\n"
            "④ 环境感知型：从周围环境/季节/时事引出\n"
            "   - 从'休息'→'天气'：'窝在家里是挺舒服，就是不知道外头冷不冷。'\n"
            "   - 从'散步'→'地点'：'散步好啊，就看在哪儿散了。'\n\n"
            
            "❌ **禁止的过渡方式**：\n"
            "- 机械连接：'说到时间''提到这个''说起来' ← 太假！\n"
            "- 空洞肯定：'XX挺好的''XX挺合适的''XX挺方便的' ← 像机器人！\n"
            "- 万能模板：每次都是'[肯定]+[道理]+[问题]' ← 千篇一律！\n"
            "- 说教口吻：'多运动对身体好' ← 别教育人家！\n\n"
            
            "═══════════════════════════════════════\n"
            "  📝 q 字段：问题要像随口聊出来的\n"
            "═══════════════════════════════════════\n"
            "**最高优先级：必须执行【任务指令】！**\n"
            "如果指令让你问B，即使刚聊的是A，也必须在ack里过渡到B再问。\n\n"
            
            "问题风格要求：\n"
            "- 像邻居串门随口问的，不是记者采访\n"
            "- 具体、能直接答的：'今早吃的啥？' 而不是 '您饮食怎么样？'\n"
            "- 口语化，加语气词：'啊/呀/呢/吧/嘛/哈'\n"
            "- 有时带入自己：'我都记不清今天星期几了，您记得不？'\n\n"
            
            "🎲 **句式要多变**（每次不同！）：\n"
            "- '哎，您...' / '那...' / '诶我问您...' / 直接问不加前缀\n"
            "- '我寻思...' / '您说...' / 用假设句/选择句\n"
            "- **禁止每句都用'对了'开头！'对了'最多出现一次！**\n"
            "- **禁止每句都加称呼！称呼隔2-3句用一次就行**\n\n"
            
            "═══════════════════════════════════════\n"
            "  🎯 完整示例（注意回应+过渡+问题都不同风格）\n"
            "═══════════════════════════════════════\n"
            "【示例1】对方说'下午3点打麻将' → 任务：问在哪儿打\n"
            '{"ack": "三点钟不早不晚，我姥爷那会儿也是下午场。这大冷天的，还得找个暖和地方。", "q": "您一般去哪儿打呀，家里还是外头？"}\n'
            "（回应：共鸣-我姥爷也这样 → 过渡：冷天要暖和 → 问地点）\n\n"
            
            "【示例2】对方说'在家打麻将' → 任务：问住在哪个区\n"
            '{"ack": "在家打多方便啊，想打多久打多久。我就好奇您住哪边。", "q": "在哪个区啊？我去过大连好几个地方。"}\n'
            "（回应：评价-方便 → 过渡：好奇住哪边 → 问区域）\n\n"
            
            "【示例3】对方说'心情还不错' → 任务：聊天气\n"
            '{"ack": "那就好，心里敞亮日子才过得舒坦。人心情好啊，跟这天儿也有关系。", "q": "今儿外头咋样啊，出太阳了没？"}\n'
            "（回应：感叹-心敞亮 → 过渡：心情和天气有关 → 问天气）\n\n"
            
            "【示例4】对方说'和老朋友聊家常' → 任务：问孩子情况\n"
            '{"ack": "老伙计们聚聚唠唠嗑，比啥都解闷儿。你们凑一块儿准少不了聊孩子的事儿。", "q": "您家孩子们呢，离得远不远？"}\n'
            "（回应：评价-解闷 → 过渡：聊天会聊孩子 → 问孩子）\n\n"
            
            "【示例5】对方说'天气晴朗不太冷' → 任务：问今天几号\n"
            '{"ack": "是嘛，晴天人精神头都不一样。这一晃又快到月底了吧。", "q": "哎，今儿几号来着？我出门急都没看日历。"}\n'
            "（回应：共鸣-精神好 → 过渡：一晃到月底 → 问日期）\n\n"
            
            "【示例6】对方说'过年打打麻将' → 任务：问日常作息\n"
            '{"ack": "过年就得乐呵乐呵，辛苦一年了嘛。不过过完年又得恢复老样子了。", "q": "平时您一般啥时候起床啊？我猜您是早起的人。"}\n'
            "（回应：感叹-辛苦一年 → 过渡：过完年恢复日常 → 问作息）\n\n"
        )
        
        user_prompt_parts = []
        
        # 聊天话题引导（内部使用，不暴露给用户）
        topic_hints = {
            '定向力': '定向力是指对时间、地点、人物的认知能力。时间维度：年月日、星期、季节、早中晚；地点维度：国家城市、具体位置、周围环境；人物维度：自己身份、家人关系、当前交流对象。可以从任意角度自然地聊，不局限于固定问法',
            '即时记忆': '即时记忆是短期记住信息的能力。可以让对方记住几个词、数字、简单指令，或者复述刚说的内容。注意要选择难度适中、生活化的内容',
            '注意力与计算': '注意力与计算考查专注力和基本运算。可以聊买菜找钱、生活中的简单计算、数数、或需要持续注意的任务。要结合生活场景，自然提出',
            '延迟回忆': '延迟回忆是在干扰后仍能记住之前信息的能力。询问之前记住的词语、事件，或者前面聊过的内容。要给出足够的提示和鼓励',
            '语言': '语言能力包括理解、表达、命名、复述等。可以聊身边物品名称、描述事物、复述句子、按指令做动作。要选择常见易懂的内容',
            '构图(临摹)': '构图能力考查视觉空间和手眼协调。可以让对方画简单图形、抄写文字、模仿动作。要从简单开始，给予充分指导'
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
            # 🔥 提取问题核心（去掉承上启下的部分，只保留问句）
            import re
            core_questions = []
            for q in all_avoid[-15:]:  # 取最近15个，范围扩大
                # 提取问号前的问句部分
                match = re.search(r'[，。]?([^，。]*[吗呢啊？?])$', q)
                if match:
                    core_questions.append(match.group(1).strip())
                else:
                    # 提取最后一句
                    last_sentence = q.split('，')[-1] if '，' in q else q
                    core_questions.append(last_sentence.strip())
            
            avoid_text = '\n  ❌ '.join(core_questions)
            user_prompt_parts.append(
                f"\n\n🚨🚨🚨【绝对禁止提问的内容】🚨🚨🚨\n"
                f"以下问题已经问过，**绝对不能再问**：\n  ❌ {avoid_text}\n\n"
                f"⚠️ 以下**变体**也绝对禁止：\n"
                f"  - '喜欢做点啥' = '喜欢干点啥' = '平时做什么'（同一个意思！）\n"
                f"  - '今年冷吗' = '今年比往年冷吗'（同一个主题！）\n"
                f"  - '看什么电视' = '看什么节目'（同一个主题！）\n"
                f"  - '吃点什么' = '吃什么饭'（同一个主题！）\n\n"
                f"⚠️ 判断标准：如果用户能用相同的话回答两个问题，那就是重复！\n"
                f"⚠️ 违反此规则是**严重错误**！必须换一个完全不同的话题！\n"
            )
        
        # 🔥 新增：自然过渡提示 - 融入 ack
        if bridge_hint:
            # 提取目标话题（格式可能是 "A→B" 或单个话题）
            target_topic = bridge_hint.split("→")[-1].strip() if "→" in bridge_hint else bridge_hint
            user_prompt_parts.append(
                f"\n🎯🎯🎯【话题约束 - 最高优先级】🎯🎯🎯\n"
                f"目标话题：「{target_topic}」\n"
                f"⚠️ **q（问题）必须直接关于「{target_topic}」**，不能只是沾边！\n"
                f"  - ✅ 正确：话题是'季节' → 问'您喜欢哪个季节？' / '这冬天冷不冷？'\n"
                f"  - ❌ 错误：话题是'季节' → 问'这个月有什么打算？'（问的是月份计划，不是季节！）\n"
                f"  - ✅ 正确：话题是'星期' → 问'今天星期几？' / '周末去哪玩？'\n"
                f"  - ❌ 错误：话题是'星期' → 问'最近忙不忙？'（问的是忙碌程度，不是星期！）\n\n"
                f"【过渡方法】在 ack 里自然过渡到「{target_topic}」：\n"
                f"  - 用户说'7点起床' + 话题'星期' → ack里:'7点起挺规律，一天天的' → 问周几\n"
                f"  - 用户说'冬天冷' + 话题'地点' → ack里:'冷是冷，咱这边更明显' → 问住哪\n"
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
        
        if conversation_history:
            # 🔥 构建结构化历史上下文 + 风格防重复
            history_list = conversation_history if isinstance(conversation_history, list) else []
            
            last_user_msg = ""
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
                
                # 构建最近3轮对话展示（精简版，节省token）
                recent = history_list[-6:]
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
            response = self._llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
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
                
                print(f"[QuestionGenTool] ✅ JSON解析成功: ack='{ack}', q='{q[:30]}...'")
                
                # 🆕 校验：如果 ack 包含问号，移除问句部分，只保留陈述部分
                if ack and ("？" in ack or "?" in ack):
                    # 在问号前截断，只保留陈述部分
                    ack_parts = re.split(r'[？?]', ack)
                    ack = ack_parts[0].strip()
                    # 如果截断后太短，使用简单回应
                    if len(ack) < 3:
                        ack = "嗯嗯"
                
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
            
            response = self._llm.invoke([
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

