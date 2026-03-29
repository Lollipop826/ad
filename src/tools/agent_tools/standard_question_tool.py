"""
特殊维度问题工具 - 用 LLM 生成自然问题 + 结构化输出

处理 MMSE 中需要特殊处理的维度：
- registration (即时记忆): LLM 生成问题，提取记忆词
- attention_calculation (注意力计算): LLM 生成问题，提取计算参数
- recall (延迟回忆): 基于之前的记忆词生成问题
- copy (临摹): 生成引导语 + 展示图片
"""

import json
import os
import re
import random
from typing import Optional, Type, Dict, Any, List
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI


# ==================== 维度配置 ====================
SPECIAL_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    # 原有标准化任务
    'registration': {
        'trigger': 'on_switch',
        'purpose': '测试即时记忆能力',
        'log_tag': '📝 即时记忆',
        'default_words': ['苹果', '桌子', '硬币'],
        'word_pools': [
            ['苹果', '桌子', '硬币'],
            ['香蕉', '椅子', '钥匙'],
            ['西瓜', '电话', '手表'],
            ['橘子', '窗户', '雨伞'],
        ],
    },
    'attention_calculation': {
        'trigger': 'always',
        'purpose': '测试注意力和计算能力',
        'log_tag': '🔢 注意力计算',
        'default_config': {'start': 100, 'step': 7},
    },
    'recall': {
        'trigger': 'on_switch',
        'purpose': '测试延迟记忆能力',
        'log_tag': '🧠 延迟回忆',
    },
    'copy': {
        'trigger': 'always',
        'purpose': '测试视觉空间构造能力',
        'log_tag': '📋 临摹',
        'requires_image': True,
        'image_id': 'pentagons',
        'image_title': '请看这两个图形，试着在纸上把它们画下来',
    },
    # 🔥 语言任务（复述句子）
    'language_repetition': {
        'trigger': 'on_switch',
        'purpose': '测试语言复述能力',
        'log_tag': '🗣️ 复述句子',
        'standard_sentence': '四十四只石狮子',  # 标准复述句子
        'instruction': '请跟我说一遍这句话："四十四只石狮子"',
    },
}


class StandardQuestionToolArgs(BaseModel):
    """特殊维度问题工具参数"""
    dimension_id: str = Field(
        ..., 
        description="维度ID: registration, attention_calculation, recall, copy, orientation, language"
    )
    is_dimension_switch: bool = Field(
        default=False,
        description="是否刚切换到此维度"
    )
    memory_words: Optional[List[str]] = Field(
        default=None,
        description="之前记忆的词（用于 recall 维度）"
    )
    patient_name: Optional[str] = Field(
        default=None,
        description="患者姓名（用于个性化）"
    )
    calculation_current_value: Optional[int] = Field(
        default=None,
        description="连续减法的当前值（用于 attention_calculation）"
    )
    calculation_step: Optional[int] = Field(
        default=7,
        description="连续减法的步长（默认7）"
    )
    last_user_message: Optional[str] = Field(
        default=None,
        description="用户上一轮说的话，用于生成自然的回应过渡"
    )


class StandardQuestionTool(BaseTool):
    """
    特殊维度问题生成工具
    
    用 LLM 生成自然的问题，同时返回结构化的关键数据。
    确保问题自然流畅，同时关键信息可追踪。
    """
    
    name: str = "StandardQuestionTool"
    description: str = """
    为 MMSE 特殊维度生成自然的问题。
    
    与普通问题生成不同，这个工具会：
    1. 生成自然流畅的问题
    2. 返回结构化的关键数据（如记忆词、计算参数）
    3. 确保 recall 时使用的词与 registration 一致
    """
    args_schema: Type[BaseModel] = StandardQuestionToolArgs
    
    _llm: Any = PrivateAttr()
    _use_local: bool = PrivateAttr()
    
    def __init__(self, use_local: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._use_local = use_local
        
        if use_local:
            # 使用本地模型池
            from src.llm.model_pool import get_pooled_llm
            self._llm = get_pooled_llm(pool_key='7b_complex')  # 用 7B 模型生成自然问题
            print("[StandardQuestionTool] 🏠 使用本地模型 (7b_complex)")
        else:
            # 使用 API
            import os as _os
            if _os.getenv("ARK_API_KEY"):
                from src.llm.http_client_pool import get_volcengine_chat_openai
                self._llm = get_volcengine_chat_openai(
                    model="doubao-seed-2-0-mini-260215",
                    temperature=0.7,
                    max_tokens=200,
                    timeout=10,
                    max_retries=1,
                )
                print("[StandardQuestionTool] 🌋 使用火山引擎")
            else:
                from src.llm.http_client_pool import get_siliconflow_chat_openai
                self._llm = get_siliconflow_chat_openai(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    temperature=0.7,
                    max_tokens=200,
                    timeout=10,
                    max_retries=1,
                )
                print("[StandardQuestionTool] 🔵 使用 SiliconFlow")
    
    def _extract_response(self, response) -> str:
        """从 LLM 响应中提取文本（兼容本地模型和 API）"""
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def _run(
        self,
        dimension_id: str,
        is_dimension_switch: bool = False,
        memory_words: Optional[List[str]] = None,
        patient_name: Optional[str] = None,
        calculation_current_value: Optional[int] = None,
        calculation_step: Optional[int] = 7,
        last_user_message: Optional[str] = None,
    ) -> str:
        """
        生成特殊维度的问题
        
        Returns:
            JSON 格式，包含 question 和结构化数据
        """
        
        # 检查是否是特殊维度
        if dimension_id not in SPECIAL_DIMENSIONS:
            return json.dumps({
                "has_standard_question": False,
                "dimension_id": dimension_id,
                "message": "此维度使用普通问题生成"
            }, ensure_ascii=False)
        
        config = SPECIAL_DIMENSIONS[dimension_id]
        trigger = config.get('trigger', 'on_switch')
        
        # 检查触发条件
        should_trigger = (trigger == 'always') or (trigger == 'on_switch' and is_dimension_switch)
        
        if not should_trigger:
            return json.dumps({
                "has_standard_question": False,
                "dimension_id": dimension_id,
                "message": "不满足触发条件（非首次进入维度）"
            }, ensure_ascii=False)
        
        print(f"[StandardQuestion] {config['log_tag']} - 生成问题...")
        
        # 根据维度类型生成问题
        if dimension_id == 'registration':
            return self._generate_registration_question(config, patient_name, last_user_message)
        elif dimension_id == 'attention_calculation':
            return self._generate_calculation_question(config, patient_name, calculation_current_value, calculation_step, last_user_message)
        elif dimension_id == 'recall':
            return self._generate_recall_question(config, memory_words, patient_name, last_user_message)
        elif dimension_id == 'copy':
            return self._generate_copy_question(config, patient_name)
        elif dimension_id == 'language_repetition':
            # 🔥 语言复述任务：返回固定句子
            return self._generate_repetition_question(config, patient_name)
        elif dimension_id in ['orientation', 'language']:
            # 新的灵活任务：交给QuestionGenerationTool处理
            return json.dumps({
                "has_standard_question": False,
                "dimension_id": dimension_id,
                "message": f"灵活任务{dimension_id}应由QuestionGenerationTool处理"
            }, ensure_ascii=False)
        
        return json.dumps({"has_standard_question": False}, ensure_ascii=False)
    
    def _generate_registration_question(self, config: dict, patient_name: Optional[str], last_user_message: Optional[str] = None) -> str:
        """生成即时记忆问题 - LLM 自主选择三个词"""
        
        context_info = f"患者叫{patient_name}。" if patient_name else ""
        
        # 构建对话上下文
        ack_instruction = ""
        if last_user_message and last_user_message.strip():
            ack_instruction = f'\n5. 对方刚说了「{last_user_message.strip()}」，question 的开头必须先简短回应这句话（最多15字，引用对方说的具体内容），然后自然过渡到让对方记词。例如：「一年半了呀，住挺久的。对了我说三个词……」'
        
        prompt = f"""你是一位温柔的老年科医生，正在和老人聊天。{context_info}
现在需要让老人记住三个词并复述。

要求：
1. **自主选择三个词**，必须满足：
   - 互不相关（不同类别，如：水果、家具、交通工具）
   - 非常常见、具体、易懂（避免抽象词）
   - 必须是中文双字词（如：苹果、桌子、硬币）
2. 用温和亲切的语气引导，如果是长辈可以用"您"
3. 清晰地说出这三个词，并让老人复述一遍
4. **仅输出JSON格式**，不要包含Markdown标记或其他文字{ack_instruction}

好的词例：
- 苹果、桌子、硬币
- 香蕉、椅子、钥匙
- 西瓜、电话、手表

请严格按此JSON格式回复：
{{"question": "生成的问句", "words": ["词1", "词2", "词3"]}}"""

        # 默认回退值
        default_words = config.get('default_words', ['苹果', '桌子', '硬币'])
        words_str = '、'.join(default_words)
        fallback_question = f"咱们来个小游戏，我说三个词，您听好了跟着说一遍：{words_str}。您来说一遍？"
        
        # 🔥 构建 ack 前缀（用于 fallback 模板也能带上回应）
        _ack_prefix = ""
        if last_user_message and last_user_message.strip():
            _msg = last_user_message.strip().rstrip('。.！!？?')
            # 提取关键短语做简短回应（不照搬原话）
            import re as _re
            # 尝试提取数字+量词短语（如"一年半"、"5楼"、"93块"）
            num_match = _re.search(r'[\d一二三四五六七八九十百千]+[\w]*[年月天楼层岁个块钱][\w]{0,2}', _msg)
            if num_match:
                phrase = num_match.group().rstrip('了吧呢啊')
                _ack_prefix = f"{phrase}呀。对了，"
            elif len(_msg) <= 8:
                _ack_prefix = f"嗯呐。对了，"
            else:
                _ack_prefix = f"嗯呐。对了，"

        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            response_text = self._extract_response(response).strip()
            
            # 清理 Markdown 代码块
            if "```" in response_text:
                import re
                match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
            
            # 尝试解析 JSON
            words = None
            question = None
            
            import re
            # 寻找最外层的 {} 
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    question = parsed.get('question', '').strip()
                    words = parsed.get('words', [])
                except json.JSONDecodeError:
                    print(f"[StandardQuestion] ⚠️ JSON解析错误: {response_text[:50]}...")
            
            # 验证数据的有效性
            if not words or not isinstance(words, list) or len(words) != 3:
                print(f"[StandardQuestion] ⚠️ 生成的词无效: {words}，使用备选词池")
                words = random.choice(config.get('word_pools', [default_words]))
                words_str = '、'.join(words)
                # 重新构建问题以确保包含这些词
                if patient_name:
                    question = f"{_ack_prefix}{patient_name}，我说三个词，您听好了：{words_str}。请您复述一遍？"
                else:
                    question = f"{_ack_prefix}我说三个词，您听好了：{words_str}。请您复述一遍？"
            
            # 确保词语在问题中
            if question and not all(w in question for w in words):
                words_str = '、'.join(words)
                if patient_name:
                    question = f"{_ack_prefix}{patient_name}，我说三个词，您听好了：{words_str}。请您复述一遍？"
                else:
                    question = f"{_ack_prefix}我说三个词，您听好了：{words_str}。请您复述一遍？"
            
            print(f"[StandardQuestion] ✅ 生成问题: {question[:50]}...")
            print(f"[StandardQuestion] 📝 LLM选择的词: {words}")
            
            return json.dumps({
                "has_standard_question": True,
                "dimension_id": "registration",
                "question": question,
                "memory_words": words,  # ⭐ 关键：返回记忆词，供 recall 使用
                "log_tag": config['log_tag'],
                "requires_image": False,
            }, ensure_ascii=False)
            
        except Exception as e:
            # ❗ 调试模式：不用兆底，直接报错
            print(f"[StandardQuestion] ❌ LLM 生成失败: {e}")
            raise e  # 让错误暴露出来
    
    def _generate_calculation_question(
        self, 
        config: dict, 
        patient_name: Optional[str],
        current_value: Optional[int] = None,
        step: Optional[int] = 7,
        last_user_message: Optional[str] = None
    ) -> str:
        """
        生成注意力计算问题（支持连续减法状态跟踪）
        
        Args:
            current_value: 当前数值。None 表示第一轮（使用100）；否则使用传入值
            step: 每次减去的数（默认7）
        """
        
        # 确定当前要问的数值
        if current_value is None:
            # 第一轮：从100开始
            start = 100
            is_first_round = True
        else:
            # 后续轮次：使用传入的当前值（这是期望的答案）
            start = current_value
            is_first_round = False
        
        step = step or 7
        expected_answer = start - step  # 期望答案
        
        context_info = f"患者叫{patient_name}。" if patient_name else ""
        
        ack_part = ""
        if is_first_round and last_user_message and last_user_message.strip():
            ack_part = f'\n- 对方刚说了「{last_user_message.strip()}」，开头先简短回应（最多15字），再自然过渡到算术场景。'
        
        if is_first_round:
            prompt = f"""你是老人的晚辈，在轻松唠嗑。{context_info}
用一个生活买菜/买东西的场景，自然地问出"{start}减{step}等于多少"。
不要说"咱们做道算术题"，要像聊家常一样带出来。

要求：一句话，口语化，必须包含数字{start}和{step}。仅输出JSON。{ack_part}
示例：
- "对了，要是您拿{start}块钱去买菜，花了{step}块，还剩多少钱呀？"
- "话说您要有{start}块钱，买个{step}块的西瓜，兜里还剩多少？"

{{"question": "生成的问句"}}"""
        else:
            prompt = f"""你是老人的晚辈，在陪老人接着算买菜的账。{context_info}
上一轮老人算出还剩{start}块钱，现在继续问：再花{step}块，还剩多少？
用口语，简短接话，一句话。仅输出JSON。

示例：
- "还剩{start}块呢，再花{step}块买点鸡蛋，还剩多少？"
- "{start}块钱，又花了{step}块，还剩多少呀？"

{{"question": "生成的问句"}}"""

        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            response_text = self._extract_response(response).strip()
            
            # 清理 Markdown
            if "```" in response_text:
                import re
                match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
            
            question = None
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    question = parsed.get('question', '').strip()
                except:
                    pass
            
            # Fallback if JSON fails or question is empty
            if not question:
                question = response_text.strip().strip('"').strip("'")
                if len(question) > 50 or "{" in question:
                    question = None

            # 验证关键信息
            if question:
                if str(start) not in question or str(step) not in question:
                    if is_first_round:
                        question = f"对了，要是您拿{start}块钱去买菜，花了{step}块，还剩多少钱呀？"
                    else:
                        question = f"{start}块钱，再花{step}块买点东西，还剩多少呀？"
            else:
                if is_first_round:
                    question = f"对了，要是您拿{start}块钱去买菜，花了{step}块，还剩多少钱呀？"
                else:
                    question = f"{start}块钱，再花{step}块买点东西，还剩多少呀？"
            
            print(f"[StandardQuestion] ✅ 生成问题: {question} (期望答案: {expected_answer})")
            
            return json.dumps({
                "has_standard_question": True,
                "dimension_id": "attention_calculation",
                "question": question,
                "calculation_config": {
                    "start": start,
                    "step": step,
                    "expected_answer": expected_answer,  # ⭐ 返回期望答案，用于下一轮
                },
                "log_tag": config['log_tag'],
                "requires_image": False,
            }, ensure_ascii=False)
            
        except Exception as e:
            print(f"[StandardQuestion] ❌ LLM 生成失败: {e}")
            raise e
    
    def _generate_recall_question(
        self, config: dict, memory_words: Optional[List[str]], patient_name: Optional[str],
        last_user_message: Optional[str] = None
    ) -> str:
        """生成延迟回忆问题"""
        
        if not memory_words:
            # 没有记忆词，用默认的
            memory_words = ['苹果', '桌子', '硬币']
            print("[StandardQuestion] ⚠️ 没有传入 memory_words，使用默认词")
        
        context_info = f"患者叫{patient_name}。" if patient_name else ""
        
        ack_instruction = ""
        if last_user_message and last_user_message.strip():
            ack_instruction = f'\n5. 对方刚说了「{last_user_message.strip()}」，开头先简短回应（最多15字），再自然过渡到问记忆词。'
        
        prompt = f"""你是一位温柔的老年科医生，正在和老人聊天。{context_info}
之前让老人记住了三个词：{', '.join(memory_words)}
现在需要问老人是否还记得这三个词。

要求：
1. 用温和的语气询问，不要给老人压力
2. **绝对不要**直接说出那三个词，而是问"刚才那三个词"或"刚才让您记的东西"
3. 给老人鼓励，让他们尝试回忆
4. **仅输出JSON格式**{ack_instruction}

示例风格：
- "诶，刚才我说的那三个词，您还能想起来吗？"
- "您还记得刚才让您记的那三样东西吗？试试看～"

请严格按此JSON格式回复：
{{"question": "生成的问句"}}"""

        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            response_text = self._extract_response(response).strip()
            
            # 清理 Markdown
            if "```" in response_text:
                import re
                match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
            
            question = None
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    question = parsed.get('question', '').strip()
                except:
                    pass
            
            # Fallback if JSON fails or question is empty
            if not question:
                question = response_text.strip().strip('"').strip("'")
                if len(question) > 50 or "{" in question:
                    question = None
            
            # 确保问题中没有直接说出答案（防止泄漏）
            if question and any(w in question for w in memory_words):
                print(f"[StandardQuestion] ⚠️ 生成的问题泄漏了答案，使用默认问题")
                if patient_name:
                    question = f"{patient_name}，刚才我说的那三个词，您还能想起来吗？试试看～"
                else:
                    question = "诶，刚才我说的那三个词，您还能想起来吗？试试看～"
            
            # 最终兜底
            if not question:
                if patient_name:
                    question = f"{patient_name}，刚才我说的那三个词，您还能想起来吗？试试看～"
                else:
                    question = "诶，刚才我说的那三个词，您还能想起来吗？试试看～"
            
            print(f"[StandardQuestion] ✅ 生成问题: {question[:50]}...")
            
            return json.dumps({
                "has_standard_question": True,
                "dimension_id": "recall",
                "question": question,
                "expected_words": memory_words,  # ⭐ 返回期望的答案，用于评分
                "log_tag": config['log_tag'],
                "requires_image": False,
            }, ensure_ascii=False)
            
        except Exception as e:
            # ❗ 调试模式：不用兆底，直接报错
            print(f"[StandardQuestion] ❌ LLM 生成失败: {e}")
            raise e
    
    def _generate_repetition_question(self, config: dict, patient_name: Optional[str]) -> str:
        """生成语言复述问题 - 固定句子"""
        
        sentence = config.get('standard_sentence', '四十四只石狮子')
        instruction = config.get('instruction', f'请跟我说一遍："{sentence}"')
        
        # 根据患者名字生成自然的引导语
        if patient_name:
            intro = f"{patient_name}，咱们来玩个小游戏，我说一句话，您跟着说一遍好不好？"
        else:
            intro = "咱们来玩个小游戏，我说一句话，您跟着说一遍好不好？"
        
        question = f'{intro}听好了："{sentence}"'
        
        return json.dumps({
            "has_standard_question": True,
            "dimension_id": "language_repetition",
            "question": question,
            "expected_answer": sentence,
            "data": {
                "standard_sentence": sentence,
                "task_type": "repetition"
            }
        }, ensure_ascii=False)
    
    def _generate_copy_question(self, config: dict, patient_name: Optional[str]) -> str:
        """生成临摹问题"""
        
        context_info = f"患者叫{patient_name}。" if patient_name else ""
        
        prompt = f"""你是一位温柔的老年科医生，正在和老人聊天。{context_info}
现在需要让老人看屏幕上的图形，然后在纸上画下来。

要求：
1. 用轻松的语气引导
2. 必须明确说明"看屏幕上的图形"
3. 必须明确说明"在纸上画下来"
4. 一句话，不要太长
5. **仅输出JSON格式**

示例风格：
- "接下来咱们画个画儿，您看屏幕上这两个图形，能在纸上画一下吗？"
- "来，您看看屏幕，试着把这两个图形画在纸上，画好了告诉我～"

请严格按此JSON格式回复：
{{"question": "生成的问句"}}"""

        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            response_text = self._extract_response(response).strip()
            
            # 清理 Markdown
            if "```" in response_text:
                import re
                match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
                if match:
                    response_text = match.group(1).strip()
            
            question = None
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    question = parsed.get('question', '').strip()
                except:
                    pass
            
            # Fallback
            if not question:
                question = response_text.strip().strip('"').strip("'")
                if len(question) > 50 or "{" in question:
                    question = None
            
            if not question:
                if patient_name:
                    question = f"{patient_name}，接下来咱们画个画儿，您看屏幕上这两个图形，能在纸上画一下吗？"
                else:
                    question = "接下来咱们画个画儿，您看屏幕上这两个图形，能在纸上画一下吗？"
            
            print(f"[StandardQuestion] ✅ 生成问题: {question[:50]}...")
            
            return json.dumps({
                "has_standard_question": True,
                "dimension_id": "copy",
                "question": question,
                "log_tag": config['log_tag'],
                "requires_image": True,
                "image_config": {
                    "image_id": config.get('image_id', 'pentagons'),
                    "title": config.get('image_title', '请看这两个图形'),
                },
            }, ensure_ascii=False)
            
        except Exception as e:
            # ❗ 调试模式：不用兆底，直接报错
            print(f"[StandardQuestion] ❌ LLM 生成失败: {e}")
            raise e
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("StandardQuestionTool 不支持异步调用")
