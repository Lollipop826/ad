"""
问题生成器

基于检索到的知识、用户画像、历史对话和情绪，生成下一个询问用户的问题
"""

from typing import List, Dict, Any, Optional
import os
import logging

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.common.types import InfoDimension, Profile
from src.llm.http_client_pool import get_siliconflow_chat_openai


class QuestionGeneratorConfig(BaseModel):
    """问题生成器配置"""
    model: str = Field(default=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-14B-Instruct"))
    base_url: Optional[str] = Field(default=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"))
    api_key: Optional[str] = Field(default=os.getenv("SILICONFLOW_API_KEY"))
    temperature: float = 0.7
    max_tokens: int = 300
    
    system_prompt: str = (
        "你是一位专业的阿尔茨海默病初筛评估医生。你的任务是基于MMSE（简易精神状态检查）量表，"
        "通过对话方式评估患者的认知功能。\n\n"
        "你的角色：\n"
        "- 你是一位温和、耐心、专业的医生\n"
        "- 你需要根据当前评估的维度（如定向力、记忆力、注意力等），结合检索到的专业知识，"
        "  设计合适的问题来评估患者\n"
        "- 问题应该自然、友好，让患者感到舒适\n"
        "- 根据患者的年龄、教育水平调整问题的难度和表达方式\n"
        "- 参考检索到的知识中的评估方法和症状描述\n\n"
        "问题设计原则：\n"
        "1. 一次只问一个问题，不要问多个问题\n"
        "2. 【极其重要】严禁重复已经问过的问题！必须仔细检查历史对话，确认没有问过相同或相似的问题\n"
        "3. 如果患者已经回答过某个问题（如日期、地点），绝对不能再问一次\n"
        "4. 问题要具体、明确，便于患者回答\n"
        "5. 语气要温和、鼓励性的\n"
        "6. 如果患者上次回答显示焦虑或紧张情绪，要更加温和\n"
        "7. 问题应该针对当前评估的维度\n"
        "8. 可以参考专业知识中的评估方法，但要用患者能理解的语言\n\n"
        "输出要求：\n"
        "- 直接输出要问患者的问题\n"
        "- 不要输出解释、分析或其他内容\n"
        "- 问题长度控制在50字以内"
    )


class QuestionGenerator:
    """问题生成器"""
    
    def __init__(self, config: Optional[QuestionGeneratorConfig] = None):
        self.config = config or QuestionGeneratorConfig()
        
        # Logger setup
        self.logger = logging.getLogger("dialogue.question_generator")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # LLM
        self.llm = get_siliconflow_chat_openai(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=30,
            max_retries=1,
        )
    
    def generate_question(
        self,
        dimension: InfoDimension,
        retrieved_knowledge: List[str],
        profile: Optional[Profile] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        last_emotion: Optional[str] = None,
    ) -> str:
        """
        生成下一个问题
        
        Args:
            dimension: 当前评估的维度
            retrieved_knowledge: 检索到的相关知识（最多3-5条）
            profile: 用户画像
            conversation_history: 最近的对话历史（最多5轮）
            last_emotion: 用户上次的情绪
        
        Returns:
            生成的问题
        """
        
        # 构建用户提示
        user_prompt_parts = []
        
        # 1. 当前评估维度
        user_prompt_parts.append(f"【当前评估维度】")
        user_prompt_parts.append(f"维度名称: {dimension.get('name', '未知')}")
        user_prompt_parts.append(f"维度描述: {dimension.get('description', '无')}")
        user_prompt_parts.append(f"优先级: {dimension.get('priority', 0)}")
        user_prompt_parts.append(f"当前状态: {dimension.get('status', 'unknown')}")
        user_prompt_parts.append("")
        
        # 2. 用户画像
        if profile:
            user_prompt_parts.append(f"【患者信息】")
            user_prompt_parts.append(f"年龄: {profile.get('age', '未知')}岁")
            user_prompt_parts.append(f"性别: {profile.get('sex', '未知')}")
            user_prompt_parts.append(f"教育年限: {profile.get('education_years', '未知')}年")
            if profile.get('notes'):
                user_prompt_parts.append(f"备注: {profile.get('notes')}")
            user_prompt_parts.append("")
        
        # 3. 检索到的专业知识
        if retrieved_knowledge:
            user_prompt_parts.append(f"【相关专业知识】")
            for i, knowledge in enumerate(retrieved_knowledge[:3], 1):  # 最多3条
                # 截断过长的知识
                knowledge_text = knowledge[:400] + "..." if len(knowledge) > 400 else knowledge
                user_prompt_parts.append(f"{i}. {knowledge_text}")
            user_prompt_parts.append("")
        
        # 4. 对话历史
        if conversation_history and len(conversation_history) > 0:
            user_prompt_parts.append(f"【最近对话】")
            # 只显示最近3轮
            recent_history = conversation_history[-6:]  # 每轮2条（user+assistant）
            for turn in recent_history:
                role = "医生" if turn["role"] == "assistant" else "患者"
                user_prompt_parts.append(f"{role}: {turn['content']}")
            user_prompt_parts.append("")
        
        # 5. 用户情绪
        if last_emotion:
            user_prompt_parts.append(f"【患者上次情绪】{last_emotion}")
            user_prompt_parts.append("")
        
        # 6. 任务说明
        user_prompt_parts.append("【任务】")
        user_prompt_parts.append(
            f"请基于以上信息，设计一个问题来评估患者的【{dimension.get('name')}】。"
        )
        
        # 根据不同维度添加特定要求
        dimension_id = dimension.get('id', '')
        if dimension_id == 'registration':
            user_prompt_parts.append("重要：这是即时记忆测试，必须给出3个具体的词让患者重复（如：苹果、桌子、硬币）。")
        elif dimension_id == 'attention_calculation':
            user_prompt_parts.append("重要：这是计算能力测试，必须给出具体的起始数字和运算规则（如：从100开始，每次减7）。")
        elif dimension_id == 'recall':
            user_prompt_parts.append("重要：这是延迟回忆测试，要求患者回忆之前记住的3个词。")
        
        user_prompt_parts.append("\n【极其重要的检查清单】")
        user_prompt_parts.append("1. 检查历史对话：患者是否已经回答过类似的问题？")
        user_prompt_parts.append("2. 如果患者已回答过日期、地点、年份等，绝对不能再问")
        user_prompt_parts.append("3. 如果是同一维度的后续问题，应该问不同的子项（如定向力：日期→地点→楼层）")
        user_prompt_parts.append("\n直接输出要问患者的问题，不要有其他内容。")
        
        user_prompt = "\n".join(user_prompt_parts)
        
        self.logger.info(f"生成问题 - 维度: {dimension.get('name')}")
        self.logger.debug(f"Prompt:\n{user_prompt}")
        
        # 调用LLM
        try:
            response = self.llm.invoke([
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            question = response.content.strip()
            
            # 清理输出
            question = self._clean_question(question)
            
            self.logger.info(f"生成的问题: {question}")
            return question
            
        except Exception as e:
            self.logger.error(f"生成问题失败: {e}")
            # 返回一个默认问题
            return self._get_fallback_question(dimension)
    
    def _clean_question(self, question: str) -> str:
        """清理生成的问题"""
        # 移除可能的引号
        question = question.strip('"').strip("'").strip()
        
        # 移除可能的前缀（如"问题："、"医生："等）
        prefixes = ["问题：", "问题:", "医生：", "医生:", "Question:", "Q:"]
        for prefix in prefixes:
            if question.startswith(prefix):
                question = question[len(prefix):].strip()
        
        # 确保以问号结尾
        if not question.endswith(("?", "？", "。", ".")):
            question += "？"
        
        return question
    
    def _get_fallback_question(self, dimension: InfoDimension) -> str:
        """获取备用问题"""
        fallback_questions = {
            "orientation": "您能告诉我今天是几月几号吗？",
            "registration": "我说三个词，请您重复一遍：苹果、桌子、硬币。",
            "attention_calculation": "请您从100开始，每次减7，连续减5次。",
            "recall": "您还记得我刚才让您记住的三个词是什么吗？",
            "language": "请您看看这支笔，这个叫什么？",
            "copy": "请您照着这个图形画一遍。"
        }
        
        dim_id = dimension.get("id", "")
        return fallback_questions.get(dim_id, f"请您描述一下您的{dimension.get('name', '情况')}。")

