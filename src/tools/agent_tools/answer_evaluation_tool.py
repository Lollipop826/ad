"""
回答评估工具 - 评估患者回答的正确性和质量

简化版：移除未使用的字段，统一代码风格
"""

from __future__ import annotations

import os
import json
import re
import time
from typing import Optional, Type, Dict, Any, ClassVar

from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

from src.utils.location_service import get_realtime_context
from src.llm.http_client_pool import get_siliconflow_chat_openai


class AnswerEvaluationToolArgs(BaseModel):
    """回答评估工具参数"""
    
    question: str = Field(..., description="医生提出的问题")
    answer: str = Field(default="", description="患者的回答")
    task_id: str = Field(default="", description="任务ID: orientation_time_weekday / registration_3words / attention_calc_life_math 等")
    expected_answer: Optional[str] = Field(default=None, description="期望的正确答案")
    patient_profile: Optional[Dict[str, Any]] = Field(default=None, description="患者信息")


class AnswerEvaluationTool(BaseTool):
    """
    回答评估工具
    
    评估患者回答的正确性，输出质量等级和认知表现。
    """
    
    name: str = "answer_evaluation_tool"
    description: str = "评估患者回答的质量和认知表现，返回是否正确、质量等级、认知表现等"
    args_schema: Type[BaseModel] = AnswerEvaluationToolArgs
    
    # 任务信息（包含评估提示）
    TASK_INFO: ClassVar[Dict[str, Dict[str, str]]] = {
        # 定向力任务
        "orientation_time_weekday": {
            "name": "星期几", 
            "desc": "询问今天星期几",
            "eval_hint": "判断患者说的星期几是否与当前实际日期一致。允许口语化表达如'周五'='星期五'"
        },
        "orientation_time_date_month_season": {
            "name": "日期/季节", 
            "desc": "询问几月几号、什么季节",
            "eval_hint": "判断日期、月份、季节是否正确。日期误差1天内可接受，季节必须正确"
        },
        "orientation_place_city_district": {
            "name": "地点", 
            "desc": "询问所在城市、区域",
            "eval_hint": "判断城市、区域是否与参考信息一致。省略'市/区'字样可接受"
        },
        # 记忆任务
        "registration_3words": {
            "name": "即时记忆", 
            "desc": "让患者复述三个词",
            "eval_hint": "判断患者是否准确复述了医生说的三个词。顺序可不同，但词必须完全匹配"
        },
        "recall_3words": {
            "name": "延迟回忆", 
            "desc": "回忆之前的三个词",
            "eval_hint": "判断患者回忆出了几个词。每答对一个词算部分正确，三个全对为完全正确"
        },
        # 注意力计算
        "attention_calc_life_math": {
            "name": "连续减法", 
            "desc": "100-7连续减法",
            "eval_hint": "判断计算结果是否正确。100-7=93, 93-7=86, 86-7=79, 79-7=72, 72-7=65"
        },
        # 语言任务
        "language_naming_watch": {
            "name": "命名-手表", 
            "desc": "说出手表的名称",
            "eval_hint": "判断是否说出'手表/表'。说'钟表/时钟'部分正确，说其他物品名称错误"
        },
        "language_naming_pencil": {
            "name": "命名-铅笔", 
            "desc": "说出铅笔的名称",
            "eval_hint": "判断是否说出'铅笔/笔'。说'钢笔/圆珠笔'部分正确，说其他物品名称错误"
        },
        "language_repetition_sentence": {
            "name": "复述句子", 
            "desc": "复述绕口令或句子",
            "eval_hint": "判断是否准确复述原句。允许轻微口音差异，但字词必须基本一致"
        },
        "language_reading_close_eyes": {
            "name": "阅读指令", 
            "desc": "阅读并执行'闭眼'指令",
            "eval_hint": "判断患者是否读出了文字内容或做出了闭眼动作。执行动作即为正确"
        },
        "language_3step_action": {
            "name": "三步指令", 
            "desc": "执行三步动作指令",
            "eval_hint": "判断是否按顺序完成三步动作。每完成一步算部分正确，全完成为完全正确"
        },
        # 构图
        "copy_pentagons": {
            "name": "临摹", 
            "desc": "临摹五边形图形",
            "eval_hint": "判断临摹图形是否包含两个五边形且有交叠。形状近似即可，不要求完美"
        },
    }
    
    _llm: ChatOpenAI = PrivateAttr()
    
    def __init__(
        self,
        use_local: bool = False,
        llm_instance = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if use_local and llm_instance:
            print("[AnswerEvalTool] 🚀 使用传入的本地模型")
            self._llm = llm_instance
        elif use_local:
            print("[AnswerEvalTool] 🚀 使用模型池（7B）")
            from src.llm.model_pool import get_pooled_llm
            self._llm = get_pooled_llm(pool_key='eval_long')
        else:
            print("[AnswerEvalTool] 🌐 使用 API")
            self._llm = get_siliconflow_chat_openai(
                model=os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
                temperature=0.1,
                timeout=20,
                max_retries=1,
            )
    
    def _extract_response(self, response) -> str:
        """从 LLM 响应中提取文本"""
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    
    def _parse_truncated_json(self, content: str) -> Dict:
        """从被截断的 JSON 中提取可用字段"""
        result = {}
        
        # 提取 is_correct
        is_correct_match = re.search(r'"is_correct"\s*:\s*(true|false)', content, re.IGNORECASE)
        if is_correct_match:
            result["is_correct"] = is_correct_match.group(1).lower() == "true"
        
        # 提取 quality_level
        quality_match = re.search(r'"quality_level"\s*:\s*"(\w+)"', content)
        if quality_match:
            result["quality_level"] = quality_match.group(1)
        
        # 提取 cognitive_performance
        cognitive_match = re.search(r'"cognitive_performance"\s*:\s*"([^"]+)"', content)
        if cognitive_match:
            result["cognitive_performance"] = cognitive_match.group(1)
        
        # 提取 is_complete
        complete_match = re.search(r'"is_complete"\s*:\s*(true|false)', content, re.IGNORECASE)
        if complete_match:
            result["is_complete"] = complete_match.group(1).lower() == "true"
        
        # 提取 evaluation_detail
        detail_match = re.search(r'"evaluation_detail"\s*:\s*"([^"]*)', content)
        if detail_match:
            result["evaluation_detail"] = detail_match.group(1)
        
        # 提取 confidence
        confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', content)
        if confidence_match:
            try:
                result["confidence"] = float(confidence_match.group(1))
            except:
                pass
        
        print(f"[AnswerEvalTool] ✅ 从截断 JSON 中提取到: {result}")
        return result
    
    def _run(
        self,
        question: str,
        answer: str = "",
        task_id: str = "",
        expected_answer: Optional[str] = None,
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """执行回答评估"""
        
        start_time = time.time()
        
        # 容错：无回答
        if not answer or not answer.strip():
            print(f"[AnswerEvalTool] ⚠️ 无回答")
            return json.dumps({
                "is_correct": False,
                "quality_level": "poor",
                "cognitive_performance": "中度异常",
                "is_complete": False,
                "evaluation_detail": "患者未作回答",
                "confidence": 1.0
            }, ensure_ascii=False)
        
        # 获取任务信息
        task_info = self.TASK_INFO.get(task_id, {"name": task_id or "通用", "desc": "认知评估", "eval_hint": "判断回答是否符合问题要求"})
        
        # 获取实时上下文（用于定向力评估）
        context = get_realtime_context()
        time_info = context['time']
        location = context['location']
        
        # 🔥 简化的系统提示（只要核心字段，避免截断）
        system_prompt = """评估患者回答，只输出JSON：{"is_correct":true/false,"quality":"excellent/good/fair/poor"}
- is_correct: 回答是否正确
- quality: excellent=完美, good=基本对, fair=部分对, poor=错误"""

        # 构建用户提示
        user_prompt = f"任务: {task_info['name']}（{task_info['desc']}）\n问题: {question}\n回答: {answer}"
        
        # 添加评估提示
        eval_hint = task_info.get('eval_hint', '')
        if eval_hint:
            user_prompt += f"\n\n【评判标准】{eval_hint}"
        
        if expected_answer:
            user_prompt += f"\n期望答案: {expected_answer}"
        
        # 定向力任务需要当前时间/地点信息
        if task_id.startswith("orientation"):
            user_prompt += f"\n\n【参考信息】"
            user_prompt += f"\n当前时间: {time_info['year']}年{time_info['month']}月{time_info['day']}日 星期{time_info['weekday']} {time_info['season']}"
            user_prompt += f"\n当前地点: {location.get('province', '')} {location.get('city', '')} {location.get('district', '')}"
        
        if patient_profile:
            user_prompt += f"\n患者: {patient_profile.get('age', '?')}岁, 受教育{patient_profile.get('education_years', '?')}年"
        
        try:
            response = self._llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            
            content = self._extract_response(response).strip()
            
            # 清理 Markdown
            if "```" in content:
                match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
                if match:
                    content = match.group(1).strip()
            
            # 解析 JSON（增加截断容错）
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # JSON 被截断，尝试手动提取字段
                    print(f"[AnswerEvalTool] ⚠️ JSON 被截断，尝试手动提取...")
                    data = self._parse_truncated_json(content)
            else:
                # 没有找到完整的 JSON，尝试手动提取
                print(f"[AnswerEvalTool] ⚠️ 未找到完整 JSON，尝试手动提取...")
                data = self._parse_truncated_json(content)
            
            # 🔥 简化的结果（只要核心字段）
            is_correct = bool(data.get("is_correct", False))
            quality = str(data.get("quality", data.get("quality_level", "fair")))
            
            # 根据 quality 推断其他字段
            quality_to_cognitive = {
                "excellent": "正常", "good": "正常", 
                "fair": "轻度异常", "poor": "中度异常"
            }
            
            result = {
                "is_correct": is_correct,
                "quality_level": quality,
                "cognitive_performance": quality_to_cognitive.get(quality, "无法判断"),
                "is_complete": is_correct,  # 简化：正确就视为完整
                "evaluation_detail": "",
                "confidence": 0.9 if is_correct else 0.8
            }
            
            elapsed = (time.time() - start_time) * 1000
            print(f"[AnswerEvalTool] ✅ 评估完成: 正确={is_correct}, 质量={quality} ({elapsed:.0f}ms)")
            
            return json.dumps(result, ensure_ascii=False)
            
        except Exception as e:
            print(f"[AnswerEvalTool] ❌ 评估失败: {e}")
            return json.dumps({
                "is_correct": False,
                "quality_level": "poor",
                "cognitive_performance": "无法判断",
                "is_complete": False,
                "evaluation_detail": "",
                "confidence": 0.0
            }, ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
