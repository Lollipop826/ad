"""
维度检测工具 - 供Agent调用

判断用户回答是否回答了问题，并识别涉及哪些MMSE维度
"""

from __future__ import annotations

import os
import json
from typing import List, Optional, Type

from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

from src.domain.dimensions import MMSE_DIMENSIONS
from src.llm.http_client_pool import get_siliconflow_chat_openai, get_volcengine_chat_openai


class DimensionDetectionToolArgs(BaseModel):
    """维度检测工具参数"""
    
    question: str = Field(..., description="医生提出的问题")
    answer: str = Field(default="", description="用户的回答")
    dimensions: Optional[List[dict]] = Field(
        default=None,
        description="可选，自定义维度列表；若为空则使用内置MMSE维度",
    )
    language: str = Field("zh", description="语言，默认中文")


class DimensionDetectionResult(BaseModel):
    """维度检测结果"""
    
    answered: bool = Field(..., description="是否回答了所问问题（语义上对答了）")
    covered_dimensions: List[str] = Field(
        default_factory=list, description="回答涉及到的维度id列表"
    )
    confidence: float = Field(0.7, description="0-1 置信度")
    rationale: Optional[str] = Field(None, description="简要判断理由")


class DimensionDetectionTool(BaseTool):
    """
    维度检测工具
    
    判断患者回答是否回答了医生的问题，并识别回答涉及哪些MMSE维度。
    """
    
    name: str = "dimension_detection_tool"
    description: str = (
        "分析患者回答涵盖哪些MMSE认知维度。"
        "输入参数：question（医生的问题），answer（患者的回答）。"
        "返回：是否回答了问题、涉及的维度列表、置信度。"
    )
    
    args_schema: Type[BaseModel] = DimensionDetectionToolArgs
    
    _llm: ChatOpenAI = PrivateAttr()
    _system_prompt: str = PrivateAttr()
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        use_local: bool = False,  # 新增参数
        llm_instance = None,      # 允许直接传入LLM实例
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if use_local and llm_instance:
            # 直接使用传入的本地LLM实例
            self._llm = llm_instance
        elif use_local:
            # 使用池化的小模型实例（0.5B，简单分类任务）
            from src.llm.model_pool import get_pooled_llm
            self._llm = get_pooled_llm(pool_key='small_classify')  # 🔥 使用 0.5B 小模型，加速 10-15倍
        else:
            # 使用API
            if os.getenv("ARK_API_KEY"):
                self._llm = get_volcengine_chat_openai(
                    model=model or os.getenv("DIMENSION_MODEL", "doubao-seed-2-0-mini-260215"),
                    temperature=temperature,
                    timeout=20,
                    max_retries=1,
                )
            else:
                self._llm = get_siliconflow_chat_openai(
                    model=model or os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    timeout=20,
                    max_retries=1,
                )
        
        self._system_prompt = (
            "你是阿尔茨海默病初筛对话助手。根据给定的维度表(MMSE)，"
            "判断患者回答是否语义上回答了医生的问题，并识别回答涉及到的维度。\n"
            "输出JSON：{answered, covered_dimensions, confidence, rationale}。\n"
            "注意：\n"
            "- answered 需基于语义匹配，不是逐字。\n"
            "- covered_dimensions 返回维度id，如 orientation/registration/attention_calculation/recall/language/copy。\n"
            "- 支持新的灵活任务：orientation_assessment/language_assessment等。\n"
            "- 如果未涉及任何维度则给空数组。"
        )
    
    def _run(
        self,
        question: str,
        answer: str = "",
        dimensions: Optional[List[dict]] = None,
        language: str = "zh",
    ) -> str:  # type: ignore[override]
        """
        执行维度检测
        
        Returns:
            JSON格式的检测结果
        """
        # 容错处理：如果question包含JSON，尝试解析
        if question.strip().startswith('{'):
            try:
                parsed = json.loads(question)
                question = parsed.get('question', question)
                answer = parsed.get('answer', answer)
            except:
                pass
        
        # 如果answer仍然为空，返回错误提示
        if not answer or answer == "":
            return json.dumps({
                "answered": False,
                "covered_dimensions": [],
                "confidence": 0.0,
                "rationale": "参数错误：未提供患者回答"
            }, ensure_ascii=False)
        
        dims = dimensions or MMSE_DIMENSIONS
        dims_json = json.dumps(dims, ensure_ascii=False)
        
        user_prompt = (
            f"语言: {language}\n"
            f"维度表(JSON): {dims_json}\n\n"
            f"医生问题: {question}\n"
            f"用户回答: {answer}\n\n"
            "请仅以JSON返回：{\"answered\": true|false, \"covered_dimensions\": [id...], "
            "\"confidence\": 0.0-1.0, \"rationale\": \"一句话理由\"}"
        )
        
        msg = self._llm.invoke([
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        
        # 兼容处理：本地模型返回str，ChatOpenAI返回AIMessage
        if hasattr(msg, "content"):
            content = (msg.content or "").strip()
        else:
            content = str(msg).strip()
        
        # 解析JSON
        try:
            data = json.loads(content)
        except Exception:
            import re
            m = re.search(r"\{[\s\S]*\}", content)
            data = json.loads(m.group(0)) if m else {}
        
        res = DimensionDetectionResult(
            answered=bool(data.get("answered", False)),
            covered_dimensions=list(data.get("covered_dimensions", []) or []),
            confidence=float(data.get("confidence", 0.7)),
            rationale=(data.get("rationale") or ""),
        )
        return json.dumps(res.model_dump(), ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")

