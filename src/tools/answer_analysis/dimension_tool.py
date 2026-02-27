from __future__ import annotations

import os
import json
from typing import List, Optional, Type

from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI

from src.domain.dimensions import MMSE_DIMENSIONS
from src.llm.http_client_pool import get_siliconflow_chat_openai


class DimensionToolArgs(BaseModel):
    """Arguments for MMSE dimension recognition and answer checking."""

    question: str = Field(..., description="医生提出的问题")
    answer: str = Field(..., description="用户的回答")
    dimensions: Optional[List[dict]] = Field(
        default=None,
        description="可选，自定义维度列表；若为空则使用内置MMSE维度",
    )
    language: str = Field("zh", description="语言，默认中文")


class DimensionDetectionResult(BaseModel):
    """Normalized detection result."""

    answered: bool = Field(..., description="是否回答了所问问题（语义上对答了）")
    covered_dimensions: List[str] = Field(
        default_factory=list, description="回答涉及到的维度id列表"
    )
    confidence: float = Field(0.7, description="0-1 置信度")
    rationale: Optional[str] = Field(None, description="简要判断理由")


class DimensionDetectionTool(BaseTool):
    name: str = "dimension_detection_tool"
    description: str = (
        "识别用户回答涉及哪些MMSE维度，并判断是否回答了医生提出的问题。"
        "输入医生问题、用户回答与维度表（可选），输出 answered、covered_dimensions、confidence、rationale。"
    )

    args_schema: Type[BaseModel] = DimensionToolArgs

    # private attrs
    _llm: ChatOpenAI = PrivateAttr()
    _system_prompt: str = PrivateAttr()

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._llm = get_siliconflow_chat_openai(
            model=model or os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
            base_url=base_url,
            api_key=api_key or os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY"),
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
            "- 如果未涉及任何维度则给空数组。"
        )

    def _run(
        self,
        question: str,
        answer: str,
        dimensions: Optional[List[dict]] = None,
        language: str = "zh",
    ) -> str:  # type: ignore[override]
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
        content = (getattr(msg, "content", "") or "").strip()

        # parse
        data: dict
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

    async def _arun(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


