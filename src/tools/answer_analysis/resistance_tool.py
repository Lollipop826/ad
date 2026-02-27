from __future__ import annotations

from typing import Optional, Type
import os
import json

from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from src.llm.http_client_pool import get_siliconflow_chat_openai


class ResistanceToolArgs(BaseModel):
    """Arguments for resistance emotion detection."""

    question: str = Field(..., description="医生提出的问题")
    answer: str = Field(..., description="用户的回答文本（ASR转写后）")
    language: str = Field("zh", description="语言，默认中文")


class ResistanceDetectionResult(BaseModel):
    """Normalized result from LLM detection."""

    is_resistant: bool = Field(..., description="是否存在抵抗/拒绝/不配合情绪")
    category: Optional[str] = Field(
        default=None,
        description="抵抗类别：refusal|avoidance|hostility|fatigue|none",
    )
    confidence: float = Field(0.7, description="置信度 0-1")
    rationale: Optional[str] = Field(default=None, description="判断理由（简要）")


class ResistanceDetectionTool(BaseTool):
    name: str = "resistance_detection_tool"
    description: str = (
        "判断用户回答中是否存在抵抗/拒绝/不配合等情绪（中文）。"
        "输入医生问题与用户回答，输出是否抵抗、类别、置信度与简要理由。"
    )

    args_schema: Type[BaseModel] = ResistanceToolArgs

    # Private runtime attributes
    _llm: ChatOpenAI = PrivateAttr()
    _system_prompt: str = PrivateAttr()

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        use_local: bool = False,  # 新增参数
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        if use_local:
            # 使用本地模型池（小模型，快速分类）
            print("[ResistanceTool] 🚀 使用模型池实例（小模型 0.5B）")
            from src.llm.model_pool import get_pooled_llm
            self._llm = get_pooled_llm(pool_key='small_classify')  # 🔥 使用小模型分类
        else:
            # 使用API
            self._llm = get_siliconflow_chat_openai(
                model=model or os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
                base_url=base_url,
                api_key=api_key or os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY"),
                temperature=temperature,
                timeout=20,
                max_retries=1,
            )

        self._system_prompt = (
            "你是临床面谈助手。判断患者回答里是否存在‘抵抗情绪’："
            "包括但不限于：拒绝回答、明显回避、不配合、敌意、强烈否定、反问敷衍、情绪升级、反驳医嘱、反复强调不想继续等。\n"
            "分类：refusal(拒绝)/avoidance(回避)/hostility(敌意)/fatigue(疲惫)/none。\n"
            "请只基于提供的问题与回答进行判断，输出JSON：{is_resistant, category, confidence, rationale}。"
        )

    def _run(self, question: str, answer: str, language: str = "zh") -> str:  # type: ignore[override]
        user_prompt = (
            f"语言: {language}\n"
            f"医生问题: {question}\n"
            f"用户回答: {answer}\n\n"
            "请以JSON返回：{\"is_resistant\": true|false, \"category\": \"refusal|avoidance|hostility|fatigue|none\", "
            "\"confidence\": 0.0-1.0, \"rationale\": \"一句话理由\"}"
        )
        msg = self._llm.invoke([
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        content = (getattr(msg, "content", "") or "").strip()

        # 尝试解析为标准结构
        try:
            data = json.loads(content)
        except Exception:
            # 宽松提取JSON片段
            import re

            m = re.search(r"\{[\s\S]*\}", content)
            if m:
                try:
                    data = json.loads(m.group(0))
                except Exception:
                    data = {}
            else:
                data = {}

        # 归一化输出
        res = ResistanceDetectionResult(
            is_resistant=bool(data.get("is_resistant", False)),
            category=(data.get("category") or ("none" if not data.get("is_resistant") else "refusal")),
            confidence=float(data.get("confidence", 0.7)),
            rationale=(data.get("rationale") or ""),
        )
        return json.dumps(res.model_dump(), ensure_ascii=False)

    async def _arun(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


