from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from src.common.types import InfoDimension, ConversationTurn, Profile, SearchQueryResult
from .generator import QuerySentenceGenerator, QuerySentenceGeneratorConfig


class ToolInfoDimension(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[int] = None
    status: Optional[str] = None
    value: Optional[str] = None


class ToolConversationTurn(BaseModel):
    role: str
    content: str
    emotion: Optional[str] = None


class ToolProfile(BaseModel):
    user_id: Optional[str] = None
    name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    education_years: Optional[int] = Field(default=None, alias="educationYears")
    notes: Optional[str] = None


class QuerySentenceToolArgs(BaseModel):
    dimension: ToolInfoDimension
    history: List[ToolConversationTurn] = []
    last_emotion: Optional[str] = None
    profile: Optional[ToolProfile] = None


class QuerySentenceTool(BaseTool):
    name: str = "query_sentence_generator"
    description: str = (
        "根据给定维度+历史对话+上次情绪(+画像) 生成一条仅供系统内部使用的中文查询语句。"
    )

    args_schema: type[QuerySentenceToolArgs] = QuerySentenceToolArgs

    generator: QuerySentenceGenerator

    def __init__(self, config: Optional[QuerySentenceGeneratorConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.generator = QuerySentenceGenerator(config=config)

    def _run(self, dimension: ToolInfoDimension, history: List[ToolConversationTurn] = [], last_emotion: Optional[str] = None, profile: Optional[ToolProfile] = None) -> SearchQueryResult:  # type: ignore[override]
        dim: InfoDimension = dimension.dict(by_alias=True)
        hist: List[ConversationTurn] = [h.dict() for h in history]
        prof: Optional[Profile] = profile.dict(by_alias=True) if profile else None
        return self.generator.generate_query(dimension=dim, history=hist, last_emotion=last_emotion, profile=prof)

    async def _arun(self, *args, **kwargs):  # pragma: no cover - async not used in demo
        raise NotImplementedError("QuerySentenceTool does not implement async.")
