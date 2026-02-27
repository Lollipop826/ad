"""
Agent工具集合 - 所有可供LLM Agent调用的工具

这个模块包含了阿尔茨海默病初筛对话系统的所有工具：
- 查询生成：根据维度和上下文生成检索查询
- 知识检索：从医学知识库检索相关信息
- 维度检测：判断用户回答涉及哪些维度
- 抵抗检测：识别用户的抵抗情绪
- 问题生成：基于知识生成评估问题
- 对话存储：保存和获取对话历史
- 回答评估：评估回答正确性并进行MMSE评分
- 评分记录：记录和管理评估得分
"""

from .query_tool import QueryTool
from .retrieval_tool import KnowledgeRetrievalTool
from .dimension_detection_tool import DimensionDetectionTool
from .score_recording_tool import ScoreRecordingTool
from .mmse_scoring_tool import MMSEScoringTool
from .resistance_detection_tool import ResistanceDetectionTool
from .comfort_response_tool import ComfortResponseTool
from .question_generation_tool import QuestionGenerationTool
from .storage_tool import ConversationStorageTool
from .answer_evaluation_tool import AnswerEvaluationTool
from .image_display_tool import ImageDisplayTool
from .standard_question_tool import StandardQuestionTool
from .dimension_switch_tool import DimensionSwitchTool

__all__ = [
    "QueryTool",
    "KnowledgeRetrievalTool",
    "DimensionDetectionTool",
    "ResistanceDetectionTool",
    "ComfortResponseTool",
    "QuestionGenerationTool",
    "ConversationStorageTool",
    "AnswerEvaluationTool",
    "ScoreRecordingTool",
    "MMSEScoringTool",
    "ImageDisplayTool",
    "StandardQuestionTool",
    "DimensionSwitchTool",
    "ALL_TOOLS",
]

# 工具列表（用于Agent初始化）
ALL_TOOLS = [
    QueryTool,
    KnowledgeRetrievalTool,
    DimensionDetectionTool,
    ResistanceDetectionTool,
    ComfortResponseTool,
    QuestionGenerationTool,
    ConversationStorageTool,
    AnswerEvaluationTool,
    ScoreRecordingTool,
    MMSEScoringTool,
    ImageDisplayTool,
    StandardQuestionTool,
    DimensionSwitchTool,
]
