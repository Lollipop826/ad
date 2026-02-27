"""
MMSE标准评分工具 - 符合国际标准的30分制评分

标准计分方式：
1. 定向力（Orientation）：10分（时间5分 + 地点5分）
2. 即时记忆（Registration）：3分
3. 注意力与计算（Attention & Calculation）：5分
4. 延迟回忆（Recall）：3分
5. 语言（Language）：8分
6. 构图（Copy）：1分

总分：30分
判断标准：
- 27-30分：认知功能正常
- 21-26分：轻度认知障碍（MCI）
- 10-20分：中度认知障碍（疑似阿尔茨海默病）
- 0-9分：重度认知障碍（阿尔茨海默病）
"""

from __future__ import annotations

import os
import json
from typing import Optional, Type, Dict, Any, List
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool


# MMSE标准分值
MMSE_STANDARD_SCORES = {
    "orientation": 10,          # 定向力：时间5分 + 地点5分
    "registration": 3,          # 即时记忆：3个词
    "attention_calculation": 5, # 注意力与计算：100-7连续5次
    "recall": 3,                # 延迟回忆：3个词
    "language": 8,              # 语言：命名2+复述1+三步指令3+阅读1+书写1
    "copy": 1,                  # 构图：临摹五边形
}

MMSE_TOTAL_SCORE = 30


class MMSEScoringToolArgs(BaseModel):
    """MMSE评分工具参数"""
    
    session_id: str = Field(..., description="会话ID")
    dimension_id: str = Field(..., description="维度ID（orientation/registration/attention_calculation/recall/language/copy，支持灵活任务如orientation_assessment/language_assessment）")
    score: int = Field(..., description="该维度得分（0到该维度满分）")
    max_score: Optional[int] = Field(None, description="该维度满分（可选，默认使用标准分值）")
    question: str = Field(default="", description="评估问题")
    answer: str = Field(default="", description="患者回答")
    evaluation_detail: str = Field(default="", description="评分依据")
    action: str = Field(default="save", description="操作类型：save（保存）/get（获取）/summary（汇总）")


class MMSEScoringResult(BaseModel):
    """MMSE评分结果"""
    
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作结果消息")
    dimension_score: Optional[int] = Field(None, description="当前维度得分")
    dimension_max_score: Optional[int] = Field(None, description="当前维度满分")
    total_score: Optional[int] = Field(None, description="累计总分")
    total_max_score: int = Field(MMSE_TOTAL_SCORE, description="MMSE总分")
    completed_max_score: Optional[int] = Field(None, description="已评估项目的满分合计（用于折算）")
    scaled_total_score: Optional[float] = Field(None, description="按已评估项目折算到30分制的总分")
    coverage: Optional[float] = Field(None, description="覆盖率=已评估满分/30")
    missing_dimensions: Optional[List[str]] = Field(None, description="未评估的维度列表")
    cognitive_status: Optional[str] = Field(None, description="认知功能状态")
    completed_dimensions: Optional[List[str]] = Field(None, description="已完成的维度")
    scoring_details: Optional[Dict[str, Any]] = Field(None, description="详细评分记录")


class MMSEScoringTool(BaseTool):
    """
    MMSE标准评分工具
    
    按照MMSE国际标准（30分制）进行评分，并提供认知功能判断。
    """
    
    name: str = "mmse_scoring_tool"
    description: str = (
        "MMSE标准评分工具（30分制）。"
        "输入参数：session_id（会话ID），dimension_id（维度ID），score（得分），action（操作类型）。"
        "支持操作：save（保存评分）、get（获取维度评分）、summary（获取总分和认知状态）。"
        "维度ID：orientation/registration/attention_calculation/recall/language/copy（支持灵活任务如orientation_assessment/language_assessment）。"
    )
    
    args_schema: Type[BaseModel] = MMSEScoringToolArgs
    
    _scoring_dir: Path = PrivateAttr()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._scoring_dir = Path("data/mmse_scores")
        self._scoring_dir.mkdir(parents=True, exist_ok=True)
    
    def _run(
        self,
        session_id: str,
        dimension_id: str,
        score: int = 0,
        max_score: Optional[int] = None,
        question: str = "",
        answer: str = "",
        evaluation_detail: str = "",
        action: str = "save",
    ) -> str:
        """
        执行MMSE评分操作
        
        Returns:
            JSON格式的评分结果
        """
        # 验证 session_id
        if not session_id or not session_id.strip():
            return json.dumps({
                "success": False,
                "message": "session_id 不能为空"
            }, ensure_ascii=False)
        
        scoring_file = self._scoring_dir / f"{session_id}_mmse.json"
        
        # 读取现有评分
        scoring_data = self._load_scoring(scoring_file)
        
        if action == "save":
            return self._save_score(
                scoring_data, scoring_file, session_id, dimension_id,
                score, max_score, question, answer, evaluation_detail
            )
        elif action == "get":
            return self._get_score(scoring_data, dimension_id)
        elif action == "summary":
            return self._get_summary(scoring_data)
        else:
            return json.dumps({
                "success": False,
                "message": f"不支持的操作类型: {action}"
            }, ensure_ascii=False)
    
    def _load_scoring(self, scoring_file: Path) -> Dict[str, Any]:
        """加载MMSE评分记录"""
        if scoring_file.exists():
            try:
                with open(scoring_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[MMSE] 加载评分文件失败: {e}")
                return self._init_scoring_data()
        else:
            return self._init_scoring_data()
    
    def _init_scoring_data(self) -> Dict[str, Any]:
        """初始化MMSE评分数据结构"""
        return {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dimensions": {},
            "total_score": 0,
            "total_max_score": MMSE_TOTAL_SCORE,
            "cognitive_status": "未评估"
        }
    
    def _save_score(
        self,
        scoring_data: Dict[str, Any],
        scoring_file: Path,
        session_id: str,
        dimension_id: str,
        score: int,
        max_score: Optional[int],
        question: str,
        answer: str,
        evaluation_detail: str
    ) -> str:
        """保存MMSE评分"""
        try:
            # 获取该维度的标准分值
            standard_max_score = MMSE_STANDARD_SCORES.get(dimension_id, 0)
            if max_score is None:
                max_score = standard_max_score
            
            # 验证维度ID
            if standard_max_score == 0:
                return json.dumps({
                    "success": False,
                    "message": f"无效的维度ID: {dimension_id}"
                }, ensure_ascii=False)
            
            # 验证分数范围
            if score < 0 or score > max_score:
                return json.dumps({
                    "success": False,
                    "message": f"分数超出范围（0-{max_score}）"
                }, ensure_ascii=False)
            
            existing = scoring_data["dimensions"].get(dimension_id)
            timestamp = datetime.now().isoformat()
            record_item = {
                "score": score,
                "max_score": max_score,
                "question": question,
                "answer": answer,
                "evaluation_detail": evaluation_detail,
                "timestamp": timestamp,
            }

            # 兼容模式：
            # - 当 max_score < standard_max_score 时，认为是子任务计分（同一维度可多次累计）
            # - 当 max_score >= standard_max_score 时，认为是整维度一次性计分（覆盖保存）
            if existing and max_score < standard_max_score:
                prev_score = int(existing.get("score", 0) or 0)
                prev_max = int(existing.get("max_score", 0) or 0)

                new_max = min(prev_max + max_score, standard_max_score)
                new_score = min(prev_score + score, new_max)

                records = list(existing.get("records") or [])
                records.append(record_item)

                scoring_data["dimensions"][dimension_id] = {
                    **existing,
                    "score": new_score,
                    "max_score": new_max,
                    "standard_max_score": standard_max_score,
                    "question": question,
                    "answer": answer,
                    "evaluation_detail": evaluation_detail,
                    "timestamp": timestamp,
                    "records": records,
                }
            else:
                scoring_data["dimensions"][dimension_id] = {
                    "score": score,
                    "max_score": max_score,
                    "standard_max_score": standard_max_score,
                    "question": question,
                    "answer": answer,
                    "evaluation_detail": evaluation_detail,
                    "timestamp": timestamp,
                    "records": [record_item],
                }
            
            # 计算总分
            total_score = sum(d["score"] for d in scoring_data["dimensions"].values())
            completed_max_score = sum(d.get("max_score", d["standard_max_score"]) for d in scoring_data["dimensions"].values())
            completed_standard_max_score = sum(d["standard_max_score"] for d in scoring_data["dimensions"].values())
            
            scoring_data["total_score"] = total_score
            scoring_data["completed_max_score"] = completed_max_score
            scoring_data["completed_standard_max_score"] = completed_standard_max_score
            scoring_data["updated_at"] = datetime.now().isoformat()
            
            # 判断认知功能状态（根据已完成的项目比例估算）
            if completed_max_score == MMSE_TOTAL_SCORE:
                scoring_data["cognitive_status"] = self._judge_cognitive_status(total_score)
            else:
                estimated_total = int(total_score / completed_max_score * MMSE_TOTAL_SCORE) if completed_max_score > 0 else 0
                scoring_data["cognitive_status"] = f"预估: {self._judge_cognitive_status(estimated_total)} (覆盖率{completed_max_score}/{MMSE_TOTAL_SCORE})"
            
            # 保存到文件
            with open(scoring_file, 'w', encoding='utf-8') as f:
                json.dump(scoring_data, f, ensure_ascii=False, indent=2)
            
            saved_dim = scoring_data["dimensions"].get(dimension_id, {})
            result = MMSEScoringResult(
                success=True,
                message=f"已记录 {dimension_id} 维度评分: {saved_dim.get('score', score)}/{saved_dim.get('max_score', max_score)}分",
                dimension_score=saved_dim.get('score', score),
                dimension_max_score=saved_dim.get('max_score', max_score),
                total_score=total_score,
                total_max_score=MMSE_TOTAL_SCORE,
                completed_max_score=completed_max_score,
                scaled_total_score=round(total_score / completed_max_score * MMSE_TOTAL_SCORE, 1) if completed_max_score else 0.0,
                coverage=round(completed_max_score / MMSE_TOTAL_SCORE, 3) if completed_max_score else 0.0,
                cognitive_status=scoring_data["cognitive_status"],
                completed_dimensions=list(scoring_data["dimensions"].keys())
            )
            
            print(f"[MMSE] ✅ 维度 {dimension_id}: {score}/{max_score}分，累计 {total_score}分")
            
            return json.dumps(result.model_dump(), ensure_ascii=False)
            
        except Exception as e:
            return json.dumps({
                "success": False,
                "message": f"保存评分失败: {str(e)}"
            }, ensure_ascii=False)
    
    def _judge_cognitive_status(self, total_score: int) -> str:
        """根据总分判断认知功能状态"""
        if total_score >= 27:
            return "认知功能正常"
        elif total_score >= 21:
            return "轻度认知障碍（MCI）"
        elif total_score >= 10:
            return "中度认知障碍（疑似阿尔茨海默病）"
        else:
            return "重度认知障碍（阿尔茨海默病）"
    
    def _get_score(self, scoring_data: Dict[str, Any], dimension_id: str) -> str:
        """获取指定维度的评分"""
        if dimension_id in scoring_data["dimensions"]:
            dim_data = scoring_data["dimensions"][dimension_id]
            result = MMSEScoringResult(
                success=True,
                message=f"{dimension_id} 维度评分: {dim_data['score']}/{dim_data['max_score']}分",
                dimension_score=dim_data["score"],
                dimension_max_score=dim_data["max_score"],
                scoring_details=dim_data
            )
        else:
            result = MMSEScoringResult(
                success=False,
                message=f"未找到 {dimension_id} 维度的评分记录",
                dimension_score=0,
                dimension_max_score=MMSE_STANDARD_SCORES.get(dimension_id, 0)
            )
        
        return json.dumps(result.model_dump(), ensure_ascii=False)
    
    def _get_summary(self, scoring_data: Dict[str, Any]) -> str:
        """获取MMSE总分和认知状态汇总"""
        total_score = scoring_data.get("total_score", 0)
        cognitive_status = scoring_data.get("cognitive_status", "未评估")
        dimensions = scoring_data.get("dimensions", {})

        completed_max_score = scoring_data.get("completed_max_score")
        if completed_max_score is None:
            completed_max_score = sum(
                d.get("max_score", d.get("standard_max_score", 0)) for d in dimensions.values()
            )

        missing_dimensions = [
            dim for dim in MMSE_STANDARD_SCORES.keys() if dim not in dimensions
        ]

        scaled_total_score = round(total_score / completed_max_score * MMSE_TOTAL_SCORE, 1) if completed_max_score else 0.0
        coverage = round(completed_max_score / MMSE_TOTAL_SCORE, 3) if completed_max_score else 0.0
        
        # 统计各维度得分
        dimension_summary = {}
        for dim_id, dim_data in dimensions.items():
            dim_standard_max = dim_data.get("standard_max_score", MMSE_STANDARD_SCORES.get(dim_id, 0))
            dim_effective_max = dim_data.get("max_score", dim_standard_max)
            dimension_summary[dim_id] = {
                "score": dim_data["score"],
                "max_score": dim_effective_max,
                "standard_max_score": dim_standard_max,
                "percentage": round(dim_data["score"] / dim_effective_max * 100, 1) if dim_effective_max else 0.0
            }
        
        result = MMSEScoringResult(
            success=True,
            message=f"MMSE总分: {total_score}/{MMSE_TOTAL_SCORE}分 - {cognitive_status}",
            total_score=total_score,
            total_max_score=MMSE_TOTAL_SCORE,
            completed_max_score=completed_max_score,
            scaled_total_score=scaled_total_score,
            coverage=coverage,
            missing_dimensions=missing_dimensions,
            cognitive_status=cognitive_status,
            completed_dimensions=list(dimensions.keys()),
            scoring_details={
                "total_score": total_score,
                "total_max_score": MMSE_TOTAL_SCORE,
                "completed_max_score": completed_max_score,
                "scaled_total_score": scaled_total_score,
                "coverage": coverage,
                "missing_dimensions": missing_dimensions,
                "cognitive_status": cognitive_status,
                "dimension_scores": dimension_summary,
                "completed_dimensions": len(dimensions),
                "total_dimensions": 6,
                "completion_rate": round(len(dimensions) / 6 * 100, 1)
            }
        )
        
        return json.dumps(result.model_dump(), ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")
