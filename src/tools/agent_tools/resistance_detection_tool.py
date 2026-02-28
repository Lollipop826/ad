"""
抵抗情绪检测工具 - 纯 LLM 方案

策略：
1. 完全使用 LLM 进行抵抗情绪检测（因为 BERT 小模型效果不佳）
2. 保持接口兼容性
3. 优化 Prompt 以提高检测准确率

标签体系（5类，但只有 distress/hostility 触发抵抗处理）：
0 - 正常配合（normal）
1 - 小抱怨/轻微回避（avoidance）- 不触发抵抗处理
2 - 情绪困扰（distress）：焦虑、悲伤、不信任
3 - 主动对抗（hostility）：愤怒、逆反、攻击
4 - 请求重复（repeat_request）：没听清、再说一遍、什么意思
"""

from __future__ import annotations

from typing import Optional, Type, ClassVar, List, Dict
import os
import json
import time
import re

from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from src.llm.http_client_pool import get_siliconflow_chat_openai


class ResistanceDetectionToolArgs(BaseModel):
    """抵抗情绪检测工具参数"""
    
    question: str = Field(..., description="医生提出的问题")
    answer: str = Field(default="", description="用户的回答")
    language: str = Field("zh", description="语言，默认中文")


class ResistanceDetectionResult(BaseModel):
    """抵抗检测结果"""
    
    is_resistant: bool = Field(..., description="是否存在抵抗/拒绝/不配合情绪")
    category: Optional[str] = Field(
        default=None,
        description="抵抗类别：normal|avoidance|distress|hostility",
    )
    category_zh: Optional[str] = Field(default=None, description="抵抗类别中文名")
    confidence: float = Field(0.7, description="置信度 0-1")
    rationale: Optional[str] = Field(default=None, description="判断理由（简要）")
    inference_time_ms: float = Field(0, description="推理耗时毫秒")
    method: str = Field("llm", description="检测方法：llm")


# 标签映射（5类）
LABEL_NAMES = {
    0: "normal",           # 正常配合
    1: "avoidance",        # 消极回避
    2: "distress",         # 情绪困扰
    3: "hostility",        # 主动对抗
    4: "repeat_request",   # 请求重复
}

LABEL_ZH = {
    0: "正常配合",
    1: "小抱怨",  # 🔥 改名，不触发抵抗
    2: "情绪困扰",
    3: "主动对抗",
    4: "请求重复",
}

LABEL_DESC = {
    0: "患者积极配合评估，正常回答问题",
    1: "患者有小抱怨或轻微借口，但不算抵抗",  # 🔥 不触发抵抗
    2: "患者焦虑害怕、悲伤失落或不信任医护人员",
    3: "患者发火生气、不耐烦或故意唱反调",
    4: "患者没听清或请求医生重复问题",
}


class ResistanceDetectionTool(BaseTool):
    """
    抵抗情绪检测工具（纯 LLM 方案）
    
    判断患者回答中是否存在抵抗、拒绝、回避、敌意等负面情绪，
    并细分为4种类别。
    """
    
    name: str = "resistance_detection_tool"
    description: str = (
        "检测患者回答中的情绪状态（抵抗/拒绝/不配合）。"
        "输入参数：question（医生的问题），answer（患者的回答）。"
        "返回：是否有抵抗情绪、情绪类别、置信度。"
    )
    
    args_schema: Type[BaseModel] = ResistanceDetectionToolArgs
    
    _llm: object = PrivateAttr(default=None)
    _llm_system_prompt: str = PrivateAttr()
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,  # 降低温度以提高稳定性
        use_local: bool = False,
        llm_instance = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # 初始化LLM
        if llm_instance:
            print("[ResistanceTool] 🔄 使用传入的LLM实例")
            self._llm = llm_instance
        elif use_local:
            # 使用本地模型池
            try:
                from src.llm.model_pool import get_pooled_llm, get_model_pool
                pool = get_model_pool()
                stats = pool.get_stats()
                available_keys = stats.get('available_keys', [])
                
                # 优先使用 small_classify (7B)，如果不可用则用 default (7B/14B)
                if 'small_classify' in available_keys:
                    self._llm = get_pooled_llm(pool_key='small_classify')
                    print("[ResistanceTool] 🔄 使用 small_classify 本地LLM")
                elif '7b_default' in available_keys:
                    self._llm = get_pooled_llm(pool_key='7b_default')
                    print("[ResistanceTool] 🔄 使用 7B 本地LLM")
                else:
                    self._llm = get_pooled_llm(pool_key='default')
                    print("[ResistanceTool] 🔄 使用默认本地LLM")
            except Exception as e:
                print(f"[ResistanceTool] ⚠️ 本地LLM初始化失败: {e}")
        else:
            # 使用API
            try:
                self._llm = get_siliconflow_chat_openai(
                    model=model or os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    timeout=10,
                    max_retries=1,
                )
                print("[ResistanceTool] 🔄 使用API LLM")
            except Exception as e:
                print(f"[ResistanceTool] ⚠️ API LLM初始化失败: {e}")
        
        # 优化后的 LLM 系统提示
        self._llm_system_prompt = """你是阿尔茨海默病认知评估场景的意图分类器。你的唯一任务是判断患者的回答属于以下5个类别中的哪一个。

【类别定义】
0-正常配合 (normal)：
- 愿意尝试回答问题（即使答错了、忘了、不知道）
- 态度温和、礼貌
- 正常的社交性回应
- 示例："好的，我试试"、"让我想想"、"哎呀我忘了"、"我不记得了"

1-消极回避 (avoidance)：
- 否认自己有问题 ("我脑子好着呢")
- 找借口回避 ("今天累了不想测")
- 转移话题 ("吃饭了吗")
- 感到羞耻 ("这么简单都不会，丢人")
- 淡化症状 ("老了都这样")

2-情绪困扰 (distress)：
- 焦虑、害怕 ("我怕查出病来")
- 悲伤、绝望 ("活着没意思")
- 猜疑、不信任 ("你是谁？查户口的？")

3-主动对抗 (hostility)：
- 生气、发火 ("烦死了")
- 不耐烦 ("有完没完")
- 故意唱反调 ("我就不说")
- 攻击性语言 ("滚开")
- 明确拒绝 ("我不做这个")

4-请求重复 (repeat_request)：🔥 最高优先级判断
- 没听清楚 ("什么？"、"你说什么？"、"啊？")
- 请求再说一遍 ("再说一遍"、"能重复一下吗")
- 没听懂 ("没听明白"、"什么意思")
- 确认性提问 ("你刚才说什么？"、"说的啥？")
- 示例：
  - "没听清你说的" → 4
  - "再说一遍好么" → 4  
  - "什么？" → 4
  - "啊？你说啥" → 4
  - "能再问一次吗" → 4

【重要区分规则】
1. 🔥 最优先：如果患者明确表示没听清/请求重复 → 4 (请求重复)
2. "不知道/忘了" + 配合态度 = 0 (正常)
3. "不知道/忘了" + 不耐烦/气愤 = 3 (对抗)
4. "不想做/不想说" = 1 (回避) 或 3 (对抗)，取决于语气强烈程度
5. "这题毫无意义/太简单了" = 1 (回避)
6. 🔥 关键规则A：只要回答中包含了问题询问的【核心信息】（如日期、地点、物品名），即使有抱怨、借口或不想做其他事，也归为 0 (正常)。
7. 🔥 关键规则B：对【闲聊话题】的个人喜好表达属于正常交流，归为 0 (正常)：
   - "不喜欢看电影/电视剧" → 0 (正常表达偏好)
   - "不爱出去玩/不想去公园" → 0 (正常表达偏好)  
   - "不太爱吃甜的" → 0 (正常表达偏好)
   - "对XX不感兴趣" → 0 (正常表达偏好)
   只有对【评估任务】（如拒绝做计算题、不想记词语）才算回避。

请分析医生的问题和患者的回答，输出JSON格式：
{"label": 0-4的数字, "confidence": 0-1的小数, "reason": "简短理由"}
"""
    
    def _llm_predict(self, question: str, answer: str) -> Dict:
        """使用LLM预测"""
        if not self._llm:
            return None
        
        try:
            start_time = time.time()
            
            user_prompt = f"医生问题：{question}\n患者回答：{answer}"
            
            msg = self._llm.invoke([
                {"role": "system", "content": self._llm_system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            
            inference_time = (time.time() - start_time) * 1000
            
            # 解析响应
            content = msg.content if hasattr(msg, "content") else str(msg)
            content = content.strip()
            
            # 尝试解析JSON
            import re
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # 简单的备用解析
                if "label" in content and "4" in content: data = {"label": 4, "confidence": 0.8}
                elif "label" in content and "0" in content: data = {"label": 0, "confidence": 0.8}
                elif "label" in content and "1" in content: data = {"label": 1, "confidence": 0.8}
                elif "label" in content and "2" in content: data = {"label": 2, "confidence": 0.8}
                elif "label" in content and "3" in content: data = {"label": 3, "confidence": 0.8}
                else: data = {"label": 0, "confidence": 0.5}
            
            label = int(data.get("label", 0))
            confidence = float(data.get("confidence", 0.7))
            
            return {
                "label": label,
                "label_name": LABEL_NAMES.get(label, "normal"),
                "label_zh": LABEL_ZH.get(label, "正常配合"),
                "confidence": confidence,
                "inference_time_ms": inference_time
            }
            
        except Exception as e:
            print(f"[ResistanceTool] ❌ LLM预测失败: {e}")
            return None

    def _quick_rule_predict(self, question: str, answer: str) -> Optional[Dict]:
        """
        规则优先拦截：
        1) 请求重复（最高优先级）
        2) 明确拒答/拒测（直接视为抵抗）
        """
        text = (answer or "").strip()
        if not text:
            return None

        norm = re.sub(r"\s+", "", text)

        repeat_patterns = [
            r"没听清", r"再说一遍", r"重复(一遍|一下)?", r"什么意思",
            r"你说什么", r"你刚才说什么", r"说啥", r"听不清", r"没听明白", r"没听懂"
        ]
        if any(re.search(p, norm) for p in repeat_patterns):
            return {
                "label": 4,
                "label_name": LABEL_NAMES[4],
                "label_zh": LABEL_ZH[4],
                "confidence": 0.98,
                "inference_time_ms": 0.0,
                "method": "rule",
            }

        refusal_patterns = [
            r"不想回答", r"不回答", r"不想说", r"不说(了)?", r"不想做", r"不做(了|这个)?",
            r"别问(了)?", r"不聊(了)?", r"拒绝", r"不愿意", r"不配合", r"不想测", r"不测了"
        ]
        if any(re.search(p, norm) for p in refusal_patterns):
            return {
                "label": 3,  # 直接触发抵抗流程
                "label_name": LABEL_NAMES[3],
                "label_zh": "明确拒绝",
                "confidence": 0.95,
                "inference_time_ms": 0.0,
                "method": "rule",
            }

        return None
    
    def _run(self, question: str = "", answer: str = "", language: str = "zh", **kwargs) -> str:
        """执行抵抗情绪检测"""
        from src.utils.tool_logger import ToolLogger
        
        logger = ToolLogger("ResistanceTool")
        logger.start(question=question[:30] if question else "N/A", answer=answer[:50] if answer else "N/A")
        
        # 容错处理
        if question and question.strip().startswith('{'):
            try:
                parsed = json.loads(question.strip())
                question = parsed.get('question', '')
                answer = parsed.get('answer', '')
            except Exception:
                pass
        
        if not answer and 'answer' in kwargs:
            answer = kwargs['answer']
        if not question and 'question' in kwargs:
            question = kwargs['question']
        
        if not answer or answer.strip() == "":
            return json.dumps({
                "is_resistant": False,
                "category": "none",
                "category_zh": "无",
                "confidence": 0.0,
                "rationale": "参数错误：未提供患者回答",
                "inference_time_ms": 0,
                "method": "none"
            }, ensure_ascii=False)
        
        # 规则优先，避免明显拒答被 LLM 误判成“小抱怨”
        result = self._quick_rule_predict(question, answer)
        if result is None:
            result = self._llm_predict(question, answer)
        
        if result is None:
            # 兜底
            print("[ResistanceTool] ⚠️ LLM检测失败，返回默认结果")
            return json.dumps({
                "is_resistant": False,
                "category": "normal",
                "category_zh": "正常配合",
                "confidence": 0.5,
                "rationale": "检测失败，返回默认结果",
                "inference_time_ms": 0,
                "method": "fallback"
            }, ensure_ascii=False)
        
        # 构建返回结果
        pred_label = result["label"]
        # 🔥 只有 distress (2) 和 hostility (3) 触发抵抗处理
        # avoidance (1) 只是小抱怨，不触发抵抗 
        is_resistant = pred_label in [2, 3]  # 只有情绪困扰和主动对抗才算抵抗
        
        output = ResistanceDetectionResult(
            is_resistant=is_resistant,
            category=result["label_name"],
            category_zh=result["label_zh"],
            confidence=result["confidence"],
            rationale=LABEL_DESC[pred_label],
            inference_time_ms=result.get("inference_time_ms", 0),
            method=result.get("method", "llm")
        )
        
        status = "抵抗" if is_resistant else "正常"
        logger.end(
            结果=f"{status}/{result['label_zh']}",
            置信度=f"{result['confidence']:.2%}",
            方法=result.get("method", "llm")
        )
        
        return json.dumps(output.model_dump(), ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async not supported")


# 便捷函数
def detect_resistance(text: str, question: str = "") -> Dict:
    """便捷函数：检测文本的抵抗情绪类型"""
    tool = ResistanceDetectionTool(use_local=True)
    result_json = tool._run(question=question, answer=text)
    return json.loads(result_json)


if __name__ == "__main__":
    print("=" * 50)
    print("🧪 抵抗情绪检测工具测试（纯 LLM 方案）")
    print("=" * 50)
    
    tool = ResistanceDetectionTool(use_local=True)
    
    test_cases = [
        # 正常配合
        ("可以聊天气吗？", "好的，我试试"),
        ("请问今天星期几？", "我想想...星期三吧"),
        # 消极回避
        ("请问现在几点了？", "别问了，我记性不好丢人"),
        ("请问您叫什么名字？", "我脑子好着呢，不用你管"),
        ("3加5等于几？", "太累了，不想答了"),
        # 情绪困扰
        ("请做这个动作", "你是谁派来的"),
        # 主动对抗
        ("请复述刚才的词语", "这个太难了换一个"),
        ("请跟我读...", "问什么问烦不烦"),
        ("你还记得我叫什么吗？", "我刚才不是说了吗"),
        # 🔥 请求重复 (新增)
        ("请记住这三个词：苹果、电视、汽车", "什么？"),
        ("今天是几月几号？", "没听清你说的"),
        ("请跟我算100减7", "再说一遍好吗"),
        ("您家住在哪里？", "啊？你说啥"),
        ("请复述这句话", "能重复一下吗"),
    ]
    
    for q, a in test_cases:
        result = tool._run(question=q, answer=a)
        data = json.loads(result)
        status = "抵抗" if data["is_resistant"] else "正常"
        print(f"[{status}] [{data['category_zh']}] ({data['confidence']:.2%}) Q:{q} A:{a}")
