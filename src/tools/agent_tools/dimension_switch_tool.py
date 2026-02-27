"""
维度切换判断工具 - LLM 智能判断是否切换到下一个维度

根据当前维度的完成情况、对话历史、患者表现，智能判断：
1. 当前维度是否已完成所有必要的子项目
2. 是否应该切换到下一个维度
3. 如果不切换，下一个应该问什么
"""

import json
import re
from typing import Type, Optional, List, Any
from pydantic import BaseModel, Field, PrivateAttr
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
import os


# MMSE 各维度的子项目定义
DIMENSION_ITEMS = {
    'orientation': {
        'name': '定向力',
        'items': {
            'year': '今年是哪一年',
            'season': '现在是什么季节',
            'month': '现在是几月',
            'date': '今天是几号',
            'weekday': '今天星期几',
            'province': '我们在哪个省/市',
            'city': '我们在哪个区/县',
            'place': '这是什么地方（医院/社区/家）',
            'floor': '这是几楼',
        },
        'required_count': 5,  # 至少问5个子项目
        'description': '时间定向（年、季、月、日、星期）和地点定向（省、区、地点、楼层）'
    },
    'registration': {
        'name': '即时记忆',
        'items': {
            'three_words': '复述三个词'
        },
        'required_count': 1,
        'description': '让患者复述三个不相关的词'
    },
    'attention_calculation': {
        'name': '注意力与计算',
        'items': {
            'calc_1': '100-7=93',
            'calc_2': '93-7=86',
            'calc_3': '86-7=79',
            'calc_4': '79-7=72',
            'calc_5': '72-7=65',
        },
        'required_count': 5,  # 最多5次
        'early_stop': 'consecutive_failures >= 2',  # 连续错2次可提前结束
        'description': '100连续减7，共5次'
    },
    'recall': {
        'name': '延迟回忆',
        'items': {
            'recall_words': '回忆之前的三个词'
        },
        'required_count': 1,
        'description': '回忆之前记忆的三个词'
    },
    'language': {
        'name': '语言',
        'items': {
            'naming_watch': '命名：手表',
            'naming_pencil': '命名：铅笔',
            'repetition': '复述句子',
            'command': '三步指令',
            'reading': '阅读理解（闭上眼睛）',
            'writing': '书写句子',
        },
        'required_count': 5,  # 至少完成5个子项目
        'description': '命名、复述、指令、阅读、书写'
    },
    'copy': {
        'name': '构图(临摹)',
        'items': {
            'pentagon': '临摹两个相交的五边形'
        },
        'required_count': 1,
        'description': '临摹两个相交的五边形'
    }
}


class DimensionSwitchToolArgs(BaseModel):
    """维度切换判断工具参数"""
    current_dimension_id: str = Field(
        ..., 
        description="当前维度ID: orientation, registration, attention_calculation, recall, language, copy"
    )
    conversation_history: str = Field(
        ...,
        description="最近的对话历史（JSON格式），用于分析已问过的问题"
    )
    last_answer_correct: Optional[bool] = Field(
        default=None,
        description="最近一次回答是否正确（用于计算连续错误）"
    )
    consecutive_failures: int = Field(
        default=0,
        description="连续错误次数（用于 attention_calculation 提前终止）"
    )


class DimensionSwitchTool(BaseTool):
    """
    维度切换判断工具
    
    使用 LLM 分析对话历史，判断：
    1. 当前维度已完成哪些子项目
    2. 是否应该切换到下一个维度
    3. 如果不切换，建议下一个问什么
    """
    
    name: str = "dimension_switch_judge"
    description: str = """
    判断是否应该切换到下一个 MMSE 评估维度。
    
    分析对话历史，识别已完成的子项目，决定是否切换维度。
    """
    args_schema: Type[BaseModel] = DimensionSwitchToolArgs
    
    _llm: Any = PrivateAttr()
    _use_local: bool = PrivateAttr()
    
    def __init__(self, use_local: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._use_local = use_local
        
        if use_local:
            from src.llm.model_pool import get_pooled_llm
            # ⚡ 用 0.5B 小模型，速度快 5-10 倍
            self._llm = get_pooled_llm(pool_key='small_eval')
            print("[DimensionSwitchTool] 🏠 使用本地模型 (small_eval - 0.5B 快速)")
        else:
            from src.llm.http_client_pool import get_siliconflow_chat_openai
            self._llm = get_siliconflow_chat_openai(
                model="Qwen/Qwen2.5-7B-Instruct",
                temperature=0.1,  # 低温度，更确定性
                max_tokens=300,
                timeout=10,
                max_retries=1,
            )
    
    def _extract_response(self, response) -> str:
        """从 LLM 响应中提取文本"""
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def _run(
        self,
        current_dimension_id: str,
        conversation_history: str,
        last_answer_correct: Optional[bool] = None,
        consecutive_failures: int = 0,
    ) -> str:
        """
        判断是否应该切换维度
        
        Returns:
            JSON 字符串，包含：
            - should_switch: 是否切换
            - completed_items: 已完成的子项目列表
            - next_item: 如果不切换，建议下一个问什么
            - reason: 判断理由
        """
        
        # 获取当前维度信息
        dim_info = DIMENSION_ITEMS.get(current_dimension_id)
        if not dim_info:
            return json.dumps({
                "should_switch": False,
                "error": f"未知维度: {current_dimension_id}"
            }, ensure_ascii=False)
        
        # 简单维度直接规则判断（不需要 LLM）
        if current_dimension_id in ['registration', 'recall', 'copy']:
            # 这些维度只需要做一次，完成就切换
            return json.dumps({
                "should_switch": True,
                "completed_items": list(dim_info['items'].keys()),
                "reason": f"{dim_info['name']}维度只需完成一次，已完成",
                "next_item": None
            }, ensure_ascii=False)
        
        # attention_calculation 特殊处理：连续错误提前终止
        if current_dimension_id == 'attention_calculation':
            if consecutive_failures >= 2:
                return json.dumps({
                    "should_switch": True,
                    "completed_items": [],
                    "reason": "连续错误2次，提前结束计算测试",
                    "next_item": None
                }, ensure_ascii=False)
        
        # 复杂维度（orientation, language, attention_calculation）用 LLM 判断
        items_desc = "\n".join([f"  - {k}: {v}" for k, v in dim_info['items'].items()])
        
        prompt = f"""你是 MMSE 认知评估专家。分析对话历史，判断当前维度是否已完成。

当前维度: {dim_info['name']} ({current_dimension_id})
维度说明: {dim_info['description']}
需要完成的子项目:
{items_desc}

最少需要完成: {dim_info['required_count']} 个子项目

对话历史:
{conversation_history[-1500:]}

判断规则：
- 如果已完成的子项目数量 >= {dim_info['required_count']}，则 should_switch = true
- 否则 should_switch = false，并建议下一个要问的子项目

返回JSON格式:
{{"should_switch": true/false, "completed_items": ["item_id1", "item_id2"], "next_item": "下一个要问的item_id或null", "reason": "判断理由"}}

只返回JSON。"""

        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            response_text = self._extract_response(response).strip()
            
            # 提取 JSON
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # 验证必要字段
                if 'completed_items' not in result:
                    result['completed_items'] = []
                if 'next_item' not in result:
                    result['next_item'] = None
                
                # ⭐ 强制规则：完成数量达标则必须切换
                completed_count = len(result.get('completed_items', []))
                if completed_count >= dim_info['required_count']:
                    result['should_switch'] = True
                    result['reason'] = f"已完成 {completed_count} 项，达到最少要求 {dim_info['required_count']} 项"
                elif 'should_switch' not in result:
                    result['should_switch'] = False
                    
                print(f"[DimensionSwitch] 📊 {dim_info['name']}: 已完成 {completed_count}/{dim_info['required_count']} 项, 切换={result['should_switch']}")
                
                return json.dumps(result, ensure_ascii=False)
            else:
                # JSON 解析失败，使用保守策略
                print(f"[DimensionSwitch] ⚠️ JSON解析失败，使用保守策略")
                return json.dumps({
                    "should_switch": False,
                    "completed_items": [],
                    "next_item": None,
                    "reason": "LLM返回格式错误，继续当前维度"
                }, ensure_ascii=False)
                
        except Exception as e:
            print(f"[DimensionSwitch] ❌ 错误: {e}")
            return json.dumps({
                "should_switch": False,
                "error": str(e),
                "reason": "判断出错，继续当前维度"
            }, ensure_ascii=False)
    
    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("DimensionSwitchTool 不支持异步调用")
