"""
通用JSON解析工具
用于处理本地模型把整个JSON当作第一个参数传入的问题
"""
import json
from typing import Dict, Any


def parse_tool_input(first_arg: str, fallback_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    解析工具输入参数
    
    Args:
        first_arg: 第一个参数（可能是JSON字符串）
        fallback_dict: 默认值字典
    
    Returns:
        解析后的参数字典
    """
    if not first_arg or not first_arg.strip().startswith('{'):
        return fallback_dict
    
    try:
        # 提取第一个完整JSON对象
        json_str = first_arg.strip()
        brace_count = 0
        json_end = -1
        
        for i, char in enumerate(json_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end > 0:
            json_str = json_str[:json_end]
        
        parsed = json.loads(json_str)
        
        # 合并解析结果和默认值
        result = fallback_dict.copy()
        result.update(parsed)
        
        return result
        
    except Exception as e:
        print(f"[JSONParser] ❌ 解析失败: {e}")
        return fallback_dict
