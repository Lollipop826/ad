#!/usr/bin/env python3
"""
情感安慰工具演示

展示如何在检测到抵抗情绪后生成温和的安慰回复
"""

import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.tools.agent_tools.comfort_response_tool import ComfortResponseTool


def print_section(title: str, char: str = "="):
    """打印分节标题"""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}\n")


def print_result(result_json: str):
    """美化打印结果"""
    result = json.loads(result_json)
    print(f"✅ 成功: {result['success']}")
    print(f"💬 安慰话语: {result['comfort_message']}")
    print(f"🎭 语气: {result['tone']}")
    if result.get('suggestion'):
        print(f"💡 建议: {result['suggestion']}")


def main():
    print_section("🤗 情感安慰工具演示", "=")
    
    # 初始化工具
    print("初始化情感安慰工具...")
    comfort_tool = ComfortResponseTool()
    print("✅ 工具初始化完成\n")
    
    # 测试案例1: 拒绝配合（refusal）
    print_section("📋 测试案例 1: 患者拒绝配合", "-")
    print("场景: 患者说「我不想做这个测试了，太麻烦了」")
    
    result1 = comfort_tool._run(
        resistance_category="refusal",
        patient_answer="我不想做这个测试了，太麻烦了",
        resistance_reason="患者明确表示不想继续测试",
        patient_age=72,
        patient_name="李奶奶"
    )
    
    print_result(result1)
    
    # 测试案例2: 回避问题（avoidance）
    print_section("📋 测试案例 2: 患者回避问题", "-")
    print("场景: 患者说「这个...我也说不清楚」")
    
    result2 = comfort_tool._run(
        resistance_category="avoidance",
        patient_answer="这个...我也说不清楚",
        resistance_reason="患者在回避直接回答问题",
        patient_age=68
    )
    
    print_result(result2)
    
    # 测试案例3: 表现敌意（hostility）
    print_section("📋 测试案例 3: 患者表现敌意", "-")
    print("场景: 患者说「你问这么多干什么？烦不烦啊！」")
    
    result3 = comfort_tool._run(
        resistance_category="hostility",
        patient_answer="你问这么多干什么？烦不烦啊！",
        resistance_reason="患者语气不耐烦，表现出敌意",
        patient_age=75,
        patient_name="王先生"
    )
    
    print_result(result3)
    
    # 测试案例4: 疲惫不堪（fatigue）
    print_section("📋 测试案例 4: 患者很疲惫", "-")
    print("场景: 患者说「我有点累了，不太想说话了...」")
    
    result4 = comfort_tool._run(
        resistance_category="fatigue",
        patient_answer="我有点累了，不太想说话了...",
        resistance_reason="患者表示身体疲惫，注意力下降",
        patient_age=80,
        patient_name="张奶奶"
    )
    
    print_result(result4)
    
    # 测试案例5: 焦虑不安
    print_section("📋 测试案例 5: 患者焦虑不安", "-")
    print("场景: 患者说「我是不是有问题？会不会很严重？」")
    
    result5 = comfort_tool._run(
        resistance_category="avoidance",
        patient_answer="我是不是有问题？会不会很严重？",
        resistance_reason="患者表现出焦虑，担心评估结果",
        patient_age=65
    )
    
    print_result(result5)
    
    print_section("✅ 演示完成", "=")
    print("\n💡 使用建议:")
    print("  1. 在 resistance_detection_tool 检测到抵抗情绪后立即调用")
    print("  2. 将生成的安慰话语作为回复，暂停评估问题")
    print("  3. 根据患者反应决定是否继续评估或建议休息")
    print("  4. 安慰应该真诚、温和，不带任何评估压力\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断执行")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

