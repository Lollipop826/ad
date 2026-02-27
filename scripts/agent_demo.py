#!/usr/bin/env python3
"""
基于Agent的对话系统演示

展示LLM Agent如何自主调用工具完成对话任务
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.agents.screening_agent import ADScreeningAgent
from src.domain.dimensions import MMSE_DIMENSIONS
from src.common.conversation_storage import ConversationStorage


def print_section(title: str, char: str = "="):
    """打印分节标题"""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}\n")


def main():
    print_section("🤖 基于Agent的对话系统演示", "=")
    
    # 1. 初始化Agent
    print("初始化Agent...")
    agent = ADScreeningAgent(temperature=0.3)
    
    print("\n可用工具:")
    for tool in agent.get_tools_info():
        print(f"  - {tool['name']}: {tool['description'][:60]}...")
    
    # 2. 创建会话
    session_id = f"agent_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = ConversationStorage()
    
    patient_profile = {
        "name": "李奶奶",
        "age": 72,
        "sex": "女",
        "education_years": 6,
        "notes": "退休工人，子女反映近期记忆力下降"
    }
    
    storage.create_session(
        session_id=session_id,
        user_id="patient_002",
        profile=patient_profile,
        dimensions=MMSE_DIMENSIONS
    )
    
    print(f"\n创建会话: {session_id}")
    print(f"患者: {patient_profile['name']}, {patient_profile['age']}岁")
    
    # 3. 模拟对话
    print_section("💬 开始对话", "-")
    
    # 第一轮：定向力评估
    print("【轮次 1 - 定向力评估】\n")
    print("👤 患者: 我最近经常搞不清今天是几号")
    
    result1 = agent.process_turn(
        user_input="我最近经常搞不清今天是几号",
        dimension=MMSE_DIMENSIONS[0],  # 定向力
        session_id=session_id,
        patient_profile=patient_profile
    )
    
    print(f"\n🤖 Agent输出:")
    print(f"  {result1['output']}")
    
    print(f"\n📝 工具调用步骤:")
    for i, step in enumerate(result1.get('intermediate_steps', []), 1):
        action, observation = step
        print(f"  {i}. 调用工具: {action.tool}")
        obs_str = str(observation)
        print(f"     观察结果: {obs_str[:100]}...")
    
    print_section("✅ 演示完成", "=")
    print(f"\n会话已保存至: data/conversations/{session_id}.json")
    print("\n💡 说明:")
    print("  - Agent自主决定了调用哪些工具")
    print("  - 可以根据情况灵活调整工具调用顺序")
    print("  - 比硬编码的流程更智能、更灵活")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断执行")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

