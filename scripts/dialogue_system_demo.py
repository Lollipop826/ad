#!/usr/bin/env python3
"""
完整对话系统演示

展示基于知识检索的问题生成流程：
1. 用户回答
2. 生成查询 → 检索知识
3. 基于知识生成下一个问题
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.tools.dialogue.conversation_manager import ConversationManager
from src.domain.dimensions import MMSE_DIMENSIONS


def print_section(title: str, char: str = "="):
    """打印分节标题"""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}\n")


def simulate_conversation():
    """模拟完整对话流程"""
    
    print_section("🏥 阿尔茨海默病初筛对话系统", "=")
    
    # 1. 初始化对话管理器
    manager = ConversationManager(
        storage_dir="data/conversations",
        vector_db_dir="kb/.chroma_semantic",
        collection_name="ad_kb_semantic"
    )
    
    # 2. 创建新会话
    session_id = f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"创建新会话: {session_id}\n")
    
    session = manager.create_session(
        session_id=session_id,
        user_id="patient_001",
        profile={
            "name": "王阿姨",
            "age": 70,
            "sex": "女",
            "education_years": 9,
            "notes": "退休工人，家人反映最近记忆力下降"
        },
        dimensions=MMSE_DIMENSIONS
    )
    
    print(f"患者信息:")
    print(f"  姓名: {session['profile']['name']}")
    print(f"  年龄: {session['profile']['age']}岁")
    print(f"  教育年限: {session['profile']['education_years']}年")
    print(f"  备注: {session['profile']['notes']}")
    
    # 3. 模拟多轮对话
    conversation_scenarios = [
        {
            "dimension": MMSE_DIMENSIONS[0],  # 定向力
            "user_input": "最近经常忘记今天是星期几",
            "emotion": "worried"
        },
        {
            "dimension": MMSE_DIMENSIONS[1],  # 即时记忆
            "user_input": "苹果、桌子、硬币",
            "emotion": "calm"
        },
        {
            "dimension": MMSE_DIMENSIONS[3],  # 延迟回忆
            "user_input": "我有点记不清了，好像有苹果？",
            "emotion": "anxious"
        },
    ]
    
    for i, scenario in enumerate(conversation_scenarios, 1):
        print_section(f"💬 对话轮次 {i}", "-")
        
        print(f"🔹 当前评估维度: {scenario['dimension']['name']}")
        print(f"   描述: {scenario['dimension']['description']}")
        print(f"   优先级: {scenario['dimension']['priority']}\n")
        
        print(f"👤 患者: {scenario['user_input']}")
        if scenario['emotion']:
            print(f"   (情绪: {scenario['emotion']})")
        print()
        
        # 处理这一轮对话
        result = manager.process_turn(
            session_id=session_id,
            user_input=scenario['user_input'],
            current_dimension=scenario['dimension'],
            user_emotion=scenario['emotion']
        )
        
        print(f"🔍 系统处理:")
        print(f"   生成查询: {result['generated_query']}")
        print(f"   检索文档: {result['retrieved_docs']} 个")
        print(f"   关键词: {', '.join(result['query_keywords'])}")
        print()
        
        print(f"🤖 医生: {result['next_question']}")
        print()
        
        # 更新维度状态（实际系统中应该基于答案分析）
        if i == 1:
            manager.update_dimension_status(
                session_id, 
                scenario['dimension']['id'], 
                "known", 
                "可能存在定向力障碍"
            )
    
    # 4. 显示会话摘要
    print_section("📊 会话摘要", "=")
    
    session = manager.get_session(session_id)
    
    print(f"会话ID: {session['session_id']}")
    print(f"患者: {session['profile']['name']}")
    print(f"开始时间: {session['start_time']}")
    print(f"对话轮次: {len(session['dialogue_turns'])}")
    print()
    
    print("对话记录:")
    for turn in session["dialogue_turns"]:
        print(f"\n轮次 {turn['turn_id']}:")
        print(f"  维度: {turn['dimension_name']} ({turn['dimension_id']})")
        print(f"  患者: {turn['user_question']}")
        if turn.get('user_emotion'):
            print(f"  情绪: {turn['user_emotion']}")
        print(f"  查询: {turn['generated_query']}")
        print(f"  医生: {turn['assistant_response']}")
        print(f"  检索: {len(turn['retrieved_documents'])} 个文档")
    
    print()
    
    print("维度评估状态:")
    for dim in session["dimensions"]:
        status_emoji = {
            "unknown": "❓",
            "asking": "🔄",
            "known": "✅",
            "skipped": "⏭️"
        }.get(dim["status"], "❓")
        
        print(f"  {status_emoji} {dim['name']}: {dim['status']}", end="")
        if dim.get("value"):
            print(f" ({dim['value']})")
        else:
            print()
    
    # 5. 导出数据
    output_file = f"data/conversations/{session_id}.jsonl"
    manager.export_session(session_id, output_file)
    
    print_section("✅ 对话演示完成", "=")
    print(f"完整会话已保存至:")
    print(f"  JSON格式: data/conversations/{session_id}.json")
    print(f"  JSONL格式: {output_file}")
    print()


if __name__ == "__main__":
    try:
        simulate_conversation()
    except KeyboardInterrupt:
        print("\n\n中断执行")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

