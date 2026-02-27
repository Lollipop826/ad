#!/usr/bin/env python3
"""
测试任务选择逻辑
手动输入对话内容，查看LLM选择的下一个任务
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import random

# 任务描述
TASK_DESCRIPTIONS = {
    "persona_collect_1": "了解兴趣爱好（聊喜欢做什么）",
    "persona_collect_2": "了解生活习惯（聊作息、饮食）",
    "orientation_time_weekday": "聊今天星期几",
    "orientation_time_date_month_season": "聊日期、月份或季节",
    "orientation_place_city_district": "聊所在城市、区域",
    "registration_3words": "让对方记住3个词",
    "recall_3words": "问刚才记的3个词",
    "language_naming_watch": "看图说出手表名称",
    "language_naming_pencil": "看图说出铅笔名称",
    "language_repetition_sentence": "复述一句话",
    "language_reading_close_eyes": "读字并做动作（闭眼）",
    "language_3step_action": "按指令做三步动作",
    "attention_calc_life_math": "简单算术（如100-7）",
    "buffer_chat": "随便聊聊",
}

# 所有可选任务
ALL_TASKS = [
    "persona_collect_1",
    "persona_collect_2", 
    "orientation_time_weekday",
    "orientation_time_date_month_season",
    "orientation_place_city_district",
    "registration_3words",
    "recall_3words",
    "language_naming_watch",
    "language_naming_pencil",
    "language_repetition_sentence",
    "attention_calc_life_math",
    "buffer_chat",
]


def llm_select_task(candidates: list, chat_history: list, llm):
    """
    让LLM根据对话上下文选择下一个任务
    """
    # 构建候选任务列表（带序号）
    options = []
    option_keywords = {}
    for i, task_id in enumerate(candidates):
        desc = TASK_DESCRIPTIONS.get(task_id, task_id)
        options.append(f"{i+1}. {desc}")
        option_keywords[i] = desc
    
    # 获取最近对话
    recent_chat = ""
    for msg in chat_history[-4:]:
        role = "对方" if msg.get('role') == 'user' else "你"
        content = msg.get('content', '')[:50]
        if content:
            recent_chat += f"{role}：{content}\n"
    
    # 系统消息
    system_message = """你是一个话题选择助手。你的任务是从给定选项中选择一个最自然的话题。

【重要】你必须且只能回复一个阿拉伯数字，如：1 或 2 或 3
【禁止】不要回复任何其他内容，不要解释原因，不要加任何文字"""

    # 用户消息
    user_message = f"""根据对话选择下一个话题。

对话上下文：
{recent_chat if recent_chat else '（刚开始聊天，选择开场话题）'}

可选话题：
{chr(10).join(options)}

回复数字（1-{len(candidates)}）："""

    print("\n" + "="*60)
    print("📤 发送给LLM的消息：")
    print("-"*60)
    print(f"[System] {system_message}")
    print(f"\n[User] {user_message}")
    print("="*60)
    
    try:
        response = llm.invoke([
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ])
        
        if hasattr(response, 'content'):
            result = response.content.strip()
        else:
            result = str(response).strip()
        
        print(f"\n📥 LLM原始输出: '{result}'")
        
        # 解析策略1：直接匹配纯数字
        if result.isdigit():
            idx = int(result) - 1
            if 0 <= idx < len(candidates):
                selected = candidates[idx]
                print(f"✅ 解析成功(纯数字): {selected} ({TASK_DESCRIPTIONS.get(selected, '')})")
                return selected
        
        # 解析策略2：提取第一个数字
        match = re.search(r'^[^\d]*(\d+)', result)
        if match:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(candidates):
                selected = candidates[idx]
                print(f"✅ 解析成功(提取数字): {selected} ({TASK_DESCRIPTIONS.get(selected, '')})")
                return selected
        
        # 解析策略3：匹配中文数字
        chinese_num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, 
                          '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        for cn, num in chinese_num_map.items():
            if cn in result:
                idx = num - 1
                if 0 <= idx < len(candidates):
                    selected = candidates[idx]
                    print(f"✅ 解析成功(中文数字): {selected} ({TASK_DESCRIPTIONS.get(selected, '')})")
                    return selected
        
        # 兜底
        selected = random.choice(candidates)
        print(f"⚠️ 解析失败，随机选择: {selected}")
        return selected
        
    except Exception as e:
        print(f"❌ LLM调用失败: {e}")
        return random.choice(candidates)


def main():
    print("\n" + "="*70)
    print("🧪 任务选择测试脚本")
    print("="*70)
    print("测试 _llm_select_task 函数，查看LLM如何根据对话选择下一个任务\n")
    
    # 初始化LLM
    print("⏳ 正在初始化LLM...")
    try:
        from src.llm.model_pool import get_pooled_llm, ModelPool
        # 初始化模型池
        ModelPool()
        llm = get_pooled_llm(pool_key='precise')
        print("✅ LLM初始化成功\n")
    except Exception as e:
        print(f"❌ LLM初始化失败: {e}")
        return
    
    # 模拟对话历史
    chat_history = []
    
    # 已完成的任务
    done_tasks = set()
    
    # 选择候选任务模式
    print("📋 候选任务模式：")
    print("1. 使用全部可用任务")
    print("2. 手动指定候选任务")
    mode = input("\n选择模式 (1/2，默认1): ").strip() or "1"
    
    if mode == "2":
        print("\n可选任务：")
        for i, task_id in enumerate(ALL_TASKS):
            print(f"  {i+1}. {task_id}: {TASK_DESCRIPTIONS.get(task_id, '')}")
        selected_ids = input("\n输入要使用的任务序号（逗号分隔，如1,2,3）: ").strip()
        candidates = []
        for s in selected_ids.split(","):
            try:
                idx = int(s.strip()) - 1
                if 0 <= idx < len(ALL_TASKS):
                    candidates.append(ALL_TASKS[idx])
            except:
                pass
        if not candidates:
            candidates = ALL_TASKS[:5]
    else:
        candidates = [t for t in ALL_TASKS if t not in done_tasks]
    
    print(f"\n📌 当前候选任务: {candidates}\n")
    
    print("-"*70)
    print("开始对话模拟（输入 'q' 退出，'clear' 清空历史，'done X' 标记任务完成）")
    print("-"*70)
    
    while True:
        # 显示当前对话历史
        if chat_history:
            print("\n📜 对话历史:")
            for msg in chat_history[-6:]:
                role = "👤 用户" if msg['role'] == 'user' else "🤖 AI"
                print(f"  {role}: {msg['content'][:60]}...")
        
        # 输入AI的上一句话（如果有）
        ai_msg = input("\n🤖 AI说（可选，直接回车跳过）: ").strip()
        if ai_msg == 'q':
            break
        if ai_msg == 'clear':
            chat_history = []
            print("✅ 对话历史已清空")
            continue
        if ai_msg.startswith('done '):
            task_to_done = ai_msg[5:].strip()
            if task_to_done in ALL_TASKS:
                done_tasks.add(task_to_done)
                candidates = [t for t in candidates if t != task_to_done]
                print(f"✅ 已标记 {task_to_done} 为完成")
            continue
        
        if ai_msg:
            chat_history.append({"role": "assistant", "content": ai_msg})
        
        # 输入用户回答
        user_msg = input("👤 用户说: ").strip()
        if user_msg == 'q':
            break
        if user_msg == 'clear':
            chat_history = []
            print("✅ 对话历史已清空")
            continue
        
        if user_msg:
            chat_history.append({"role": "user", "content": user_msg})
        
        # 调用LLM选择任务
        print("\n🔄 正在调用LLM选择下一个任务...")
        selected = llm_select_task(candidates, chat_history, llm)
        
        print(f"\n🎯 LLM选择的下一个任务: {selected}")
        print(f"   描述: {TASK_DESCRIPTIONS.get(selected, '未知')}")
        print("-"*70)


if __name__ == "__main__":
    main()
