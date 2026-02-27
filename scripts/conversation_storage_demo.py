#!/usr/bin/env python3
"""
对话存储系统示例

演示如何保存完整的对话流程：
1. 用户提问
2. 生成查询
3. 检索知识
4. LLM回答
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.common.conversation_storage import ConversationStorage, RetrievalResult
from src.tools.retrieval.paragraph_retrieval import paragraph_retrieval
from src.tools.retrieval.sentence_filter import SentenceFilter, split_sentences
from src.tools.query_sentence.generator import QuerySentenceGenerator
from src.domain.dimensions import MMSE_DIMENSIONS


def simulate_conversation_turn(
    storage: ConversationStorage,
    session_id: str,
    user_question: str,
    dimension_id: str,
    persist_dir: str = "kb/.chroma_semantic",
    collection_name: str = "ad_kb_semantic",
):
    """模拟一个完整的对话轮次"""
    
    print("=" * 80)
    print(f"💬 对话轮次 - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)
    
    # 1. 用户提问
    print(f"👤 用户: {user_question}")
    
    # 2. 获取当前维度
    dimension = next((d for d in MMSE_DIMENSIONS if d["id"] == dimension_id), None)
    if not dimension:
        print(f"❌ 维度 {dimension_id} 不存在")
        return
    
    print(f"📋 当前维度: {dimension['name']} ({dimension['id']})")
    
    # 3. 生成查询语句
    print("\n🔍 生成检索查询...")
    query_gen = QuerySentenceGenerator()
    
    # 获取历史对话
    session = storage.load_session(session_id)
    history = [
        {"role": turn["user_question"] and "user" or "assistant", 
         "content": turn["user_question"] or turn["assistant_response"]}
        for turn in session["dialogue_turns"]
    ]
    
    query_result = query_gen.generate_query(
        dimension=dimension,
        history=history,
        profile=session.get("profile")
    )
    
    generated_query = query_result["query"]
    print(f"   生成的查询: {generated_query}")
    print(f"   关键词: {', '.join(query_result.get('keywords', []))}")
    
    # 4. 检索知识
    print("\n📚 检索相关知识...")
    docs = paragraph_retrieval(
        query=generated_query,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model="BAAI/bge-m3",  # 必须与建索引时的模型一致
        k=5
    )
    
    # 5. 句子高亮（可选）
    sf = SentenceFilter()
    retrieved_docs: list[RetrievalResult] = []
    
    for i, doc in enumerate(docs[:3], 1):  # 只取前3个
        # 计算句子得分
        import torch
        sentences = split_sentences(doc.page_content)
        pairs = [[generated_query, s] for s in sentences]
        
        with torch.inference_mode():
            scores = sf.model.predict(pairs).tolist()
        
        highlighted = [
            {"text": s, "score": float(score), "is_relevant": score >= 0.6}
            for s, score in zip(sentences, scores)
        ]
        
        retrieved_docs.append({
            "rank": i,
            "text": doc.page_content,
            "metadata": doc.metadata,
            "highlighted_sentences": highlighted
        })
        
        print(f"   [{i}] {doc.metadata.get('filename', '未知')} (块 {doc.metadata.get('chunk_index', '?')})")
    
    # 6. 生成回答（这里简化为使用检索到的第一个段落）
    print("\n🤖 生成回答...")
    # 实际应该调用LLM，这里简化处理
    assistant_response = f"根据检索结果，关于{dimension['name']}：\n\n{docs[0].page_content[:200]}..."
    print(f"   {assistant_response}")
    
    # 7. 保存对话轮次
    print("\n💾 保存对话轮次...")
    storage.add_turn(
        session_id=session_id,
        user_question=user_question,
        generated_query=generated_query,
        query_keywords=query_result.get("keywords", []),
        retrieved_documents=retrieved_docs,
        assistant_response=assistant_response,
        dimension_id=dimension_id,
        dimension_name=dimension["name"],
        retrieval_method="paragraph_with_highlight",
        response_metadata={
            "retrieval_count": len(docs),
            "confidence": query_result.get("confidence", 0.5)
        }
    )
    
    print("✅ 对话轮次已保存")
    print()


def main():
    parser = argparse.ArgumentParser(description="对话存储系统演示")
    parser.add_argument("--session_id", type=str, default=None, help="会话ID（不提供则创建新会话）")
    parser.add_argument("--storage_dir", type=str, default="data/conversations", help="存储目录")
    parser.add_argument("--persist_dir", type=str, default="kb/.chroma_semantic", help="向量库路径")
    args = parser.parse_args()
    
    # 初始化存储
    storage = ConversationStorage(storage_dir=args.storage_dir)
    
    # 创建或加载会话
    if args.session_id:
        session_id = args.session_id
        print(f"📂 加载会话: {session_id}")
        session = storage.load_session(session_id)
    else:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"🆕 创建新会话: {session_id}")
        session = storage.create_session(
            session_id=session_id,
            user_id="demo_user_001",
            profile={
                "name": "张三",
                "age": 72,
                "sex": "男",
                "education_years": 12,
                "notes": "退休教师"
            },
            dimensions=MMSE_DIMENSIONS
        )
    
    print(f"   用户: {session.get('profile', {}).get('name', '未知')}")
    print(f"   已有轮次: {len(session['dialogue_turns'])}")
    print()
    
    # 模拟几轮对话
    print("🎬 开始对话演示...\n")
    
    # 第一轮：定向力
    simulate_conversation_turn(
        storage=storage,
        session_id=session_id,
        user_question="我有时候记不清今天是几号",
        dimension_id="orientation",
        persist_dir=args.persist_dir
    )
    
    # 第二轮：记忆力
    simulate_conversation_turn(
        storage=storage,
        session_id=session_id,
        user_question="经常忘记把东西放在哪里了",
        dimension_id="recall",
        persist_dir=args.persist_dir
    )
    
    # 显示会话摘要
    print("=" * 80)
    print("📊 会话摘要")
    print("=" * 80)
    
    session = storage.load_session(session_id)
    print(f"会话ID: {session['session_id']}")
    print(f"用户: {session.get('profile', {}).get('name', '未知')}")
    print(f"开始时间: {session['start_time']}")
    print(f"对话轮次: {len(session['dialogue_turns'])}")
    print()
    
    for turn in session["dialogue_turns"]:
        print(f"轮次 {turn['turn_id']}:")
        print(f"  问题: {turn['user_question']}")
        print(f"  维度: {turn.get('dimension_name', '未知')}")
        print(f"  查询: {turn['generated_query']}")
        print(f"  检索: {len(turn['retrieved_documents'])} 个文档")
        print()
    
    # 导出为JSONL
    output_file = f"{args.storage_dir}/{session_id}.jsonl"
    storage.export_session_to_jsonl(session_id, output_file)
    print(f"💾 会话已导出至: {output_file}")
    print()
    
    print("=" * 80)
    print("✅ 演示完成！")
    print("=" * 80)
    print(f"\n查看完整会话: {args.storage_dir}/{session_id}.json")
    print(f"查看JSONL格式: {output_file}")


if __name__ == "__main__":
    main()

