#!/usr/bin/env python3
"""
段落检索 + 句子高亮（不删除）

保留完整段落，只标注最相关的句子
"""

import os
import sys
import argparse
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from langchain_core.documents import Document
from src.tools.retrieval.paragraph_retrieval import paragraph_retrieval
from src.tools.retrieval.sentence_filter import SentenceFilter, split_sentences


def highlight_relevant_sentences(
    query: str,
    text: str,
    sf: SentenceFilter,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    标注段落中的相关句子，但保留完整内容
    
    Returns:
        {
            'text': 完整文本,
            'sentences': [{'text': ..., 'score': ..., 'is_relevant': ...}, ...],
            'relevant_count': 相关句子数量
        }
    """
    import torch
    
    sentences = split_sentences(text)
    if not sentences:
        return {'text': text, 'sentences': [], 'relevant_count': 0}
    
    pairs = [[query, s] for s in sentences]
    with torch.inference_mode():
        scores = sf.model.predict(pairs).tolist()
    
    sentence_info = []
    relevant_count = 0
    
    for s, score in zip(sentences, scores):
        is_relevant = score >= threshold
        if is_relevant:
            relevant_count += 1
        sentence_info.append({
            'text': s,
            'score': score,
            'is_relevant': is_relevant
        })
    
    return {
        'text': text,
        'sentences': sentence_info,
        'relevant_count': relevant_count
    }


def main():
    parser = argparse.ArgumentParser(description="段落检索 + 句子高亮（保留完整段落）")
    parser.add_argument("--query", type=str, required=True, help="查询语句")
    parser.add_argument("--persist_dir", type=str, default="kb/.chroma_semantic", help="向量数据库路径")
    parser.add_argument("--collection", type=str, default="ad_kb_semantic", help="集合名")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3", help="双编码器模型")
    parser.add_argument("--rerank_model", type=str, default="BAAI/bge-reranker-base", help="交叉编码器模型")
    parser.add_argument("--k", type=int, default=10, help="召回段落数量")
    parser.add_argument("--threshold", type=float, default=0.5, help="句子相关性阈值（用于高亮）")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    parser.add_argument("--top_n", type=int, default=5, help="最终显示的结果数量")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔍 段落检索 + 句子高亮")
    print("=" * 80)
    print(f"查询: {args.query}")
    print(f"召回数量: Top-{args.k}")
    print(f"高亮阈值: {args.threshold}")
    print()
    
    # 段落检索
    docs: List[Document] = paragraph_retrieval(
        query=args.query,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        device=args.device,
        k=args.k,
    )
    
    print(f"✅ 召回 {len(docs)} 个候选段落\n")
    
    # 句子高亮
    sf = SentenceFilter(model_name=args.rerank_model, device=args.device)
    
    results = []
    for i, doc in enumerate(docs, 1):
        highlight_info = highlight_relevant_sentences(
            args.query, 
            doc.page_content, 
            sf, 
            args.threshold
        )
        
        results.append({
            'rank': i,
            'metadata': doc.metadata,
            'highlight_info': highlight_info
        })
    
    print("=" * 80)
    print(f"📊 检索结果 (Top-{min(args.top_n, len(results))})")
    print("=" * 80)
    
    for result in results[:args.top_n]:
        info = result['highlight_info']
        print(f"\n[{result['rank']}] 文件: {result['metadata'].get('filename', '未知')}")
        print(f"    块索引: {result['metadata'].get('chunk_index', 'N/A')}")
        print(f"    相关句子: {info['relevant_count']}/{len(info['sentences'])}")
        print(f"    内容:")
        
        # 显示带高亮标记的完整段落
        for i, sent_info in enumerate(info['sentences'], 1):
            if sent_info['is_relevant']:
                # 高亮相关句子
                print(f"      ⭐ [{sent_info['score']:.3f}] {sent_info['text']}")
            else:
                # 普通显示不太相关的句子
                print(f"         [{sent_info['score']:.3f}] {sent_info['text']}")
        
        print("-" * 80)
    
    print(f"\n✅ 检索完成！")


if __name__ == "__main__":
    main()

