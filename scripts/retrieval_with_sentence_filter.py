#!/usr/bin/env python3
"""
完整的检索Pipeline：段落检索 + 句子级过滤

工作流程：
1. 使用双编码器进行段落级检索，召回Top-K个候选段落
2. 对每个段落应用句子级过滤器，只保留相关句子
3. 返回过滤后的结果
"""

import os
import sys
import argparse
from typing import List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from langchain_core.documents import Document
from src.tools.retrieval.paragraph_retrieval import paragraph_retrieval
from src.tools.retrieval.sentence_filter import SentenceFilter


def retrieval_with_sentence_filter(
    query: str,
    persist_dir: str,
    collection_name: str = "ad_kb_semantic",
    embedding_model: str = "BAAI/bge-m3",
    rerank_model: str = "BAAI/bge-reranker-base",
    k: int = 20,
    sentence_threshold: float = 0.3,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    完整的检索Pipeline
    
    Args:
        query: 查询语句
        persist_dir: 向量数据库路径
        collection_name: 集合名称
        embedding_model: 双编码器模型
        rerank_model: 交叉编码器模型（用于句子过滤）
        k: 召回的段落数量
        sentence_threshold: 句子相关性阈值（0-1，越高越严格）
        device: 设备（cpu/cuda）
    
    Returns:
        过滤后的结果列表，每个结果包含：
        - original_text: 原始段落文本
        - filtered_text: 过滤后的文本（只包含相关句子）
        - metadata: 元数据
        - sentence_count: 保留的句子数量
    """
    
    print("=" * 80)
    print("🔍 第一步：段落级检索（双编码器）")
    print("=" * 80)
    print(f"查询: {query}")
    print(f"召回数量: Top-{k}")
    print()
    
    # 第一步：段落级检索
    docs: List[Document] = paragraph_retrieval(
        query=query,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        device=device,
        k=k,
    )
    
    print(f"✅ 召回 {len(docs)} 个候选段落\n")
    
    print("=" * 80)
    print("🎯 第二步：句子级过滤（交叉编码器）")
    print("=" * 80)
    print(f"过滤模型: {rerank_model}")
    print(f"相关性阈值: {sentence_threshold}")
    print()
    
    # 第二步：句子级过滤
    sentence_filter = SentenceFilter(model_name=rerank_model, device=device)
    
    results = []
    total_kept_sentences = 0
    total_discarded = 0
    
    for i, doc in enumerate(docs, 1):
        original_text = doc.page_content
        filtered_text = sentence_filter.keep(query, original_text, threshold=sentence_threshold)
        
        if not filtered_text:
            total_discarded += 1
            continue
        
        # 统计句子数量
        from src.tools.retrieval.sentence_filter import split_sentences
        original_sentences = split_sentences(original_text)
        filtered_sentences = split_sentences(filtered_text)
        kept_count = len(filtered_sentences)
        total_kept_sentences += kept_count
        
        results.append({
            "rank": len(results) + 1,
            "original_rank": i,
            "original_text": original_text,
            "filtered_text": filtered_text,
            "metadata": doc.metadata,
            "sentence_count": kept_count,
            "original_sentence_count": len(original_sentences),
        })
        
        print(f"[{i}] 保留 {kept_count}/{len(original_sentences)} 个句子")
    
    print()
    print(f"✅ 过滤完成：保留 {len(results)}/{len(docs)} 个段落，共 {total_kept_sentences} 个相关句子")
    print(f"   丢弃 {total_discarded} 个完全不相关的段落")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="段落检索 + 句子级过滤 Pipeline")
    parser.add_argument("--query", type=str, required=True, help="查询语句")
    parser.add_argument("--persist_dir", type=str, default="kb/.chroma_semantic", help="向量数据库路径")
    parser.add_argument("--collection", type=str, default="ad_kb_semantic", help="集合名")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3", help="双编码器模型")
    parser.add_argument("--rerank_model", type=str, default="BAAI/bge-reranker-base", help="交叉编码器模型")
    parser.add_argument("--k", type=int, default=20, help="召回段落数量")
    parser.add_argument("--threshold", type=float, default=0.3, help="句子相关性阈值")
    parser.add_argument("--device", type=str, default="cpu", help="设备: cpu/cuda/mps")
    parser.add_argument("--top_n", type=int, default=5, help="最终显示的结果数量")
    args = parser.parse_args()
    
    results = retrieval_with_sentence_filter(
        query=args.query,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        rerank_model=args.rerank_model,
        k=args.k,
        sentence_threshold=args.threshold,
        device=args.device,
    )
    
    print("\n" + "=" * 80)
    print(f"📊 最终结果 (Top-{min(args.top_n, len(results))})")
    print("=" * 80)
    
    for result in results[:args.top_n]:
        print(f"\n[{result['rank']}] 文件: {result['metadata'].get('filename', '未知')}")
        print(f"    块索引: {result['metadata'].get('chunk_index', 'N/A')}")
        print(f"    保留句子: {result['sentence_count']}/{result['original_sentence_count']}")
        print(f"    过滤后内容:")
        
        # 显示过滤后的句子，每个句子一行
        from src.tools.retrieval.sentence_filter import split_sentences
        sentences = split_sentences(result['filtered_text'])
        for i, sent in enumerate(sentences, 1):
            print(f"      [{i}] {sent}")
        print("-" * 80)
    
    print(f"\n✅ 检索完成！共返回 {len(results)} 个相关段落")


if __name__ == "__main__":
    main()

