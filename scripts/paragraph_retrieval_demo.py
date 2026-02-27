import argparse
from src.tools.retrieval.paragraph_retrieval import paragraph_retrieval


def main():
    parser = argparse.ArgumentParser(description="段落级检索Demo - 测试双编码器检索效果")
    parser.add_argument("--query", type=str, required=True, help="查询语句")
    parser.add_argument("--persist_dir", type=str, default="kb/.chroma", help="向量数据库路径")
    parser.add_argument("--collection", type=str, default="ad_kb", help="集合名")
    parser.add_argument("--k", type=int, default=10, help="返回结果数量")
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="嵌入模型")
    parser.add_argument("--force_download", action="store_true", help="强制下载模型")
    args = parser.parse_args()

    print(f"查询语句: {args.query}")
    print(f"向量库路径: {args.persist_dir}")
    print(f"返回Top-{args.k}结果:")
    print("=" * 80)

    # 执行段落级检索
    docs = paragraph_retrieval(
        query=args.query,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.model,
        k=args.k
    )

    # 显示检索结果
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        filename = metadata.get('filename', '未知文件')
        chunk_idx = metadata.get('chunk_index', 'N/A')
        chunk_strategy = metadata.get('chunking_strategy', 'N/A')
        text = doc.page_content
        
        print(f"[{i}] 文件: {filename}")
        print(f"    块索引: {chunk_idx} | 策略: {chunk_strategy}")
        print(f"    内容: {text[:200]}...")
        print("-" * 60)

    print(f"\n检索完成，共返回 {len(docs)} 个相关段落")


if __name__ == "__main__":
    main()
