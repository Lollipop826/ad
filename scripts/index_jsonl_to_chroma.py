import os
import json
import argparse
import shutil
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata


def clean_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """清理元数据，确保Chroma兼容性"""
    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list):
            # 将list转换为逗号分隔的字符串
            cleaned[key] = ", ".join(str(v) for v in value if v)
        elif isinstance(value, dict):
            # 将dict转换为JSON字符串
            cleaned[key] = json.dumps(value, ensure_ascii=False)
        else:
            # 其他类型转为字符串
            cleaned[key] = str(value)
    return cleaned


def load_documents_from_jsonl_dir(directory: str) -> List[Document]:
    documents: List[Document] = []
    directory = os.path.abspath(directory)
    for root, _, files in os.walk(directory):
        for name in files:
            if not name.lower().endswith(".jsonl"):
                continue
            path = os.path.join(root, name)
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = (record.get("text") or "").strip()
                    if not text:
                        continue
                    metadata = record.get("metadata") or {}
                    if not isinstance(metadata, dict):
                        metadata = {}
                    metadata = dict(metadata)
                    metadata.setdefault("source", path)
                    metadata.setdefault("jsonl_line", line_no)
                    # 清理元数据以确保Chroma兼容性
                    metadata = clean_metadata_for_chroma(metadata)
                    documents.append(Document(page_content=text, metadata=metadata))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Index JSONL chunks into a Chroma vector store (local HF embeddings)")
    parser.add_argument("--chunks_dir", type=str, default=os.path.abspath("kb/chunks_md"))
    parser.add_argument("--persist_dir", type=str, default=os.path.abspath("kb/.chroma"))
    parser.add_argument("--collection", type=str, default="ad_kb")
    parser.add_argument("--hf_model", type=str, default="BAAI/bge-m3")
    parser.add_argument("--hf_device", type=str, default="cpu", help="Device for HF model: cpu|cuda|mps")
    parser.add_argument("--normalize", action="store_true", help="Normalize embeddings (cosine-sim friendly)")
    parser.add_argument("--no_wipe", action="store_true")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for adding documents")
    args = parser.parse_args()

    print("=" * 60)
    print("📚 开始构建向量索引...")
    print("=" * 60)
    
    print(f"\n[1/5] 📂 加载文档...")
    docs = load_documents_from_jsonl_dir(args.chunks_dir)
    print(f"✅ 成功加载 {len(docs)} 个文档块")

    if not args.no_wipe and os.path.exists(args.persist_dir):
        print(f"\n[2/5] 🗑️  清理旧索引: {args.persist_dir}")
        shutil.rmtree(args.persist_dir)
        print("✅ 清理完成")
    else:
        print(f"\n[2/5] ⏭️  跳过清理（保留现有索引）")

    print(f"\n[3/5] 🤖 加载 Embedding 模型: {args.hf_model}")
    print(f"      设备: {args.hf_device}")
    print(f"      归一化: {args.normalize}")
    model_kwargs = {"device": args.hf_device}
    encode_kwargs = {"normalize_embeddings": bool(args.normalize)}
    embeddings = HuggingFaceEmbeddings(model_name=args.hf_model, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    print("✅ 模型加载完成")
    
    print(f"\n[4/5] 🔧 初始化向量数据库...")
    vectordb = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=args.persist_dir,
    )
    print("✅ 数据库初始化完成")
    
    if docs:
        print(f"\n[5/5] 🚀 开始索引文档（批次大小: {args.batch_size}）...")
        batch_size = args.batch_size
        total_batches = (len(docs) + batch_size - 1) // batch_size
        
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"      处理批次 {batch_num}/{total_batches} ({len(batch)} 个文档)...", end=" ", flush=True)
            vectordb.add_documents(batch)
            print("✅")
        
        print(f"\n      💾 持久化向量数据库...")
        vectordb.persist()
        print("      ✅ 持久化完成")
    
    print("\n" + "=" * 60)
    print("🎉 索引构建完成！")
    print("=" * 60)
    print(f"📊 统计信息:")
    print(f"   - 文档数量: {len(docs)}")
    print(f"   - 集合名称: {args.collection}")
    print(f"   - 存储位置: {args.persist_dir}")
    print(f"   - Embedding模型: {args.hf_model}")
    print("=" * 60)


if __name__ == "__main__":
    main()


