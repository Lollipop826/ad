#!/usr/bin/env python3
"""
重建向量数据库脚本
修复Chroma向量数据库损坏问题
"""

import os
import sys
from pathlib import Path

# 强制使用本地缓存的模型，禁用网络访问
os.environ["HF_HUB_OFFLINE"] = "1"  # HuggingFace Hub 离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Transformers 离线模式  
os.environ["HF_DATASETS_OFFLINE"] = "1"  # Datasets 离线模式

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.tools.retrieval.paragraph_retrieval import build_paragraph_vector_db


def main():
    print("="*80)
    print("🔧 开始重建向量数据库")
    print("="*80)
    
    # 配置
    base_dir = Path(__file__).resolve().parent
    chunks_dir = str(base_dir / "kb" / "chunks_semantic_per_file")
    persist_dir = str(base_dir / "kb" / ".chroma_semantic")
    collection_name = "ad_kb_semantic"
    # 使用本地缓存的模型路径
    embedding_model = "/root/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
    device = "cuda"
    
    print(f"\n📂 源数据目录: {chunks_dir}")
    print(f"💾 向量数据库目录: {persist_dir}")
    print(f"📦 集合名称: {collection_name}")
    print(f"🤖 嵌入模型: {embedding_model}")
    print(f"🖥️  设备: {device}")
    
    # 检查源数据目录
    if not os.path.exists(chunks_dir):
        print(f"\n❌ 错误: 源数据目录不存在: {chunks_dir}")
        return 1
    
    # 统计JSONL文件数量
    jsonl_files = []
    for root, _, files in os.walk(chunks_dir):
        for name in files:
            if name.lower().endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, name))
    
    print(f"\n📊 找到 {len(jsonl_files)} 个JSONL文件")
    
    if len(jsonl_files) == 0:
        print("⚠️  警告: 没有找到JSONL文件")
        return 1
    
    # 显示文件列表（前5个）
    print("\n📄 文件示例:")
    for i, path in enumerate(jsonl_files[:5], 1):
        filename = os.path.basename(path)
        print(f"   {i}. {filename}")
    if len(jsonl_files) > 5:
        print(f"   ... 以及其他 {len(jsonl_files) - 5} 个文件")
    
    print("\n⏳ 开始构建向量数据库...")
    print("   (这可能需要几分钟，请耐心等待)")
    
    try:
        # 构建向量数据库
        num_docs = build_paragraph_vector_db(
            chunks_dir=chunks_dir,
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            device=device,
            wipe=True  # 清除旧数据库
        )
        
        print(f"\n✅ 成功！")
        print(f"📊 已索引 {num_docs} 个文档段落")
        print(f"💾 向量数据库已保存到: {persist_dir}")
        print("\n" + "="*80)
        print("🎉 向量数据库重建完成！")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 错误: 构建失败")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
