"""
使用国内镜像下载HuggingFace模型
速度更快，更稳定
"""

import os
import sys

def download_with_mirror():
    """使用镜像站点下载模型"""
    
    print("=" * 70)
    print("🚀 使用国内镜像下载HuggingFace模型")
    print("=" * 70)
    
    # 设置HuggingFace镜像
    mirror_url = "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = mirror_url
    
    print(f"\n✓ 已设置镜像站点: {mirror_url}")
    print("  (这样下载会快很多!)\n")
    
    models = [
        ("BAAI/bge-m3", "嵌入模型 (1.3GB)"),
        ("BAAI/bge-reranker-base", "重排序模型 (300MB)"),
    ]
    
    total_models = len(models)
    
    for i, (model_name, description) in enumerate(models, 1):
        print("-" * 70)
        print(f"[{i}/{total_models}] 下载: {model_name}")
        print(f"       {description}")
        print("-" * 70)
        
        try:
            if "reranker" in model_name:
                # 重排序模型
                print("  → 初始化 CrossEncoder...")
                from sentence_transformers import CrossEncoder
                import time
                start = time.time()
                model = CrossEncoder(model_name, device="cpu")
                elapsed = time.time() - start
                print(f"  ✅ 下载完成! (耗时 {elapsed:.1f}秒)")
            else:
                # 嵌入模型
                print("  → 初始化 SentenceTransformer...")
                from sentence_transformers import SentenceTransformer
                import time
                start = time.time()
                model = SentenceTransformer(model_name, device="cpu")
                elapsed = time.time() - start
                print(f"  ✅ 下载完成! (耗时 {elapsed:.1f}秒)")
                
        except Exception as e:
            print(f"  ❌ 下载失败: {str(e)}")
            print(f"\n可能的解决方案:")
            print(f"  1. 检查网络连接")
            print(f"  2. 尝试直接从浏览器下载模型文件")
            print(f"  3. 使用梯子访问原站")
            continue
    
    print("\n" + "=" * 70)
    print("✅ 模型下载完成!")
    print("=" * 70)
    
    # 测试模型
    print("\n测试模型是否可用...")
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        
        print("\n1. 测试嵌入模型...")
        embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        test_vec = embed_model.encode("测试文本")
        print(f"   ✓ 嵌入维度: {len(test_vec)}")
        
        print("\n2. 测试重排序模型...")
        rerank_model = CrossEncoder("BAAI/bge-reranker-base", device="cpu")
        score = rerank_model.predict([["查询", "文档"]])[0]
        print(f"   ✓ 重排序得分: {score:.4f}")
        
        print("\n✅ 所有模型测试通过!")
        
    except Exception as e:
        print(f"\n❌ 模型测试失败: {e}")
    
    print("\n" + "=" * 70)
    print("📌 下一步:")
    print("   1. 模型已缓存到 ~/.cache/huggingface/")
    print("   2. 后续使用会自动从缓存加载")
    print("   3. 重启应用即可享受秒开速度!")
    print("=" * 70)

if __name__ == "__main__":
    print("\n⚠️  注意: 首次下载大约需要 1.5GB 空间和几分钟时间")
    print("   但这是一次性的，下载后就永久使用本地缓存!\n")
    
    download_with_mirror()

