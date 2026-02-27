"""
预先下载所有需要的HuggingFace模型到本地
这样就不会每次启动都尝试联网下载了
"""

import os
from pathlib import Path

def download_models():
    """下载所需的所有模型"""
    
    print("=" * 60)
    print("开始下载HuggingFace模型...")
    print("=" * 60)
    
    # 设置缓存目录
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"\n缓存目录: {cache_dir}")
    
    models = [
        "BAAI/bge-m3",              # 嵌入模型
        "BAAI/bge-reranker-base",   # 重排序模型
    ]
    
    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] 下载模型: {model_name}")
        print("-" * 60)
        
        try:
            # 下载嵌入模型
            if "reranker" not in model_name:
                print(f"  → 使用 sentence-transformers 下载...")
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name)
                print(f"  ✓ {model_name} 下载成功!")
                print(f"    本地路径: {model._model_card_vars.get('model_path', 'N/A')}")
            else:
                # 下载重排序模型
                print(f"  → 使用 CrossEncoder 下载...")
                from sentence_transformers import CrossEncoder
                model = CrossEncoder(model_name)
                print(f"  ✓ {model_name} 下载成功!")
                
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            print(f"  提示: 如果网络问题，可以使用镜像站点")
            continue
    
    print("\n" + "=" * 60)
    print("✅ 所有模型下载完成!")
    print("=" * 60)
    
    # 检查模型是否存在
    print("\n检查已下载的模型:")
    import glob
    pattern = str(cache_dir / "models--*")
    downloaded = glob.glob(pattern)
    
    for model_dir in downloaded:
        model_name = Path(model_dir).name.replace("models--", "").replace("--", "/")
        print(f"  ✓ {model_name}")
    
    if not downloaded:
        print("  未找到已下载的模型")
    
    print("\n" + "=" * 60)
    print("下一步:")
    print("  模型已缓存到本地，后续使用会自动从缓存加载")
    print("  如果还是慢，可以设置HF_ENDPOINT环境变量使用镜像")
    print("=" * 60)

if __name__ == "__main__":
    download_models()

