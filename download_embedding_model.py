"""
下载 RAG 所需的 embedding 模型（使用 ModelScope 镜像）
"""
import os
from modelscope import snapshot_download
from tqdm import tqdm

print("🚀 开始下载 BAAI/bge-m3 模型（ModelScope 镜像）...")
print("📦 模型大小约 2.3GB，请耐心等待...")
print("=" * 60)

try:
    model_path = snapshot_download(
        model_id='Xorbits/bge-m3',  # ModelScope 上的镜像
        cache_dir='~/.cache/modelscope',
        revision='master'
    )
    print("\n" + "=" * 60)
    print(f"✅ 模型下载成功！")
    print(f"📂 保存位置: {model_path}")
    print("\n现在可以运行 Streamlit 了！")
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n尝试使用 HuggingFace 镜像...")
    
    # 回退到 HF 镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        from huggingface_hub import snapshot_download as hf_download
        model_path = hf_download(
            repo_id="BAAI/bge-m3",
            cache_dir="~/.cache/huggingface",
            resume_download=True,
        )
        print(f"✅ 模型下载成功（HF镜像）！")
        print(f"📂 保存位置: {model_path}")
    except Exception as e2:
        print(f"❌ HF镜像也失败: {e2}")
        print("💡 建议检查网络或手动下载")
