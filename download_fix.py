import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

print("🚀 开始下载 Embedding 模型 (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)...")
try:
    snapshot_download(
        repo_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        resume_download=True
    )
    print("✅ 模型下载完成！")
except Exception as e:
    print(f"❌ 下载失败: {e}")
