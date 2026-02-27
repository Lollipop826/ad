#!/bin/bash

# 知识库可视化应用启动脚本

echo "📚 启动知识库浏览器"
echo "=================================================="

# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

# 进入项目根目录（重要！）
cd "$(dirname "$0")/.."

echo ""
echo "当前目录: $(pwd)"
echo "正在启动应用..."
python3 -m streamlit run kb_viewer/app.py --server.port 8504

echo ""
echo "=================================================="

