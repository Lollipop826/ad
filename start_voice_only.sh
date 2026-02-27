#!/bin/bash

# 加载存储配置
if [ -f ".env.storage" ]; then
    source .env.storage
    echo "✅ 已加载外部存储配置"
fi

echo "======================================"
echo "🎤 启动 WebSocket 语音服务 (端口 8502)"
echo "======================================"

# 清理旧进程
echo "🔪 清理旧进程..."
pkill -9 -f voice_server.py 2>/dev/null
sleep 2
echo "✅ 旧进程已清理"

# 进入项目目录
cd /root/autodl-tmp/langchain-mcp-adapters-main

# 启动 WebSocket 服务
echo ""
echo "🚀 启动 WebSocket 服务..."
mkdir -p /root/autodl-tmp/tmp

# 设置环境变量
export PYTHONPATH="${PWD}:${PWD}/ZipVoice-master:${PYTHONPATH}"
export LD_LIBRARY_PATH="/root/miniconda3/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
export CUDA_HOME="/usr/local/cuda-11.8"
export HF_ENDPOINT="https://hf-mirror.com"

# 解决 k2 与其他 CUDA 模型的内存冲突
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u voice_server.py 2>&1 | tee /root/autodl-tmp/tmp/voice_server.log

echo ""
echo "❌ 服务已停止"
