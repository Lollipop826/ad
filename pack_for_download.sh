#!/bin/bash

# ============================================================
# 打包项目文件（排除模型文件）用于下载到 Mac
# ============================================================

PROJECT_DIR="/root/autodl-tmp/langchain-mcp-adapters-main"
OUTPUT_DIR="/root/autodl-tmp/langchain-mcp-adapters-main_packed"
ARCHIVE_NAME="langchain-mcp-adapters-main_no_models_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "=========================================="
echo "📦 打包项目文件（排除模型）"
echo "=========================================="
echo "项目目录: $PROJECT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "压缩包: $ARCHIVE_NAME"
echo ""

# 创建临时目录
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_DIR"

echo "[1/3] 复制文件（排除模型和缓存）..."

# 使用 rsync 复制，排除模型文件
rsync -av \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.pth' \
  --exclude='*.pt' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.onnx' \
  --exclude='*.whl' \
  --exclude='venv/' \
  --exclude='.venv/' \
  --exclude='.git/' \
  --exclude='.DS_Store' \
  --exclude='*.log' \
  --exclude='*.log.*' \
  --exclude='tmp/' \
  --exclude='temp/' \
  --exclude='cache/' \
  --exclude='.cache/' \
  --exclude='models/' \
  --exclude='checkpoints/' \
  --exclude='ckpt/' \
  --exclude='pretrained_models/' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  --exclude='data/speakers/*.npz' \
  --exclude='data/voice_calls/' \
  --exclude='data/conversations/*.json' \
  --exclude='data/conversations/*.jsonl' \
  --exclude='data/scores/*.json' \
  --exclude='data/mmse_scores/' \
  --exclude='kb/chunks_*/' \
  --exclude='kb/mk/' \
  --exclude='kb/mk_final_cleaned/' \
  --exclude='.run_logs/' \
  --exclude='ZipVoice-master/egs/' \
  --exclude='ZipVoice-master/runtime/' \
  --exclude='PaddleSpeech/audio/' \
  --exclude='PaddleSpeech/dataset/' \
  --exclude='PaddleSpeech/examples/' \
  --exclude='PaddleSpeech/tests/' \
  --exclude='PaddleSpeech/demos/' \
  --exclude='PaddleSpeech/docs/' \
  --exclude='PaddleSpeech/docker/' \
  --exclude='Paddle-3.2.0/' \
  --exclude='~/.*' \
  --exclude='*.tar.gz' \
  --exclude='*.zip' \
  --exclude='node_modules/' \
  . "$OUTPUT_DIR/"

echo ""
echo "[2/3] 创建压缩包..."
cd "$(dirname "$OUTPUT_DIR")"
tar -czf "$ARCHIVE_NAME" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"

# 计算大小
SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)

echo ""
echo "[3/3] 完成！"
echo ""
echo "=========================================="
echo "✅ 打包完成"
echo "=========================================="
echo "压缩包位置: $ARCHIVE_NAME"
echo "文件大小: $SIZE"
echo ""
echo "下载方式："
echo "1. 使用 scp:"
echo "   scp root@您的服务器IP:$ARCHIVE_NAME ~/Downloads/"
echo ""
echo "2. 或使用 rsync:"
echo "   rsync -avzP root@您的服务器IP:$ARCHIVE_NAME ~/Downloads/"
echo ""
echo "3. 在 Mac 上解压:"
echo "   cd ~/Downloads && tar -xzf $(basename $ARCHIVE_NAME)"
echo "=========================================="

