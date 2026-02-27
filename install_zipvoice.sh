#!/bin/bash

# ============================================================
# ZipVoice 依赖安装脚本
# ============================================================

echo "=========================================="
echo "  安装 ZipVoice 依赖"
echo "=========================================="

# 设置 HuggingFace 镜像（中国大陆用户）
export HF_ENDPOINT=https://hf-mirror.com

# 安装基础依赖
echo ""
echo "[1/3] 安装基础依赖..."
pip install -q \
    vocos \
    lhotse \
    huggingface_hub \
    safetensors \
    pydub

# 安装文本处理依赖
echo ""
echo "[2/3] 安装文本处理依赖..."
pip install -q \
    cn2an \
    inflect \
    jieba \
    pypinyin

# 安装 piper_phonemize（如果需要）
echo ""
echo "[3/3] 安装语音处理依赖..."
pip install -q --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html piper_phonemize 2>/dev/null || {
    echo "⚠️ piper_phonemize 安装失败（可选依赖，不影响使用）"
}

# 验证安装
echo ""
echo "=========================================="
echo "  验证安装"
echo "=========================================="

python3 -c "
import sys
errors = []

try:
    from vocos import Vocos
    print('✓ Vocos vocoder')
except ImportError as e:
    errors.append(f'✗ Vocos: {e}')

try:
    from lhotse.utils import fix_random_seed
    print('✓ lhotse')
except ImportError as e:
    errors.append(f'✗ lhotse: {e}')

try:
    import safetensors
    print('✓ safetensors')
except ImportError as e:
    errors.append(f'✗ safetensors: {e}')

try:
    from huggingface_hub import hf_hub_download
    print('✓ huggingface_hub')
except ImportError as e:
    errors.append(f'✗ huggingface_hub: {e}')

try:
    import cn2an
    print('✓ cn2an')
except ImportError as e:
    errors.append(f'✗ cn2an: {e}')

try:
    import jieba
    print('✓ jieba')
except ImportError as e:
    errors.append(f'✗ jieba: {e}')

try:
    import pypinyin
    print('✓ pypinyin')
except ImportError as e:
    errors.append(f'✗ pypinyin: {e}')

print()
if errors:
    print('❌ 以下依赖安装失败:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    print('✅ 所有依赖已安装完成！')
"

echo ""
echo "=========================================="
echo "  安装完成！"
echo "=========================================="
echo ""
echo "使用方法："
echo "  python voice_server.py"
echo ""

