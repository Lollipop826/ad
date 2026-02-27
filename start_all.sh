#!/bin/bash

# 加载存储配置 - 使用外部存储避免系统盘满
if [ -f ".env.storage" ]; then
    source .env.storage
    echo "✅ 已加载外部存储配置"
fi

echo "======================================"
echo "🚀 AD筛查语音系统 - 启动脚本"
echo "======================================"

# 杀死所有相关进程
echo ""
echo "🔪 正在清理旧进程..."
pkill -9 -f voice_server.py 2>/dev/null
pkill -9 -f streamlit 2>/dev/null
pkill -9 -f main_app.py 2>/dev/null
sleep 2
echo "✅ 旧进程已清理"

# 进入项目目录
cd /root/autodl-tmp/langchain-mcp-adapters-main

# 启动 WebSocket 后端服务器
echo ""
echo "🔧 启动 WebSocket 后端服务（端口 8502）..."

# 同时输出到控制台和日志文件（使用外部存储）
mkdir -p /root/autodl-tmp/tmp
python -u voice_server.py 2>&1 | tee /root/autodl-tmp/tmp/voice_server_startup.log | sed 's/^/[WebSocket] /' &
VOICE_PID=$!
echo "✅ WebSocket 服务已启动 (PID: $VOICE_PID)"
echo "📝 日志文件: /root/autodl-tmp/tmp/voice_server_startup.log"

# 等待后端初始化
echo "⏳ 等待后端初始化（约30秒）..."
echo "🔍 可以运行查看启动日志: tail -f /root/autodl-tmp/tmp/voice_server_startup.log"
sleep 30

# 启动 Streamlit 前端
echo ""
echo "🌐 启动 Streamlit 前端（端口 8501）..."
streamlit run main_app.py --server.port 8501 2>&1 | sed 's/^/[Streamlit] /' &
STREAMLIT_PID=$!
echo "✅ Streamlit 服务已启动 (PID: $STREAMLIT_PID)"

# 显示访问地址
echo ""
echo "======================================"
echo "✅ 所有服务已启动！"
echo "======================================"
echo ""
echo "📍 访问地址："
echo "   🎨 完整界面（Streamlit）: http://123.127.15.138:8501"
echo "   🎤 语音界面（WebSocket）: http://123.127.15.138:8502"
echo ""
echo "💡 使用说明："
echo "   - Streamlit 界面：点击「语音对话」→「开始通话」"
echo "   - WebSocket 界面：直接点击「开始对话」"
echo ""
echo "⏹️  停止服务："
echo "   按 Ctrl+C 停止，或运行: pkill -9 -f 'voice_server|streamlit'"
echo ""
echo "======================================"
echo "📊 实时日志输出（按 Ctrl+C 退出）："
echo "======================================"
echo ""

# 保持脚本运行，实时显示日志
wait
