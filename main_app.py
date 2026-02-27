"""
阿尔茨海默病初筛系统 - 统一入口
集成了患者评估和知识库浏览两个功能
"""

import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 导入现代样式
from styles.modern import get_modern_css, get_sidebar_content

st.set_page_config(
    page_title="AD初筛系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用现代 CSS
st.markdown(get_modern_css(), unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    # 获取当前页面对应的侧边栏内容标识
    current_page_key = "main"
    if 'current_page' in st.session_state:
        if st.session_state.current_page == "评估系统":
            current_page_key = "assessment"
        elif st.session_state.current_page == "知识库":
            current_page_key = "knowledge"
        elif st.session_state.current_page == "知识库构建":
            current_page_key = "builder"
            
    st.markdown(get_sidebar_content(current_page_key), unsafe_allow_html=True)

# 导航栏容器样式
st.markdown("""
<style>
.nav-card {
    background-color: white;
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
    text-align: center;
    cursor: pointer;
}
.nav-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    border-color: #4A90E2;
}
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "主页"

# 顶部导航区域 (始终显示)
with st.container():
    cols = st.columns([1, 1, 1, 1, 1])
    
    # 定义导航项
    nav_items = [
        ("🏠 主页", "主页", "primary"),
        ("🩺 开始评估", "评估系统", "primary"),
        ("📚 知识库", "知识库", "secondary"),
        ("🔧 构建工具", "知识库构建", "secondary"),
        ("📞 语音对话", "语音通话", "secondary")
    ]
    
    for col, (label, page_name, kind) in zip(cols, nav_items):
        is_active = st.session_state.current_page == page_name
        btn_type = "primary" if is_active else "secondary"
        if col.button(label, key=f"nav_{page_name}", use_container_width=True, type=btn_type):
            st.session_state.current_page = page_name
            st.rerun()

st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)

# 主内容区域
if st.session_state.current_page == "主页":
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; background: white; border-radius: 24px; border: 1px solid #edf2f7; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02); margin-bottom: 2rem;'>
        <h1 style='font-size: 48px; background: linear-gradient(135deg, #2d3748 0%, #4A90E2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem;'>
            阿尔茨海默病智能认知初筛
        </h1>
        <p style='font-size: 20px; color: #718096; max-width: 600px; margin: 0 auto 2rem;'>
            结合大语言模型与专业医学知识库，为您提供便捷、专业的认知功能初步评估服务。
        </p>
        <div style='display: flex; gap: 1rem; justify-content: center;'>
            <div style='background: #ebf8ff; color: #2b6cb0; padding: 0.5rem 1rem; border-radius: 20px; font-size: 14px; font-weight: 600;'>🤖 AI 智能对话</div>
            <div style='background: #f0fff4; color: #2f855a; padding: 0.5rem 1rem; border-radius: 20px; font-size: 14px; font-weight: 600;'>📚 RAG 知识增强</div>
            <div style='background: #fffaf0; color: #c05621; padding: 0.5rem 1rem; border-radius: 20px; font-size: 14px; font-weight: 600;'>🎙️ 多模态交互</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 功能卡片展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.02);'>
            <div style='width: 48px; height: 48px; background: #ebf8ff; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem; font-size: 24px;'>🩺</div>
            <h3 style='margin-bottom: 1rem;'>患者评估系统</h3>
            <p style='color: #718096; line-height: 1.6; margin-bottom: 1.5rem;'>
                基于 MMSE 标准的智能化认知功能评估。支持语音对话，实时分析用户回答，自动生成评估报告。
            </p>
            <ul style='color: #4a5568; margin-left: 1rem; margin-bottom: 0;'>
                <li>多维度认知能力测试</li>
                <li>实时情绪状态监测</li>
                <li>自动生成评分记录</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.02);'>
            <div style='width: 48px; height: 48px; background: #f0fff4; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem; font-size: 24px;'>📚</div>
            <h3 style='margin-bottom: 1rem;'>医学知识库</h3>
            <p style='color: #718096; line-height: 1.6; margin-bottom: 1.5rem;'>
                集成专业医学文献和诊疗指南，为 AI 评估提供可靠的知识支撑。支持全文检索和知识问答。
            </p>
            <ul style='color: #4a5568; margin-left: 1rem; margin-bottom: 0;'>
                <li>专业文献检索</li>
                <li>支持自定义文档上传</li>
                <li>RAG 增强回答准确性</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 16px; border: 1px solid #e2e8f0; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.02);'>
            <div style='width: 48px; height: 48px; background: #fffaf0; border-radius: 12px; display: flex; align-items: center; justify-content: center; margin-bottom: 1rem; font-size: 24px;'>📞</div>
            <h3 style='margin-bottom: 1rem;'>实时语音对话</h3>
            <p style='color: #718096; line-height: 1.6; margin-bottom: 1.5rem;'>
                完全语音化的交互体验，模拟真实的医生问诊场景。支持实时语音识别(ASR)和语音合成(TTS)。
            </p>
            <ul style='color: #4a5568; margin-left: 1rem; margin-bottom: 0;'>
                <li>本地 Whisper 语音识别</li>
                <li>Edge TTS 高质量语音</li>
                <li>自动语音活动检测(VAD)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.current_page == "评估系统":
    # 动态导入并执行评估页面
    with open("pages/assessment.py", encoding='utf-8') as f:
        code = f.read()
        exec(code, globals())
elif st.session_state.current_page == "知识库":
    # 动态导入并执行知识库页面
    with open("pages/knowledge_base.py", encoding='utf-8') as f:
        code = f.read()
        exec(code, globals())
elif st.session_state.current_page == "知识库构建":
    # 动态导入并执行知识库构建页面
    with open("pages/knowledge_builder.py", encoding='utf-8') as f:
        code = f.read()
        exec(code, globals())
elif st.session_state.current_page == "语音通话":
    # 动态导入并执行语音通话页面
    with open("pages/voice_call.py", encoding='utf-8') as f:
        code = f.read()
        exec(code, globals())

