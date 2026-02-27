"""
Streamlit通用工具函数
用于减少代码重复
"""
import streamlit as st


def init_session_state():
    """初始化通用的session state变量"""
    defaults = {
        'session_started': False,
        'chat_history': [],
        'current_dimension_index': 0,
        'patient_profile': {'name': '', 'age': 70, 'education_years': 6, 'sex': '女'},
        'waiting_for_answer': False,
        'debug_info': [],
        'voice_enabled': True,
        'voice_input_text': "",
        'voice_recognition_result': "",
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_audio_player(audio_base64: str) -> str:
    """生成简洁的音频播放器HTML"""
    return f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'


def get_chatgpt_style_css() -> str:
    """返回ChatGPT风格的CSS样式"""
    return """
<style>
    /* 隐藏Streamlit默认元素 */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* 全局样式 */
    * {
        font-family: "Söhne", "Segoe UI", "Helvetica Neue", sans-serif;
    }
    
    .main {
        background: #ffffff;
        padding: 0;
    }
    
    .block-container {
        max-width: 768px;
        padding: 0 0 160px 0;
    }
    
    /* 对话消息容器 */
    .msg {
        display: flex;
        gap: 1.5rem;
        padding: 1.5rem 1rem;
        align-items: flex-start;
    }
    
    /* AI消息背景 */
    .msg:has(.avatar-ai) {
        background: #f7f7f8;
    }
    
    /* 头像样式 */
    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 2px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 600;
        color: white;
    }
    
    .avatar-user {
        background: #16a085;
    }
    
    .avatar-ai {
        background: #27ae60;
    }
    
    /* 消息内容 */
    .msg-content {
        flex: 1;
        line-height: 1.75;
        font-size: 16px;
        color: #353740;
        padding-top: 2px;
    }
    
    /* 输入区域 - 固定底部 */
    .input-wrapper {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #ffffff;
        padding: 1rem 0 2rem;
        z-index: 1000;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.05);
    }
    
    .input-container {
        max-width: 768px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* 输入框样式 */
    .stTextArea textarea {
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
        padding: 12px 52px 12px 16px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
        resize: none !important;
        box-shadow: 0 0 0 0 rgba(0,0,0,0) !important;
        transition: all 0.15s ease !important;
        background: #ffffff !important;
        min-height: 52px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #27ae60 !important;
        box-shadow: 0 0 0 1px #27ae60 !important;
        outline: none !important;
    }
    
    /* 发送按钮 */
    .stButton button {
        background: #27ae60 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        box-shadow: none !important;
        transition: background 0.2s ease !important;
    }
    
    .stButton button:hover {
        background: #229954 !important;
    }
    
    .stButton button:disabled {
        background: #d1d5db !important;
        cursor: not-allowed !important;
    }
</style>
"""
