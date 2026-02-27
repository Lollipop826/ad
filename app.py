"""
阿尔茨海默病初筛对话系统 - 真Claude风格
支持本地Whisper语音识别
"""

import streamlit as st
import json
import os
from datetime import datetime
import time
import threading
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置ffmpeg路径（用于本地Whisper）
os.environ['PATH'] = f"{os.path.expanduser('~/bin')}:{os.environ.get('PATH', '')}"

st.set_page_config(
    page_title="AD认知评估系统",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

import sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 使用Fast Agent（并行优化版本）
from src.agents.screening_agent_fast import ADScreeningAgentFast as ADScreeningAgent
AGENT_VERSION = "Fast (并行优化)"

from src.domain.dimensions import MMSE_DIMENSIONS
from src.tools.voice.tts_tool import VoiceTTS
from src.tools.voice.streamlit_local_asr import render_local_asr_interface
import base64

def get_agent():
    """创建并返回Agent实例"""
    agent = ADScreeningAgent()
    print(f"[INFO] Agent版本: {AGENT_VERSION}, 类名: {agent.__class__.__name__}")
    return agent

@st.cache_resource
def get_tts():
    """获取TTS实例（缓存）"""
    return VoiceTTS(voice="xiaoxiao")  # 使用晓晓音色

# ChatGPT风格界面CSS
st.markdown("""
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
    
    /* 流式光标 */
    .streaming-cursor {
        display: inline-block;
        width: 8px;
        height: 20px;
        background: #27ae60;
        margin-left: 2px;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
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
        position: absolute;
        right: 2.5rem;
        bottom: 2.7rem;
        background: #27ae60 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        box-shadow: none !important;
        z-index: 10;
        min-height: 36px !important;
        height: 36px !important;
        transition: background 0.2s ease !important;
    }
    
    .stButton button:hover {
        background: #229954 !important;
    }
    
    .stButton button:disabled {
        background: #d1d5db !important;
        cursor: not-allowed !important;
    }
    
    /* 欢迎屏幕 */
    .welcome {
        text-align: center;
        padding: 10rem 2rem;
        max-width: 640px;
        margin: 0 auto;
    }
    
    .welcome h1 {
        font-size: 32px;
        font-weight: 600;
        color: #202123;
        margin-bottom: 1rem;
    }
    
    .welcome p {
        font-size: 16px;
        color: #6e6e80;
        line-height: 1.6;
        margin-top: 1rem;
    }
    
    /* 侧边栏样式 - ChatGPT风格 */
    [data-testid="stSidebar"] {
        background: #202123;
        padding: 0.5rem;
    }
    
    [data-testid="stSidebar"] h3 {
        font-size: 12px;
        font-weight: 600;
        text-transform: none;
        letter-spacing: 0;
        color: #ececf1;
        margin: 1rem 0.75rem 0.5rem;
        padding: 0;
        border: none;
    }
    
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stSelectbox select {
        border: 1px solid #40414f !important;
        border-radius: 6px !important;
        font-size: 14px !important;
        padding: 10px 12px !important;
        background: #40414f !important;
        color: #ececf1 !important;
        transition: all 0.2s !important;
    }
    
    [data-testid="stSidebar"] .stTextInput input:focus,
    [data-testid="stSidebar"] .stNumberInput input:focus,
    [data-testid="stSidebar"] .stSelectbox select:focus {
        border-color: #27ae60 !important;
        background: #40414f !important;
    }
    
    [data-testid="stSidebar"] .stButton button {
        position: static !important;
        width: 100% !important;
        background: #27ae60 !important;
        margin: 0.5rem 0;
        border-radius: 6px !important;
        padding: 12px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background: #229954 !important;
    }
    
    /* 进度指示器 */
    .prog {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 12px;
        margin: 4px 0.75rem;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 400;
    }
    
    .prog-done {
        background: #2e7d32;
        color: #ffffff;
    }
    
    .prog-now {
        background: #27ae60;
        color: #ffffff;
    }
    
    .prog-wait {
        background: #40414f;
        color: #8e8ea0;
    }
    
    /* 进度条 */
    .stProgress > div > div {
        background: #27ae60;
        height: 4px;
        border-radius: 2px;
    }
    
    /* 滚动条美化 */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d1d5db;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9ca3af;
    }
    
    /* 侧边栏滚动条 */
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background: #40414f;
    }
    
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
        background: #565869;
    }
    
    /* 加载动画 */
    .stSpinner > div {
        border-top-color: #10a37f !important;
    }
    
    /* Expander样式 */
    .streamlit-expanderHeader {
        background: #f7f7f8;
        border-radius: 6px;
        padding: 10px 14px !important;
        font-weight: 500;
        color: #353740;
        font-size: 14px;
    }
    
    .streamlit-expanderHeader:hover {
        background: #ececf1;
    }
    
    /* 隐藏label */
    .stTextArea label {
        display: none;
    }
    
    /* 侧边栏标签颜色 */
    [data-testid="stSidebar"] label {
        color: #ececf1 !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State初始化
defaults = {
    'agent': None,
    'session_id': None,
    'chat_history': [],
    'current_dimension_index': 0,
    'patient_profile': {'name': '', 'age': 70, 'education_years': 6, 'sex': '女'},
    'session_started': False,
    'waiting_for_answer': False,
    'debug_info': [],
    'voice_enabled': True,
    'voice_input_text': "",
    'voice_recognition_result': "",
    'asr_server_started': False,
    'enable_streaming': True,  # 流式输出开关
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def get_current_dimension():
    if st.session_state.current_dimension_index < len(MMSE_DIMENSIONS):
        return MMSE_DIMENSIONS[st.session_state.current_dimension_index]
    return None

def generate_first_question():
    dimension = get_current_dimension()
    # 更自然、更有亲和力的开场问题
    questions = {
        "定向力": "您好！咱们聊聊天，顺便做个简单的评估。您知道今天是几号吗？",
        "即时记忆": "好的，我说三个词，您听完后重复一遍：苹果、桌子、外套。",
        "记忆力 - 即刻记忆": "好的，我说三个词，您听完后重复一遍：苹果、桌子、外套。",
        "注意力与计算力": "咱们算个数儿，100减7是多少？",
        "延迟记忆": "刚才那三个词，您还记得吗？",
        "记忆力 - 延迟回忆": "刚才那三个词，您还记得吗？",
        "语言能力": "您看这是什么？（指着笔）",
        "视空间能力": "您照着这个样子画一个五边形。"
    }
    return questions.get(dimension['name'], f"好，现在咱们聊聊{dimension['name']}。")


# 侧边栏
with st.sidebar:
    st.markdown("### 患者信息")
    
    disabled = st.session_state.session_started
    
    name = st.text_input("姓名", value=st.session_state.patient_profile['name'], disabled=disabled, placeholder="请输入")
    
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("年龄", 40, 100, st.session_state.patient_profile['age'], disabled=disabled)
    with c2:
        sex = st.selectbox("性别", ["女", "男"], index=0 if st.session_state.patient_profile['sex'] == '女' else 1, disabled=disabled)
    
    edu = st.number_input("教育年限", 0, 20, st.session_state.patient_profile['education_years'], disabled=disabled)
    
    if not disabled:
        st.session_state.patient_profile = {'name': name, 'age': age, 'sex': sex, 'education_years': edu}
    
    st.markdown("### 评估进度")
    
    for i, dim in enumerate(MMSE_DIMENSIONS):
        if i < st.session_state.current_dimension_index:
            st.markdown(f'<div class="prog prog-done">✓ {dim["name"]}</div>', unsafe_allow_html=True)
        elif i == st.session_state.current_dimension_index:
            st.markdown(f'<div class="prog prog-now">• {dim["name"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="prog prog-wait">○ {dim["name"]}</div>', unsafe_allow_html=True)
    
    st.progress(st.session_state.current_dimension_index / len(MMSE_DIMENSIONS))
    
    st.markdown("### 语音设置")
    voice_enabled = st.toggle("🎤 启用语音对话", value=st.session_state.voice_enabled, help="开启后可以语音输入，AI回复也会自动朗读")
    st.session_state.voice_enabled = voice_enabled
    
    if voice_enabled:
        st.caption("💡 点击「按住说话」按钮录音")
    
    # 流式输出开关
    st.markdown("### ⚡ 性能设置")
    enable_streaming = st.checkbox(
        "启用流式输出",
        value=st.session_state.enable_streaming,
        disabled=st.session_state.session_started,
        help="边生成边显示，首字延迟降低85%"
    )
    st.session_state.enable_streaming = enable_streaming
    
    if enable_streaming:
        st.caption("🌊 实时显示，体验更流畅")
    else:
        st.caption("🐢 等待完整响应后显示")
    
    st.markdown("### 控制")
    
    if not st.session_state.session_started:
        if st.button("开始评估"):
            if not name:
                st.error("请输入患者姓名")
            else:
                with st.spinner("初始化..."):
                    st.session_state.agent = get_agent()
                    st.session_state.session_id = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.session_started = True
                    st.session_state.current_dimension_index = 0
                    
                    q = generate_first_question()
                    st.session_state.chat_history.append({'role': 'ai', 'content': q})
                    st.session_state.debug_info.append({'raw_output': '首次问候', 'agent_steps': []})
                    
                    # 如果启用语音，生成首次问候的语音
                    if st.session_state.voice_enabled:
                        try:
                            tts = get_tts()
                            audio_path = tts.text_to_speech(q)
                            with open(audio_path, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                                st.session_state.last_audio = base64.b64encode(audio_bytes).decode()
                        except Exception as e:
                            print(f"[WARNING] 首次语音合成失败: {e}")
                    
                    st.session_state.waiting_for_answer = True
                st.rerun()
    else:
        st.caption(f"会话: {st.session_state.session_id}")
        if st.button("重新开始"):
            st.session_state.session_started = False
            st.session_state.chat_history = []
            st.session_state.debug_info = []
            st.session_state.current_dimension_index = 0
            st.session_state.waiting_for_answer = False
            st.rerun()

# 主界面
if not st.session_state.chat_history:
    st.markdown('''
    <div class="welcome">
        <h1>AD认知评估</h1>
        <p>基于AI的阿尔茨海默病初筛对话系统</p>
        <p style="margin-top: 2rem;">请在左侧填写患者信息并开始评估</p>
    </div>
    ''', unsafe_allow_html=True)
else:
    # 显示对话 - Claude风格
    ai_msg_count = 0
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg['role'] == 'ai':
            st.markdown(f'''
            <div class="msg">
                <div class="avatar avatar-ai">🩺</div>
                <div class="msg-content">{msg['content']}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # 如果是最新的AI消息且启用了语音，自动播放
            if st.session_state.voice_enabled and idx == len(st.session_state.chat_history) - 1 and 'last_audio' in st.session_state:
                audio_b64 = st.session_state.last_audio
                audio_html = f'''<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'''
                st.markdown(audio_html, unsafe_allow_html=True)
            
            # 显示调试信息
            if ai_msg_count < len(st.session_state.debug_info) and st.session_state.debug_info[ai_msg_count]:
                with st.expander("🔍 查看推理过程", expanded=False):
                    debug = st.session_state.debug_info[ai_msg_count]
                    
                    # 显示性能统计（Fast Agent专用）
                    if 'performance' in debug:
                        perf = debug['performance']
                        st.markdown("### ⚡ 性能统计")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("总耗时", f"{perf.get('total_time', 0):.2f}秒")
                        with col2:
                            st.metric("第1层", f"{perf.get('layer1_time', 0):.2f}秒")
                        with col3:
                            st.metric("第2层", f"{perf.get('layer2_time', 0):.2f}秒")
                        with col4:
                            st.metric("第3层", f"{perf.get('layer3_time', 0):.2f}秒")
                        
                        dim_type = "简单维度 (跳过检索)" if perf.get('is_simple_dimension') else "复杂维度 (完整流程)"
                        st.info(f"📊 {dim_type}")
                        st.markdown("---")
                    
                    # 显示Agent步骤（兼容新旧格式）
                    if 'agent_steps' in debug and debug['agent_steps']:
                        st.markdown("### 🤖 执行步骤")
                        
                        # 按层分组显示（Fast Agent）
                        layers = {}
                        for step in debug['agent_steps']:
                            layer = step.get('layer', 0)
                            if layer not in layers:
                                layers[layer] = []
                            layers[layer].append(step)
                        
                        if layers:
                            for layer_num in sorted(layers.keys()):
                                st.markdown(f"**第{layer_num}层:**")
                                for step in layers[layer_num]:
                                    tool_name = step.get('tool', '未知')
                                    st.markdown(f"- `{tool_name}` ({step.get('time', 0):.2f}秒)")
                                    
                                    if 'result' in step:
                                        with st.expander(f"查看 {tool_name} 结果"):
                                            st.json(step['result'])
                        else:
                            # 旧格式（Standard Agent）
                            for step in debug['agent_steps']:
                                st.markdown(f"**步骤 {step.get('step_num', '?')}:** `{step.get('action', '未知操作')}`")
                                
                                if 'action_input' in step:
                                    st.markdown("**输入参数:**")
                                    st.json(step['action_input'])
                                
                                if 'observation' in step:
                                    st.markdown("**执行结果:**")
                                    obs = step['observation']
                                    if len(obs) > 300:
                                        st.text_area("Result", obs[:300] + "\n...(已截断)", height=150, disabled=True, key=f"obs_long_{ai_msg_count}_{step.get('step_num', 0)}")
                                    else:
                                        st.text_area("Result", obs, height=100, disabled=True, key=f"obs_short_{ai_msg_count}_{step.get('step_num', 0)}")
                                
                                if 'observation_dict' in step:
                                    st.json(step['observation_dict'])
                                
                                if 'error' in step:
                                    st.error(step['error'])
                                
                                if 'raw' in step:
                                    with st.expander("查看原始数据"):
                                        st.code(step['raw'])
                                
                                st.markdown("---")
                    elif 'no_steps_reason' in debug:
                        st.warning(f"⚠️ {debug['no_steps_reason']}")
                    
                    # 显示完整结果（用于调试）
                    if 'full_result' in debug:
                        with st.expander("📋 完整返回结果（调试用）"):
                            st.code(debug['full_result'])
                    
                    # 显示原始输出
                    if 'raw_output' in debug:
                        st.markdown("### 📝 最终输出")
                        st.code(debug['raw_output'], language="text")
            
            ai_msg_count += 1
        else:
            st.markdown(f'''
            <div class="msg">
                <div class="avatar avatar-user">I</div>
                <div class="msg-content">{msg['content']}</div>
            </div>
            ''', unsafe_allow_html=True)

# 输入区 - 固定底部
if st.session_state.session_started and st.session_state.waiting_for_answer:
    st.markdown('<div class="input-wrapper"><div class="input-container">', unsafe_allow_html=True)
    
    # 本地语音识别（完全本地Whisper）
    if st.session_state.voice_enabled:
        # 使用本地ASR组件
        voice_result = render_local_asr_interface(
            model_size="base",
            language="zh",
            key_prefix="local_asr"
        )
        
        # 如果识别到结果，自动填入输入框
        if voice_result:
            st.session_state.voice_input_text = voice_result
    
    
    with st.form(key="f", clear_on_submit=True):
        # 使用语音识别结果作为默认值
        default_value = st.session_state.get('voice_input_text', '')
        placeholder_text = "输入患者回答..." + ("（或使用上方语音识别）" if st.session_state.voice_enabled else "")
        
        answer = st.text_area(
            "患者回答", 
            value=default_value,
            placeholder=placeholder_text, 
            height=52, 
            key="input", 
            label_visibility="collapsed"
        )
        
        # 清空语音输入文本
        if default_value:
            st.session_state.voice_input_text = ""
        submit = st.form_submit_button("发送")
        
        if submit and answer:
            st.session_state.chat_history.append({'role': 'user', 'content': answer})
            st.session_state.waiting_for_answer = False
            
            # ⚠️ 先判断是否需要切换维度（在调用Agent之前）
            ai_msgs = [m for m in st.session_state.chat_history if m['role'] == 'ai']
            should_switch_dimension = len(ai_msgs) >= 3 and st.session_state.current_dimension_index < len(MMSE_DIMENSIONS) - 1
            dimension_switched = False
            
            if should_switch_dimension:
                st.session_state.current_dimension_index += 1
                dimension_switched = True
            
            # 获取当前维度（如果刚切换，这就是新维度）
            dim = get_current_dimension()
            hist = [{"role": "assistant" if m['role'] == 'ai' else "user", "content": m['content']} for m in st.session_state.chat_history]
            
            # 🌊 根据开关选择流式或普通模式
            if st.session_state.enable_streaming:
                # 流式输出模式
                st.markdown("---")
                response_placeholder = st.empty()
                full_response = ""
                
                try:
                    for chunk in st.session_state.agent.process_turn_streaming(
                        user_input=answer,
                        dimension=dim,
                        session_id=st.session_state.session_id,
                        patient_profile=st.session_state.patient_profile,
                        chat_history=hist
                    ):
                        if chunk['type'] == 'token':
                            full_response = chunk['full_text']
                            # 实时显示（带流式光标）
                            response_placeholder.markdown(f'''
                            <div class="msg">
                                <div class="avatar avatar-ai">AI</div>
                                <div class="msg-content">{full_response}<span class="streaming-cursor"></span></div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        elif chunk['type'] == 'done':
                            # 完成，移除光标
                            full_response = chunk['content']
                            response_placeholder.markdown(f'''
                            <div class="msg">
                                <div class="avatar avatar-ai">AI</div>
                                <div class="msg-content">{full_response}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            # 显示性能信息
                            metadata = chunk.get('metadata', {})
                            st.caption(f"⚡ 总耗时: {metadata.get('total_time', 0):.2f}秒 | 流式耗时: {metadata.get('stream_time', 0):.2f}秒")
                    
                    next_q = full_response
                    
                    # 如果切换了维度，添加提示
                    if dimension_switched:
                        next_q = f"很好。现在我们评估一下{dim['name']}。\n\n{next_q}"
                    
                    # 保存到历史
                    st.session_state.chat_history.append({'role': 'ai', 'content': next_q})
                    st.session_state.waiting_for_answer = True
                    
                    # 语音合成
                    if st.session_state.voice_enabled:
                        try:
                            tts = get_tts()
                            audio_path = tts.text_to_speech(next_q)
                            with open(audio_path, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                                st.session_state.last_audio = base64.b64encode(audio_bytes).decode()
                        except Exception as e:
                            print(f"[WARNING] 语音合成失败: {e}")
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"处理失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.session_state.waiting_for_answer = True
                    
            else:
                # 普通模式
                with st.spinner("分析中..."):
                    try:
                        # 调用Agent（已经是新维度了）
                        result = st.session_state.agent.process_turn(
                            user_input=answer,
                            dimension=dim,
                            session_id=st.session_state.session_id,
                            patient_profile=st.session_state.patient_profile,
                            chat_history=hist
                        )
                        
                        next_q = result.get('output', '请继续。')
                        
                        # 如果切换了维度，在问题前加提示
                        if dimension_switched:
                            next_q = f"很好。现在我们评估一下{dim['name']}。\n\n{next_q}"
                        
                        # 收集调试信息（兼容Fast和Standard Agent）
                        debug_info = {
                            'raw_output': result.get('output', ''),
                            'full_result': str(result)[:1000] if 'error' not in result else str(result),
                            'agent_steps': []
                        }
                        
                        # 添加性能信息（Fast Agent专用）
                        if 'performance' in result:
                            debug_info['performance'] = result['performance']
                        
                        # 提取intermediate_steps - 兼容新旧格式
                        if 'intermediate_steps' in result and result['intermediate_steps']:
                            steps = result['intermediate_steps']
                            
                            # 判断是Fast Agent格式（dict）还是Standard Agent格式（tuple）
                            if steps and isinstance(steps[0], dict):
                                # Fast Agent格式：直接使用
                                debug_info['agent_steps'] = steps
                            else:
                                # Standard Agent格式：解析tuple
                                for idx, step in enumerate(steps):
                                    try:
                                        step_info = {'step_num': idx + 1}
                                        
                                        if isinstance(step, (list, tuple)) and len(step) >= 2:
                                            action = step[0]
                                            observation = step[1]
                                            
                                            # 提取工具名称
                                            if hasattr(action, 'tool'):
                                                step_info['action'] = action.tool
                                            elif hasattr(action, 'name'):
                                                step_info['action'] = action.name
                                            else:
                                                step_info['action'] = str(type(action).__name__)
                                            
                                            # 提取工具输入
                                            if hasattr(action, 'tool_input'):
                                                step_info['action_input'] = action.tool_input
                                            elif hasattr(action, 'input'):
                                                step_info['action_input'] = action.input
                                            else:
                                                step_info['action_input'] = {'raw': str(action)[:200]}
                                            
                                            # 提取观察结果
                                            step_info['observation'] = str(observation)[:800]
                                            
                                            # 尝试解析特殊的观察结果
                                            if isinstance(observation, dict):
                                                step_info['observation_dict'] = observation
                                        else:
                                            step_info['raw'] = str(step)[:500]
                                        
                                        debug_info['agent_steps'].append(step_info)
                                    except Exception as e:
                                        debug_info['agent_steps'].append({
                                            'error': f'提取步骤{idx+1}失败: {str(e)}',
                                            'raw': str(step)[:300]
                                        })
                        else:
                            debug_info['no_steps_reason'] = 'intermediate_steps为空或不存在'
                        
                        st.session_state.debug_info.append(debug_info)
                        
                        st.session_state.chat_history.append({'role': 'ai', 'content': next_q})
                        
                        # 如果启用语音，生成并播放AI回复
                        if st.session_state.voice_enabled:
                            try:
                                tts = get_tts()
                                audio_path = tts.text_to_speech(next_q)
                                
                                # 将音频文件转为base64以便在网页中播放
                                with open(audio_path, "rb") as audio_file:
                                    audio_bytes = audio_file.read()
                                    st.session_state.last_audio = base64.b64encode(audio_bytes).decode()
                            except Exception as e:
                                print(f"[WARNING] 语音合成失败: {e}")
                        
                        st.session_state.waiting_for_answer = True
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"错误: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.session_state.waiting_for_answer = True
    
    st.markdown('</div></div>', unsafe_allow_html=True)
