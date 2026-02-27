

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

# st.set_page_config(
#     page_title="AD认知评估系统",
#     page_icon="🏥",
#     layout="centered",
#     initial_sidebar_state="expanded"
# )

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

# ChatGPT风格界面CSS - 已适配 Modern 主题
st.markdown("""
<style>
    /* 聊天容器优化 */
    .block-container {
        max-width: 800px;
        padding-bottom: 160px;
    }
    
    /* 对话消息容器 - 移除旧样式 */
    .msg {
        display: none;
    }
    
    /* 优化 Streamlit 聊天气泡 */
    .stChatMessage {
        background-color: transparent;
        border: none;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* AI消息 - 白色/灰色调 */
    .stChatMessage[data-testid="chat-message-assistant"],
    .msg:has(.avatar-ai) {
        background-color: #FFFFFF;
        border-radius: 20px 20px 20px 5px; /* 左下角直角 */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #F0F0F0;
        margin-right: 2rem;
        display: flex;
        gap: 1rem;
        padding: 1.5rem;
        align-items: flex-start;
    }

    /* 用户消息 - 蓝色调 */
    .stChatMessage[data-testid="chat-message-user"],
    .msg:has(.avatar-user) {
        background-color: #E3F2FD; /* 浅蓝背景 */
        border-radius: 20px 20px 5px 20px; /* 右下角直角 */
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.1);
        border: 1px solid #BBDEFB;
        margin-left: 2rem;
        display: flex;
        gap: 1.5rem;
        padding: 1.5rem 1rem;
        align-items: flex-start;
    }

    /* 头像样式 */
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .avatar-user {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: white;
    }
    
    .avatar-ai {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
    }
    
    /* 消息内容 */
    .msg-content {
        flex: 1;
        line-height: 1.8;
        font-size: 16px;
        color: #2d3748;
        padding-top: 2px;
    }
    
    /* 流式光标 */
    .streaming-cursor {
        display: inline-block;
        width: 8px;
        height: 18px;
        background: #00b894;
        margin-left: 4px;
        animation: blink 1s infinite;
        vertical-align: middle;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* 进度指示器优化 */
    .prog {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .prog-done {
        background: rgba(46, 125, 50, 0.1);
        color: #2e7d32;
        border: 1px solid rgba(46, 125, 50, 0.2);
    }
    
    .prog-now {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }
    
    .prog-wait {
        background: transparent;
        color: #a0aec0;
        border: 1px dashed #cbd5e0;
    }
    
    /* =================================================================
       底部输入区域固定定位
       注意：直接针对 Streamlit 的 Form 进行定位，而不是外层包裹 div
       ================================================================= */
    [data-testid="stForm"] {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(12px);
        padding: 1.5rem 2rem 2rem 2rem; /* 增加底部 padding 适应不同屏幕 */
        z-index: 9999;
        border-top: 1px solid #e2e8f0;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.05);
    }
    
    /* 调整 Form 内部布局 */
    [data-testid="stForm"] > div {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* 隐藏 Form 的边框（如果有） */
    [data-testid="stForm"] {
        border: none;
        border-top: 1px solid #e2e8f0;
    }
    
    /* 调整主内容区域的底部边距，防止内容被固定底栏遮挡 */
    .block-container {
        padding-bottom: 180px !important;
    }

    /* 输入框样式美化 */
    .stTextArea textarea {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
        resize: none !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
        background: white !important;
        min-height: 50px !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.15) !important;
    }
    
    /* 隐藏多余的label */
    .stTextArea label {
        display: none;
    }
    
    /* 调整侧边栏内输入框 */
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #f7fafc;
        border-color: #e2e8f0;
        color: #2d3748;
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

def get_agent():
    """创建并返回Agent实例"""
    agent = ADScreeningAgent()
    print(f"[INFO] Agent版本: {AGENT_VERSION}, 类名: {agent.__class__.__name__}")
    return agent

@st.cache_resource
def get_tts():
    """获取TTS实例（缓存）"""
    return VoiceTTS(voice="xiaoxiao")  # 使用晓晓音色

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
    
    # 将语音输入面板移到侧边栏
    if voice_enabled and st.session_state.session_started and st.session_state.waiting_for_answer:
        st.markdown("---")
        st.caption("👇 点击下方按钮录入患者语音")
        
        # 使用本地ASR组件
        voice_result = render_local_asr_interface(
            model_size="base",
            language="zh",
            key_prefix="local_asr_sidebar"
        )
        
        # 如果识别到结果，自动填入主界面的输入框
        if voice_result:
            st.session_state.voice_input_text = voice_result
            st.toast(f"已识别：{voice_result}", icon="✅")

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
                    
                    # 过滤掉可能的无效步骤
                    if 'agent_steps' in debug:
                        filtered_steps = []
                        for s in debug['agent_steps']:
                            raw_str = str(s.get('raw', ''))
                            result_str = str(s.get('result', ''))
                            if 'key: arrow_right' not in raw_str and 'key: arrow_right' not in result_str:
                                filtered_steps.append(s)
                        debug['agent_steps'] = filtered_steps
                    
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
                                            # 使用 st.code 或 st.json 避免格式混乱
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
                    if 'full_result' in debug and 'key: arrow_right' not in str(debug['full_result']):
                        with st.expander("📋 完整返回结果（调试用）"):
                            st.code(debug['full_result'])
                    
                    # 显示原始输出
                    if 'raw_output' in debug and 'key: arrow_right' not in str(debug['raw_output']):
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
    with st.form(key="f", clear_on_submit=True):
        # 使用语音识别结果作为默认值
        default_value = st.session_state.get('voice_input_text', '')
        placeholder_text = "输入患者回答..."
        
        # 使用列布局来放置输入框和发送按钮
        col_input, col_btn = st.columns([6, 1])
        
        with col_input:
            answer = st.text_area(
                "患者回答", 
                value=default_value,
                placeholder=placeholder_text, 
                height=52, 
                key="input", 
                label_visibility="collapsed"
            )
        
        with col_btn:
            # 调整按钮垂直位置以对齐
            st.markdown('<div style="height: 8px"></div>', unsafe_allow_html=True)
            submit = st.form_submit_button("发送", use_container_width=True)
        
        # 清空语音输入文本
        if default_value:
            st.session_state.voice_input_text = ""
        
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
