"""
最简单的本地ASR应用 - 直接使用Whisper，无需服务器
"""

import streamlit as st
import os
import sys

# 设置ffmpeg路径
os.environ['PATH'] = f"{os.path.expanduser('~/bin')}:{os.environ.get('PATH', '')}"

sys.path.append('src')

from src.agents.screening_agent import ADScreeningAgent
from src.domain.dimensions import MMSE_DIMENSIONS
from datetime import datetime
import whisper
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np

st.set_page_config(
    page_title="AD认知评估 - 本地语音版",
    page_icon="🎤",
    layout="centered"
)

# 初始化
defaults = {
    'agent': None,
    'session_id': None,
    'chat_history': [],
    'current_dimension_index': 0,
    'patient_profile': {'name': '', 'age': 70, 'education_years': 6, 'sex': '女'},
    'session_started': False,
    'waiting_for_answer': False,
    'whisper_model': None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

@st.cache_resource
def get_whisper_model():
    """加载Whisper模型"""
    return whisper.load_model("base")

# 标题
st.title("🎤 AD认知评估 - 本地语音版")
st.markdown("完全本地运行，保护隐私")

# 侧边栏
with st.sidebar:
    st.header("患者信息")
    name = st.text_input("姓名", value=st.session_state.patient_profile['name'])
    age = st.number_input("年龄", 40, 100, st.session_state.patient_profile['age'])
    sex = st.selectbox("性别", ["女", "男"])
    edu = st.number_input("教育年限", 0, 20, st.session_state.patient_profile['education_years'])
    
    st.session_state.patient_profile = {'name': name, 'age': age, 'sex': sex, 'education_years': edu}
    
    st.markdown("---")
    st.header("评估进度")
    for i, dim in enumerate(MMSE_DIMENSIONS):
        if i < st.session_state.current_dimension_index:
            st.success(f"✓ {dim['name']}")
        elif i == st.session_state.current_dimension_index:
            st.info(f"→ {dim['name']}")
        else:
            st.text(f"○ {dim['name']}")
    
    st.progress(st.session_state.current_dimension_index / len(MMSE_DIMENSIONS))

# 主界面
if not st.session_state.session_started:
    st.info("请填写患者信息后开始评估")
    if st.button("开始评估", type="primary"):
        if not name:
            st.error("请输入患者姓名")
        else:
            st.session_state.agent = ADScreeningAgent()
            st.session_state.session_id = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.session_started = True
            st.session_state.waiting_for_answer = True
            st.session_state.chat_history.append({
                'role': 'ai',
                'content': "您好！我们开始进行认知评估。您知道今天是几号吗？"
            })
            st.rerun()
else:
    # 显示对话历史
    for msg in st.session_state.chat_history:
        if msg['role'] == 'ai':
            with st.chat_message("assistant"):
                st.write(msg['content'])
        else:
            with st.chat_message("user"):
                st.write(msg['content'])
    
    if st.session_state.waiting_for_answer:
        # 语音输入
        st.markdown("### 🎤 语音输入")
        
        col1, col2 = st.columns(2)
        with col1:
            duration = st.slider("录音时长（秒）", 1, 10, 3)
        with col2:
            if st.button("🎤 开始录音", type="primary"):
                try:
                    with st.spinner(f"正在录音{duration}秒..."):
                        # 录音
                        sample_rate = 16000
                        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
                        sd.wait()
                        st.success("录音完成！")
                    
                    with st.spinner("正在识别..."):
                        # 保存临时文件
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                            sf.write(f.name, audio, sample_rate)
                            temp_path = f.name
                        
                        # 加载模型并识别
                        if st.session_state.whisper_model is None:
                            st.session_state.whisper_model = get_whisper_model()
                        
                        result = st.session_state.whisper_model.transcribe(
                            temp_path,
                            language='zh',
                            fp16=False
                        )
                        
                        recognized_text = result['text'].strip()
                        os.unlink(temp_path)
                        
                        if recognized_text:
                            st.session_state.chat_history.append({
                                'role': 'user',
                                'content': recognized_text
                            })
                            
                            # 获取AI回复
                            dim = MMSE_DIMENSIONS[st.session_state.current_dimension_index]
                            hist = [{"role": "assistant" if m['role'] == 'ai' else "user", "content": m['content']} 
                                   for m in st.session_state.chat_history]
                            
                            ai_result = st.session_state.agent.process_turn(
                                user_input=recognized_text,
                                dimension=dim,
                                session_id=st.session_state.session_id,
                                patient_profile=st.session_state.patient_profile,
                                chat_history=hist
                            )
                            
                            next_q = ai_result.get('output', '请继续。')
                            st.session_state.chat_history.append({
                                'role': 'ai',
                                'content': next_q
                            })
                            
                            # 检查是否需要切换维度
                            ai_msgs = [m for m in st.session_state.chat_history if m['role'] == 'ai']
                            if len(ai_msgs) >= 3 and st.session_state.current_dimension_index < len(MMSE_DIMENSIONS) - 1:
                                st.session_state.current_dimension_index += 1
                            
                            st.rerun()
                        else:
                            st.warning("未识别到内容，请重试")
                            
                except Exception as e:
                    st.error(f"错误: {str(e)}")
        
        # 文字输入
        st.markdown("### ⌨️ 或者文字输入")
        with st.form("text_input"):
            text_answer = st.text_area("输入回答", height=100)
            if st.form_submit_button("发送"):
                if text_answer:
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': text_answer
                    })
                    
                    # 获取AI回复
                    dim = MMSE_DIMENSIONS[st.session_state.current_dimension_index]
                    hist = [{"role": "assistant" if m['role'] == 'ai' else "user", "content": m['content']} 
                           for m in st.session_state.chat_history]
                    
                    ai_result = st.session_state.agent.process_turn(
                        user_input=text_answer,
                        dimension=dim,
                        session_id=st.session_state.session_id,
                        patient_profile=st.session_state.patient_profile,
                        chat_history=hist
                    )
                    
                    next_q = ai_result.get('output', '请继续。')
                    st.session_state.chat_history.append({
                        'role': 'ai',
                        'content': next_q
                    })
                    
                    # 检查是否需要切换维度
                    ai_msgs = [m for m in st.session_state.chat_history if m['role'] == 'ai']
                    if len(ai_msgs) >= 3 and st.session_state.current_dimension_index < len(MMSE_DIMENSIONS) - 1:
                        st.session_state.current_dimension_index += 1
                    
                    st.rerun()
    else:
        st.success("评估完成！")
        if st.button("重新开始"):
            st.session_state.session_started = False
            st.session_state.chat_history = []
            st.session_state.current_dimension_index = 0
            st.rerun()

