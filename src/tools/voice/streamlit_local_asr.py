"""
Streamlit本地ASR组件 - 简单直接版本
使用Whisper进行本地语音识别
"""

import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np
import os

@st.cache_resource
def load_whisper_model(model_size="base"):
    """加载并缓存Whisper模型"""
    return whisper.load_model(model_size)

def render_local_asr_interface(model_size="base", language="zh", key_prefix="local_asr"):
    """
    渲染本地ASR界面
    
    Args:
        model_size: Whisper模型大小 (tiny, base, small, medium, large)
        language: 识别语言 (zh, en, auto)
        key_prefix: 组件key前缀
        
    Returns:
        识别结果文本，如果没有则返回None
    """
    
    # 初始化session state
    if f"{key_prefix}_result" not in st.session_state:
        st.session_state[f"{key_prefix}_result"] = ""
    
    # 语音输入区域
    st.markdown("### 🎤 语音输入")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        duration = st.slider(
            "录音时长（秒）", 
            min_value=1, 
            max_value=10, 
            value=3,
            key=f"{key_prefix}_duration"
        )
    
    with col2:
        if st.button("🎤 开始录音", key=f"{key_prefix}_record", type="primary"):
            try:
                # 录音
                with st.spinner(f"正在录音 {duration} 秒..."):
                    sample_rate = 16000
                    audio = sd.rec(
                        int(duration * sample_rate), 
                        samplerate=sample_rate, 
                        channels=1,
                        dtype=np.float32
                    )
                    sd.wait()
                
                st.success("✅ 录音完成！")
                
                # 识别
                with st.spinner("正在识别..."):
                    # 保存临时文件
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                        sf.write(f.name, audio, sample_rate)
                        temp_path = f.name
                    
                    try:
                        # 加载模型
                        model = load_whisper_model(model_size)
                        
                        # 识别
                        result = model.transcribe(
                            temp_path,
                            language=language,
                            fp16=False,
                            verbose=False
                        )
                        
                        text = result['text'].strip()
                        
                        if text:
                            st.session_state[f"{key_prefix}_result"] = text
                            st.success(f"✅ 识别成功: {text}")
                        else:
                            st.warning("⚠️ 未识别到内容，请重试")
                            
                    finally:
                        # 清理临时文件
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
            except Exception as e:
                st.error(f"❌ 错误: {str(e)}")
    
    with col3:
        if st.button("🗑️ 清除", key=f"{key_prefix}_clear"):
            st.session_state[f"{key_prefix}_result"] = ""
            st.rerun()
    
    # 显示识别结果
    if st.session_state[f"{key_prefix}_result"]:
        st.text_area(
            "识别结果",
            value=st.session_state[f"{key_prefix}_result"],
            height=80,
            key=f"{key_prefix}_display",
            disabled=True
        )
        
        # 返回结果并清空（用于表单提交）
        result = st.session_state[f"{key_prefix}_result"]
        return result
    
    return None

# 简化版本 - 只返回录音按钮和结果
def simple_voice_input(model_size="base", language="zh"):
    """
    简化的语音输入组件
    
    Returns:
        识别结果文本
    """
    if st.button("🎤 语音输入 (3秒)", type="primary"):
        try:
            with st.spinner("正在录音..."):
                sample_rate = 16000
                duration = 3
                audio = sd.rec(
                    int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    dtype=np.float32
                )
                sd.wait()
            
            with st.spinner("正在识别..."):
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    sf.write(f.name, audio, sample_rate)
                    temp_path = f.name
                
                try:
                    model = load_whisper_model(model_size)
                    result = model.transcribe(temp_path, language=language, fp16=False, verbose=False)
                    text = result['text'].strip()
                    
                    if text:
                        st.success(f"✅ {text}")
                        return text
                    else:
                        st.warning("未识别到内容")
                        return ""
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
        except Exception as e:
            st.error(f"错误: {e}")
            return ""
    
    return None
