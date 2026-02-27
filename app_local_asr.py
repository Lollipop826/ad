"""
本地ASR版本的AD筛查应用
使用Whisper进行本地语音识别，完全离线运行
"""

import streamlit as st
import os
import base64
import requests
import threading
import time
from pathlib import Path

# 设置ffmpeg路径（确保Whisper能找到）
os.environ['PATH'] = f"{os.path.expanduser('~/bin')}:{os.environ.get('PATH', '')}"

# 设置页面配置
st.set_page_config(
    page_title="AD认知评估 - 本地ASR版",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入必要的模块
import sys
sys.path.append('src')

# 动态导入Agent
USE_FAST_AGENT = os.getenv("USE_FAST_AGENT", "0")
if USE_FAST_AGENT == "1":
    from src.agents.screening_agent_fast import ADScreeningAgentFast as ADScreeningAgent
    AGENT_VERSION = "Fast"
else:
    from src.agents.screening_agent import ADScreeningAgent
    AGENT_VERSION = "Standard"

from src.domain.dimensions import MMSE_DIMENSIONS
from src.tools.voice.tts_tool import VoiceTTS
from src.tools.voice.local_asr_api import start_asr_server

# 缓存函数
@st.cache_resource
def get_agent():
    agent = ADScreeningAgent()
    print(f"[INFO] 使用Agent版本: {AGENT_VERSION}")
    return agent

@st.cache_resource
def get_tts():
    """获取TTS实例（缓存）"""
    return VoiceTTS(voice="xiaoxiao")  # 使用晓晓音色

# 初始化session state
defaults = {
    'session_started': False,
    'chat_history': [],
    'current_dimension_index': 0,
    'session_id': f"session_{int(time.time())}",
    'patient_profile': {'name': '', 'age': 65, 'sex': '女', 'education_years': 6},
    'waiting_for_answer': False,
    'debug_info': [],
    'voice_enabled': True,
    'voice_input_text': "",
    'voice_recognition_result': "",
    'asr_server_started': False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# 启动ASR服务器
if not st.session_state.asr_server_started:
    try:
        # 首先检查服务器是否已在运行
        try:
            response = requests.get("http://127.0.0.1:5001/health", timeout=1)
            if response.status_code == 200:
                st.session_state.asr_server_started = True
                print("[ASR] ✅ 本地ASR服务器已运行")
            else:
                raise Exception("需要启动服务器")
        except:
            # 在后台启动ASR服务器
            def start_server():
                # 确保ffmpeg在PATH中
                os.environ['PATH'] = f"{os.path.expanduser('~/bin')}:{os.environ.get('PATH', '')}"
                start_asr_server(model_size="base", port=5001)
            
            server_thread = threading.Thread(target=start_server, daemon=True)
            server_thread.start()
            
            # 等待服务器启动
            time.sleep(3)
            
            # 测试连接
            try:
                response = requests.get("http://127.0.0.1:5001/health", timeout=2)
                if response.status_code == 200:
                    st.session_state.asr_server_started = True
                    print("[ASR] ✅ 本地ASR服务器已启动")
                else:
                    raise Exception("服务器响应异常")
            except Exception as e:
                print(f"[ASR] ❌ ASR服务器连接失败: {e}")
                st.session_state.voice_enabled = False
    except Exception as e:
        print(f"[ASR] ❌ ASR服务器启动失败: {e}")
        st.session_state.voice_enabled = False

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
    
    st.markdown("### 语音设置")
    voice_enabled = st.toggle("🎤 启用本地语音识别", value=st.session_state.voice_enabled, help="使用本地Whisper模型进行语音识别，完全离线运行")
    st.session_state.voice_enabled = voice_enabled
    
    if st.session_state.asr_server_started:
        st.success("✅ 本地ASR服务器运行中")
    else:
        st.error("❌ ASR服务器未启动")
    
    st.markdown("### 系统信息")
    st.caption(f"Agent版本: {AGENT_VERSION}")
    st.caption("语音识别: 本地Whisper")
    st.caption("语音合成: Edge TTS")

# 主界面
if not st.session_state.chat_history:
    st.markdown('''
    <div style="text-align: center; padding: 2rem;">
        <h1>🩺 AD认知评估 - 本地语音版</h1>
        <p style="font-size: 1.2rem; color: #666;">基于本地AI的阿尔茨海默病初筛对话系统</p>
        <p style="margin-top: 2rem;">✅ 完全离线运行，保护隐私</p>
        <p>✅ 本地Whisper语音识别</p>
        <p>✅ Edge TTS语音合成</p>
        <p style="margin-top: 2rem;">请在左侧填写患者信息并开始评估</p>
    </div>
    ''', unsafe_allow_html=True)
else:
    # 显示对话
    ai_msg_count = 0
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg['role'] == 'ai':
            st.markdown(f'''
            <div style="display: flex; margin-bottom: 1rem;">
                <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem; flex-shrink: 0;">
                    <span style="color: white; font-size: 1.2rem;">🩺</span>
                </div>
                <div style="flex: 1; background: #f8f9fa; padding: 1rem; border-radius: 12px; border-left: 4px solid #667eea;">
                    {msg['content']}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # 如果是最新的AI消息且启用了语音，自动播放
            if st.session_state.voice_enabled and idx == len(st.session_state.chat_history) - 1 and 'last_audio' in st.session_state:
                audio_b64 = st.session_state.last_audio
                # 音频播放器
                audio_html = f'''<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>'''
                st.markdown(audio_html, unsafe_allow_html=True)
            
            ai_msg_count += 1
        else:
            st.markdown(f'''
            <div style="display: flex; margin-bottom: 1rem; flex-direction: row-reverse;">
                <div style="width: 40px; height: 40px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-left: 1rem; flex-shrink: 0;">
                    <span style="color: white; font-size: 1.2rem;">👤</span>
                </div>
                <div style="flex: 1; background: #e3f2fd; padding: 1rem; border-radius: 12px; border-right: 4px solid #f093fb;">
                    {msg['content']}
                </div>
            </div>
            ''', unsafe_allow_html=True)

# 输入区
if st.session_state.session_started and st.session_state.waiting_for_answer:
    st.markdown('<div style="position: fixed; bottom: 0; left: 0; right: 0; background: white; padding: 1rem; border-top: 1px solid #e0e0e0; z-index: 1000;">', unsafe_allow_html=True)
    
    # 本地语音识别
    if st.session_state.voice_enabled:
        voice_html = """
        <div style="padding: 10px 0;">
            <div style="text-align: center;">
                <button id="voiceBtn" style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    padding: 15px 30px;
                    border-radius: 25px;
                    font-size: 16px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.3s;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                    margin: 0 10px;
                " onmousedown="startRecording()" onmouseup="stopRecording()" onmouseleave="stopRecording()" ontouchstart="startRecording()" ontouchend="stopRecording()" ontouchcancel="stopRecording()">
                    🎤 按住说话
                </button>
                <button id="stopBtn" onclick="stopRecording()" style="
                    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
                    color: white;
                    border: none;
                    padding: 15px 30px;
                    border-radius: 25px;
                    font-size: 16px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.3s;
                    box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
                    margin: 0 10px;
                    display: none;
                ">
                    ⏹️ 强制停止
                </button>
            </div>
            <div id="voiceStatus" style="text-align: center; margin-top: 10px; font-size: 14px; color: #666;">
                🎙️ 点击按钮开始录音
            </div>
        </div>
        
        <div id="streamingText" style="
            margin: 15px 0;
            padding: 15px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            min-height: 60px;
            display: none;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 12px; color: #666; margin-bottom: 8px;">
                💬 本地AI识别：
            </div>
            <div id="interimResult" style="
                font-size: 16px;
                color: #999;
                font-style: italic;
                margin-bottom: 5px;
                min-height: 20px;
            "></div>
            <div id="finalResult" style="
                font-size: 18px;
                color: #2c3e50;
                font-weight: 500;
                min-height: 20px;
            "></div>
        </div>
        <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let finalTranscript = '';
        let recordingTimeout;
        
        // 初始化音频录制
        async function initAudioRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = function(event) {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = function() {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    audioChunks = [];
                    transcribeAudio(audioBlob);
                };
                
                console.log('本地ASR音频录制初始化成功');
            } catch (error) {
                console.error('音频录制初始化失败:', error);
                document.getElementById('voiceStatus').innerHTML = '❌ 无法访问麦克风';
                document.getElementById('voiceBtn').disabled = true;
            }
        }
        
        function startRecording() {
            if (!mediaRecorder) {
                initAudioRecording();
                return;
            }
            
            audioChunks = [];
            finalTranscript = '';
            
            const btn = document.getElementById('voiceBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('voiceStatus');
            const streamingText = document.getElementById('streamingText');
            const interimResult = document.getElementById('interimResult');
            const finalResult = document.getElementById('finalResult');
            
            btn.style.background = 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)';
            btn.innerHTML = '🔴 录音中...';
            btn.style.transform = 'scale(1.05)';
            stopBtn.style.display = 'inline-block';
            status.innerHTML = '🎙️ 请说话...（说完松开按钮）';
            streamingText.style.display = 'block';
            interimResult.innerHTML = '🎙️ 正在录音...';
            finalResult.innerHTML = '';
            
            mediaRecorder.start();
            isRecording = true;
            
            // 15秒超时
            recordingTimeout = setTimeout(function() {
                if (isRecording) {
                    status.innerHTML = '⏱️ 录音超时，已自动停止';
                    stopRecording();
                }
            }, 15000);
        }
        
        // 发送音频到本地ASR服务器
        async function transcribeAudio(audioBlob) {
            try {
                const status = document.getElementById('voiceStatus');
                const interimResult = document.getElementById('interimResult');
                const finalResult = document.getElementById('finalResult');
                
                interimResult.innerHTML = '🔄 正在识别...';
                status.innerHTML = '🤖 本地AI识别中...';
                
                // 转换为base64
                const reader = new FileReader();
                reader.onload = async function() {
                    const base64Audio = reader.result.split(',')[1];
                    
                    // 发送到本地ASR服务器
                    const response = await fetch('http://127.0.0.1:5001/transcribe', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            audio: base64Audio,
                            language: 'zh'
                        })
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        finalTranscript = result.text;
                        finalResult.innerHTML = finalTranscript;
                        interimResult.innerHTML = '✅ 识别完成';
                        status.innerHTML = '✅ 识别完成';
                        
                        if (finalTranscript.trim()) {
                            copyToTextarea(finalTranscript);
                        }
                    } else {
                        throw new Error('识别失败');
                    }
                };
                reader.readAsDataURL(audioBlob);
                
            } catch (error) {
                console.error('本地ASR识别失败:', error);
                document.getElementById('voiceStatus').innerHTML = '❌ 识别失败';
                document.getElementById('interimResult').innerHTML = '❌ 识别失败';
            }
        }
        
        // 将文字复制到输入框
        function copyToTextarea(text) {
            const textarea = parent.document.querySelector('textarea[key="input"]');
            if (!textarea) {
                const textareas = parent.document.querySelectorAll('textarea');
                for (let ta of textareas) {
                    if (ta.placeholder && ta.placeholder.includes('患者回答')) {
                        ta.value = text;
                        ta.dispatchEvent(new Event('input', { bubbles: true }));
                        ta.dispatchEvent(new Event('change', { bubbles: true }));
                        ta.focus();
                        return;
                    }
                }
            } else {
                textarea.value = text;
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                textarea.dispatchEvent(new Event('change', { bubbles: true }));
                textarea.focus();
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                isRecording = false;
                if (recordingTimeout) {
                    clearTimeout(recordingTimeout);
                }
            }
        }
        
        function resetButton() {
            const btn = document.getElementById('voiceBtn');
            const stopBtn = document.getElementById('stopBtn');
            const streamingText = document.getElementById('streamingText');
            
            btn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            btn.innerHTML = '🎤 按住说话';
            btn.style.transform = 'scale(1)';
            stopBtn.style.display = 'none';
            
            // 3秒后隐藏识别框
            setTimeout(() => {
                streamingText.style.display = 'none';
            }, 3000);
        }
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', function() {
            console.log('本地ASR语音识别组件已加载');
            initAudioRecording();
        });
        </script>
        """
        st.components.v1.html(voice_html, height=300)
    
    # 文本输入
    answer = st.text_area("", placeholder="输入患者回答...", height=52, key="input", label_visibility="collapsed")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("发送", type="primary", use_container_width=True):
            if answer.strip():
                # 处理用户回答
                st.session_state.chat_history.append({"role": "user", "content": answer})
                
                # 调用Agent处理
                agent = get_agent()
                try:
                    result = agent.process_turn(answer)
                    next_q = result.get('next_question', '')
                    
                    if next_q:
                        st.session_state.chat_history.append({"role": "ai", "content": next_q})
                        
                        # 如果启用语音，生成语音
                        if st.session_state.voice_enabled:
                            try:
                                tts = get_tts()
                                audio_path = tts.text_to_speech(next_q)
                                with open(audio_path, "rb") as audio_file:
                                    audio_bytes = audio_file.read()
                                    st.session_state.last_audio = base64.b64encode(audio_bytes).decode()
                            except Exception as e:
                                print(f"[WARNING] 语音合成失败: {e}")
                    
                    st.session_state.waiting_for_answer = True
                except Exception as e:
                    st.error(f"处理失败: {e}")
                
                st.rerun()
    
    with col2:
        if st.button("重新开始", use_container_width=True):
            st.session_state.session_started = False
            st.session_state.chat_history = []
            st.session_state.current_dimension_index = 0
            st.session_state.waiting_for_answer = False
            st.rerun()
    
    with col3:
        if st.button("结束评估", use_container_width=True):
            st.session_state.session_started = False
            st.session_state.waiting_for_answer = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 开始按钮
if not st.session_state.session_started:
    if st.button("开始评估", type="primary"):
        if not st.session_state.patient_profile['name']:
            st.error("请先填写患者姓名")
        else:
            st.session_state.session_started = True
            st.session_state.chat_history = []
            st.session_state.current_dimension_index = 0
            
            # 生成第一个问题
            current_dimension = MMSE_DIMENSIONS[st.session_state.current_dimension_index]
            q = f"您好，{st.session_state.patient_profile['name']}！我是您的评估医生。现在我们来做一个简单的认知评估。首先，请告诉我今天是几月几日？"
            
            st.session_state.chat_history.append({"role": "ai", "content": q})
            
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
        st.session_state.current_dimension_index = 0
        st.session_state.waiting_for_answer = False
        st.rerun()
