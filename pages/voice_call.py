
"""
实时语音对话模式 - 全自动连续对话
使用本地Whisper + VAD（语音活动检测）
"""

import streamlit as st
import streamlit.components.v1 as components
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置环境变量
os.environ['PATH'] = f"{os.path.expanduser('~/bin')}:{os.environ.get('PATH', '')}"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像源加速下载
# os.environ["HF_HUB_OFFLINE"] = "1"  # 注释掉离线模式，允许首次下载模型
# os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 注释掉离线模式

from src.agents.screening_agent import ADScreeningAgent
from src.domain.dimensions import MMSE_DIMENSIONS
from src.tools.voice.tts_tool import VoiceTTS
from datetime import datetime
from langchain_openai import ChatOpenAI
import soundfile as sf
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    audio_recorder = None
    
# 导入简单 VAD
import sys
sys.path.append('/root/autodl-tmp/langchain-mcp-adapters-main')
from components.simple_vad import simple_vad_recorder
import numpy as np
import whisper
import tempfile
import time
import asyncio
import json
import shutil
import base64

st.set_page_config(
    page_title="语音通话模式 - AD评估",
    page_icon="📞",
    layout="centered"
)

# 缓存Whisper模型
@st.cache_resource
def get_whisper_model():
    """加载Whisper模型（缓存）- 使用tiny模型（最快速度）"""
    try:
        print("[Whisper] 正在加载tiny模型（最快版）...")
        # 使用tiny模型：最快速度，适合实时对话
        # tiny模型虽然准确度稍低，但速度最快，适合语音对话场景
        model = whisper.load_model("tiny")
        print("[Whisper] ✅ 模型加载成功")
        return model
    except Exception as e:
        print(f"[Whisper] ❌ 模型加载失败: {e}")
        return None

# 缓存Agent实例
@st.cache_resource
def get_call_agent():
    """获取Agent实例（缓存）- 快速初始化"""
    try:
        print("[Voice Call] 正在快速初始化Agent...")
        # 使用快速初始化模式
        agent = ADScreeningAgent()
        print("[Voice Call] ✅ Agent初始化成功")
        return agent
    except Exception as e:
        print(f"[Voice Call] ❌ Agent初始化失败: {e}")
        return None

# 缓存TTS实例
@st.cache_resource
def get_tts():
    """获取TTS实例（缓存）- Edge TTS高质量初始化"""
    try:
        print("[TTS] 正在初始化Edge TTS（Microsoft云端高质量合成）...")
        # 使用Edge TTS温柔中文女声，高质量
        tts = VoiceTTS(voice="xiaoxiao", rate="-10%", volume="+0%")
        print("[TTS] ✅ Edge TTS高质量初始化成功（温柔中文女声）")
        return tts
    except Exception as e:
        print(f"[TTS] ❌ Edge TTS初始化失败: {e}")
        return None

@st.cache_resource
def get_openai_llm():
    """缓存化的LLM客户端（SiliconFlow/OpenAI兼容接口）。"""
    from src.llm.http_client_pool import get_siliconflow_chat_openai
    model = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    llm = get_siliconflow_chat_openai(
        model=model,
        temperature=0.7,
        timeout=30,
        max_retries=1,
    )
    print(f"[Greeting/Voice] Init shared LLM model={model} base_url={os.getenv('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1')}")
    return llm

def generate_llm_greeting(patient_info):
    """使用LLM生成个性化开场白（单句，温和自然）。失败时回退到固定文案。"""
    try:
        llm = get_openai_llm()
        name = patient_info.get('name') or ""
        age = patient_info.get('age') or 70
        sex = patient_info.get('sex') or "女"
        edu = patient_info.get('education_years') or 6

        system_prompt = (
            "你是一位温柔、耐心的老年科医生。只输出一段不超过40字的中文开场白，"
            "要自然亲切、不给压力，像家人聊天；可根据姓名/年龄/性别/教育年限调整称呼和用词；"
            "末尾加一个轻松的过渡短语，为开始提问做铺垫。不要加入引号和额外说明。"
        )
        user_prompt = (
            f"患者信息：姓名={name or '未知'}，年龄={age}，性别={sex}，教育年限={edu}年。"
        )
        resp = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        text = (resp.content or "").strip()
        if not text:
            raise ValueError("empty llm output")
        print(f"[Greeting/Voice] LLM greeting: {text}")
        return text
    except Exception as e:
        print(f"[Greeting] LLM生成失败，回退到默认：{e}")
        return "您好！我是AI评估助手，咱们轻松聊几句，先从简单的开始。"

def play_audio_sync(audio_file):
    """同步播放音频（使用Streamlit自动播放）"""
    try:
        print(f"[UI] 播放请求: {audio_file}")
        if os.path.exists(audio_file):
            import soundfile as _sf
            try:
                _info = _sf.info(audio_file)
                print(f"[UI] 文件存在 sr={_info.samplerate}Hz frames={_info.frames} dur={_info.frames/float(_info.samplerate):.3f}s")
            except Exception as e:
                print(f"[UI] 无法读取音频信息: {e}")
        else:
            print("[UI] 文件不存在")
        
        # 使用st.audio的autoplay参数实现自动播放
        with open(audio_file, 'rb') as _f:
            _audio_bytes = _f.read()
        
        # 使用autoplay=True启用自动播放
        st.audio(_audio_bytes, format='audio/wav', autoplay=True)
        
        # 等待音频播放完成（估算）
        data, samplerate = sf.read(audio_file)
        duration = len(data) / samplerate
        print(f"[UI] 等待播放完成（预计{duration:.1f}秒）...")
        time.sleep(duration)
        
    except Exception as e:
        print(f"[TTS] ❌ 音频播放失败: {e}")

def save_call_history():
    """保存对话历史（包括音频文件）"""
    if not st.session_state.call_session_id or not st.session_state.call_history:
        return None
    
    try:
        # 创建保存目录
        base_dir = "data/voice_calls"
        os.makedirs(base_dir, exist_ok=True)
        
        session_dir = os.path.join(base_dir, st.session_state.call_session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # 保存对话记录（JSON）
        history_data = {
            'session_id': st.session_state.call_session_id,
            'patient_info': st.session_state.patient_info,
            'start_time': st.session_state.call_history[0].get('timestamp', ''),
            'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'messages': []
        }
        
        for i, msg in enumerate(st.session_state.call_history):
            message_data = {
                'index': i,
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg.get('timestamp', ''),
            }
            
            # 如果有音频文件，复制到session目录
            if 'audio_file' in msg and msg['audio_file'] and os.path.exists(msg['audio_file']):
                audio_filename = f"{i:03d}_{msg['role']}.wav"
                dest_audio = os.path.join(session_dir, audio_filename)
                shutil.copy2(msg['audio_file'], dest_audio)
                message_data['audio_file'] = audio_filename
            
            history_data['messages'].append(message_data)
        
        # 保存JSON文件
        json_file = os.path.join(session_dir, 'conversation.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
        
        print(f"[保存] ✅ 对话记录已保存到: {session_dir}")
        return session_dir
    except Exception as e:
        print(f"[保存] ❌ 保存失败: {e}")
        return None

def load_call_sessions():
    """加载所有对话记录"""
    base_dir = "data/voice_calls"
    if not os.path.exists(base_dir):
        return []
    
    sessions = []
    for session_id in sorted(os.listdir(base_dir), reverse=True):
        session_dir = os.path.join(base_dir, session_id)
        json_file = os.path.join(session_dir, 'conversation.json')
        
        if os.path.isdir(session_dir) and os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        'session_id': session_id,
                        'patient_name': data.get('patient_info', {}).get('name', '未知'),
                        'start_time': data.get('start_time', ''),
                        'end_time': data.get('end_time', ''),
                        'message_count': len(data.get('messages', [])),
                        'data': data,
                        'dir': session_dir
                    })
            except Exception as e:
                print(f"[加载] ❌ 加载失败 {session_id}: {e}")
    
    return sessions

# 简单的VAD：检测音量
def simple_vad(audio_data, threshold=0.02):
    """简单的VAD（音量检测）"""
    return np.abs(audio_data).mean() > threshold


def websocket_vad_component(session_id, dimension, patient_info):
    """WebSocket 自动 VAD 组件"""
    # 添加版本号强制刷新缓存
    import time
    cache_buster = int(time.time())
    
    html_code = f"""
    <!-- Cache buster: {cache_buster} -->
    <div style="padding: 20px; background: #f0f2f6; border-radius: 10px; margin: 20px 0;">
        <div id="vad-status" style="padding: 15px; background: white; border-radius: 8px; margin-bottom: 15px; text-align: center; font-size: 18px;">
            <span style="color: #999;">⚪ 正在连接...</span>
        </div>
        <div id="volume-meter" style="height: 30px; background: #e0e0e0; border-radius: 15px; overflow: hidden; margin-bottom: 15px;">
            <div id="volume-bar" style="height: 100%; width: 0%; background: linear-gradient(90deg, #4caf50, #ff9800, #f44336); transition: width 0.1s;"></div>
        </div>
        <div style="text-align: center;">
            <button id="toggle-btn" onclick="toggleRecording()" style="padding: 12px 30px; font-size: 16px; border: none; border-radius: 8px; cursor: pointer; background: #4caf50; color: white; font-weight: bold;">
                🎤 开始监听
            </button>
        </div>
        
        <!-- MMSE 评分面板 -->
        <div id="mmse-panel" style="margin-top: 20px; padding: 15px; background: white; border-radius: 8px; display: none;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 8px;">
                <span style="font-weight: bold;">📊 MMSE 评估进度</span>
                <span id="mmse-total" style="color: #4caf50; font-weight: bold; font-size: 1.2em;">0/30</span>
            </div>
            <div id="mmse-status-text" style="font-size: 13px; color: #666; margin-bottom: 15px; text-align: right;">未评估</div>
            <div id="mmse-dims" style="display: flex; flex-direction: column; gap: 8px;"></div>
        </div>
    </div>

    <script>
    let ws = null;
    let audioContext = null;
    let mediaStream = null;
    let scriptProcessor = null;
    let isRecording = false;
    let streamingPlayer = null;
    
    // 🎵 流式音频播放器
    class StreamingAudioPlayer {{
        constructor() {{
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.chunks = [];
            this.isPlaying = false;
            this.nextStartTime = 0;
            this.activeSources = [];
        }}
        
        addChunk(chunkBase64, sampleRate = 22050) {{
            const binaryString = atob(chunkBase64);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {{
                bytes[i] = binaryString.charCodeAt(i);
            }}
            const floatArray = new Float32Array(bytes.buffer);
            const audioBuffer = this.audioContext.createBuffer(1, floatArray.length, sampleRate);
            audioBuffer.getChannelData(0).set(floatArray);
            this.playBuffer(audioBuffer);
        }}
        
        playBuffer(audioBuffer) {{
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            const currentTime = this.audioContext.currentTime;
            const startTime = Math.max(currentTime, this.nextStartTime);
            source.start(startTime);
            this.nextStartTime = startTime + audioBuffer.duration;
            this.activeSources.push(source);
            source.onended = () => {{
                const index = this.activeSources.indexOf(source);
                if (index > -1) this.activeSources.splice(index, 1);
            }};
        }}
        
        stop() {{
            this.activeSources.forEach(source => {{
                try {{ source.stop(); source.disconnect(); }} catch (e) {{}}
            }});
            this.activeSources = [];
            this.nextStartTime = this.audioContext.currentTime;
        }}
    }}
    
    // 动态获取 WebSocket URL（支持不同的访问地址）
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // 从父窗口获取 hostname（Streamlit 在 iframe 中运行）
    let hostname = '123.127.15.138';  // 默认值
    try {{
        if (window.parent && window.parent.location && window.parent.location.hostname) {{
            hostname = window.parent.location.hostname;
        }}
    }} catch (e) {{
        console.log('[WebSocket] 无法获取父窗口hostname，使用默认值');
    }}
    const WS_URL = protocol + '//' + hostname + ':8502/ws';
    const SESSION_ID = '{session_id}';
    
    console.log('[WebSocket] 连接地址:', WS_URL);
    
    function updateStatus(text, color) {{
        const statusEl = document.getElementById('vad-status');
        const colors = {{
            'connecting': '#999',
            'connected': '#4caf50',
            'listening': '#2196f3',
            'speaking': '#f44336',
            'processing': '#ff9800',
            'error': '#f44336'
        }};
        statusEl.innerHTML = `<span style="color: ${{colors[color] || '#999'}};">${{text}}</span>`;
    }}
    
    function connectWebSocket() {{
        ws = new WebSocket(WS_URL);
        
        ws.onopen = () => {{
            console.log('[WebSocket] 已连接');
            updateStatus('✅ 已连接，点击开始监听', 'connected');
        }};
        
        ws.onclose = () => {{
            console.log('[WebSocket] 断开');
            updateStatus('❌ 连接断开，3秒后重连...', 'error');
            setTimeout(connectWebSocket, 3000);
        }};
        
        ws.onerror = (error) => {{
            console.error('[WebSocket] 错误:', error);
            updateStatus('❌ 连接错误', 'error');
        }};
        
        ws.onmessage = (event) => {{
            const data = JSON.parse(event.data);
            handleMessage(data);
        }};
    }}
    
    function handleMessage(data) {{
        console.log('[收到消息]', data);
        
        if (data.type === 'vad_start') {{
            updateStatus('🎤 检测到语音...', 'speaking');
            if (streamingPlayer) streamingPlayer.stop(); // 打断播放
        }} else if (data.type === 'asr_result') {{
            updateStatus('✅ 识别: ' + data.text, 'processing');
        }} else if (data.type === 'ai_response') {{
            updateStatus('🤖 AI: ' + data.text, 'processing');
        }} else if (data.type === 'tts_start') {{
            // 开始流式播放
            streamingPlayer = new StreamingAudioPlayer();
            updateStatus('🎧 正在聆听AI回复...', 'listening');
        }} else if (data.type === 'tts_chunk') {{
            // 接收音频块
            if (streamingPlayer) {{
                streamingPlayer.addChunk(data.chunk, data.sample_rate || 22050);
            }}
        }} else if (data.type === 'tts_end') {{
            // 播放结束
            console.log('TTS接收完成');
        }} else if (data.type === 'tts_audio') {{
            // 兼容非流式
            playAudio(data.audio);
            updateStatus('🎧 正在监听...', 'listening');
        }} else if (data.type === 'update_score') {{
            updateScorePanel(data.data);
        }}
    }}
    
    function updateScorePanel(data) {{
        document.getElementById('mmse-panel').style.display = 'block';
        document.getElementById('mmse-total').textContent = `${{data.total_score}}/${{data.total_max_score}}`;
        document.getElementById('mmse-status-text').textContent = data.cognitive_status || '评估中...';
        
        const dimNames = {{
            'orientation': '定向力',
            'registration': '即时记忆',
            'attention_calculation': '注意力',
            'recall': '延迟回忆',
            'language': '语言能力',
            'copy': '构图能力'
        }};
        
        const scores = data.scoring_details?.dimension_scores || {{}};
        const orderedDims = ['orientation', 'registration', 'attention_calculation', 'recall', 'language', 'copy'];
        const dimsContainer = document.getElementById('mmse-dims');
        dimsContainer.innerHTML = '';
        
        orderedDims.forEach(dimId => {{
            const info = scores[dimId];
            const name = dimNames[dimId] || dimId;
            const score = info ? info.score : 0;
            const max = info ? info.max_score : (dimId === 'orientation' ? 10 : 3); // 简化默认值
            const percent = (score / max) * 100;
            
            const html = `
                <div style="font-size: 13px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                        <span>${{name}}</span>
                        <span style="color: #666;">${{score}}/${{max}}</span>
                    </div>
                    <div style="background: #e0e0e0; height: 6px; border-radius: 3px; overflow: hidden;">
                        <div style="background: ${{info ? '#4caf50' : '#ccc'}}; width: ${{info ? percent : 0}}%; height: 100%;"></div>
                    </div>
                </div>
            `;
            dimsContainer.insertAdjacentHTML('beforeend', html);
        }});
    }}
    
    async function toggleRecording() {{
        if (!isRecording) {{
            await startRecording();
        }} else {{
            stopRecording();
        }}
    }}
    
    async function startRecording() {{
        try {{
            mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
            audioContext = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate: 16000 }});
            const source = audioContext.createMediaStreamSource(mediaStream);
            
            // 创建分析器用于音量显示
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 2048;
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            source.connect(analyser);
            
            // 创建音频处理器
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {{
                if (!isRecording) return;
                
                // 更新音量条
                analyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
                document.getElementById('volume-bar').style.width = (average / 255 * 100) + '%';
                
                // 发送音频数据
                const inputData = e.inputBuffer.getChannelData(0);
                const int16Array = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {{
                    int16Array[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                }}
                
                if (ws && ws.readyState === WebSocket.OPEN) {{
                    ws.send(JSON.stringify({{
                        type: 'audio',
                        data: Array.from(int16Array)
                    }}));
                }}
            }};
            
            source.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);
            
            isRecording = true;
            document.getElementById('toggle-btn').textContent = '⏹️ 停止监听';
            document.getElementById('toggle-btn').style.background = '#f44336';
            updateStatus('🎧 正在监听...', 'listening');
            console.log('[录音] 已开始');
            
        }} catch (error) {{
            console.error('[录音] 错误:', error);
            alert('无法访问麦克风：' + error.message);
            updateStatus('❌ 麦克风访问失败', 'error');
        }}
    }}
    
    function stopRecording() {{
        isRecording = false;
        if (scriptProcessor) scriptProcessor.disconnect();
        if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
        if (audioContext) audioContext.close();
        
        document.getElementById('toggle-btn').textContent = '🎤 开始监听';
        document.getElementById('toggle-btn').style.background = '#4caf50';
        document.getElementById('volume-bar').style.width = '0%';
        updateStatus('⏸️ 已停止', 'connected');
        console.log('[录音] 已停止');
    }}
    
    function playAudio(base64Audio) {{
        try {{
            const audio = new Audio('data:audio/wav;base64,' + base64Audio);
            audio.play();
        }} catch (error) {{
            console.error('[播放] 错误:', error);
        }}
    }}
    
    // 页面加载时连接
    connectWebSocket();
    </script>
    """
    
    components.html(html_code, height=200)


def handle_new_conversation(user_text, ai_response):
    """处理新的对话轮次"""
    # 添加用户消息到历史
    st.session_state.call_history.append({
        'role': 'user',
        'content': user_text,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'audio_file': None
    })
    
    # 添加 AI 回复到历史
    if ai_response:
        st.session_state.call_history.append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'audio_file': None
        })
    
    # 刷新页面显示
    st.rerun()


if 'call_session_id' not in st.session_state:
    st.session_state.call_session_id = None
if 'call_history' not in st.session_state:
    st.session_state.call_history = []
if 'call_active' not in st.session_state:
    st.session_state.call_active = False
if 'current_dimension' not in st.session_state:
    st.session_state.current_dimension = 0
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {'name': '', 'age': 70, 'sex': '女', 'education_years': 6}
if 'listening' not in st.session_state:
    st.session_state.listening = False
if 'last_process_time' not in st.session_state:
    st.session_state.last_process_time = 0
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = 'voice'  # 'voice' 或 'text'

# 页面刷新时重置非活跃通话状态（避免音频缓存问题）
if 'page_loaded' not in st.session_state:
    st.session_state.page_loaded = True
    # 如果通话不活跃，清理旧状态
    if not st.session_state.call_active:
        st.session_state.call_history = []
        st.session_state.call_session_id = None

# 页面标题
st.title("📞 全自动语音对话（本地Whisper）")
st.markdown("---")

# 侧边栏 - 患者信息
with st.sidebar:
    st.header("📋 患者信息")
    
    disabled = st.session_state.call_active
    
    name = st.text_input("姓名", value=st.session_state.patient_info['name'], disabled=disabled)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄", 40, 100, st.session_state.patient_info['age'], disabled=disabled)
    with col2:
        sex = st.selectbox("性别", ["女", "男"], disabled=disabled)
    edu = st.number_input("教育年限", 0, 20, st.session_state.patient_info['education_years'], disabled=disabled)
    
    if not disabled:
        st.session_state.patient_info = {'name': name, 'age': age, 'sex': sex, 'education_years': edu}
    
    st.markdown("---")
    st.header("📊 通话进度")
    if st.session_state.call_active:
        dim = MMSE_DIMENSIONS[st.session_state.current_dimension]
        st.info(f"当前维度: {dim['name']}")
        st.progress(st.session_state.current_dimension / len(MMSE_DIMENSIONS))
    else:
        st.caption("通话未开始")

# 主界面
if not st.session_state.call_active:
    # 未开始通话 - 豆包风格欢迎页
    st.markdown("""
    <style>
    .welcome-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .welcome-icon {
        font-size: 80px;
        margin-bottom: 1rem;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .welcome-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .welcome-desc {
        color: rgba(255,255,255,0.95);
        font-size: 1.1rem;
        line-height: 1.8;
        margin: 1rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #333;
        margin: 0.5rem 0;
    }
    
    .feature-desc {
        color: #666;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    </style>
    
    <div class="welcome-card">
        <div class="welcome-icon">📞</div>
        <div class="welcome-title">AI 语音对话评估</div>
        <div class="welcome-desc">
            全自动语音识别 • 智能连续对话 • 本地隐私保护<br>
            像打电话一样自然交流，轻松完成认知评估
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 自动播放提示
    st.info("💡 **首次使用提示**：浏览器需要用户交互才能自动播放音频。如弹出提示，请点击【确定】启用自动播放。")
    
    # 开始通话和历史记录按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("📋 历史记录", use_container_width=True):
            st.session_state.show_history = not st.session_state.get('show_history', False)
            st.rerun()
    
    with col2:
        if st.button("📞 开始语音通话", type="primary", use_container_width=True, 
                     help="点击开始，系统会自动监听并识别您的语音"):
            if not name:
                st.error("❌ 请先在左侧填写患者姓名")
            else:
                with st.spinner("🚀 正在初始化AI助手、语音模型和温柔女声..."):
                    agent = get_call_agent()
                    whisper_model = get_whisper_model()
                    tts = get_tts()
                    
                    if agent is None:
                        st.error("❌ AI助手初始化失败，请刷新页面重试")
                    elif whisper_model is None:
                        st.error("❌ 语音模型加载失败，请检查Whisper安装")
                    elif tts is None:
                        st.error("❌ 语音合成失败，将继续使用文字模式")
                        st.session_state.call_agent = agent
                        st.session_state.whisper_model = whisper_model
                        st.session_state.tts = None
                        st.session_state.call_session_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        st.session_state.call_active = True
                        st.session_state.call_history = []
                        st.session_state.current_dimension = 0
                        st.session_state.listening = True
                        
                        # 使用LLM生成个性化开场白
                        welcome = generate_llm_greeting(st.session_state.patient_info)
                        st.session_state.call_history.append({
                            'role': 'assistant',
                            'content': welcome,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'audio_file': None
                        })
                        st.success("✅ 初始化成功！正在进入通话...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.session_state.call_agent = agent
                        st.session_state.whisper_model = whisper_model
                        st.session_state.tts = tts
                        st.session_state.call_session_id = f"call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        st.session_state.call_active = True
                        st.session_state.call_history = []
                        st.session_state.current_dimension = 0
                        st.session_state.listening = True
                        
                        # 使用LLM生成个性化开场白
                        welcome = generate_llm_greeting(st.session_state.patient_info)
                        
                        # 生成欢迎语音并保存
                        welcome_audio = None
                        try:
                            temp_audio = asyncio.run(tts.text_to_speech_async(welcome))
                            # 保存到持久目录
                            os.makedirs(f"data/voice_calls/{st.session_state.call_session_id}/audio", exist_ok=True)
                            welcome_audio = f"data/voice_calls/{st.session_state.call_session_id}/audio/ai_000.wav"
                            shutil.copy2(temp_audio, welcome_audio)
                            os.unlink(temp_audio)
                        except Exception as e:
                            print(f"[TTS] ❌ 欢迎语音生成失败: {e}")
                        
                        st.session_state.call_history.append({
                            'role': 'assistant',
                            'content': welcome,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'audio_file': welcome_audio
                        })
                        
                        # 播放欢迎语音
                        if welcome_audio and os.path.exists(welcome_audio):
                            try:
                                play_audio_sync(welcome_audio)
                            except Exception as e:
                                print(f"[TTS] ❌ 欢迎语音播放失败: {e}")
                        
                        st.success("✅ 初始化成功！正在进入通话...")
                        time.sleep(1)
                        st.rerun()
    
    # 功能特点卡片
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎤</div>
            <div class="feature-title">智能监听</div>
            <div class="feature-desc">自动检测语音活动，说话时自动录音，停顿时自动识别</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎙️</div>
            <div class="feature-title">温柔女声</div>
            <div class="feature-desc">AI回复自动语音播报，温柔女声安抚患者，真人般的对话体验</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <div class="feature-title">本地处理</div>
            <div class="feature-desc">使用本地Whisper模型，数据不上传，完全离线运行</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 显示历史记录
    if st.session_state.get('show_history', False):
        st.markdown("---")
        st.markdown("### 📋 历史对话记录")
        sessions = load_call_sessions()
        
        if not sessions:
            st.info("💡 暂无历史记录")
        else:
            for session in sessions:
                with st.expander(f"🗂️ {session['patient_name']} - {session['start_time']} ({session['message_count']}条消息)"):
                    st.markdown(f"**患者姓名**: {session['patient_name']}")
                    st.markdown(f"**开始时间**: {session['start_time']}")
                    st.markdown(f"**结束时间**: {session['end_time']}")
                    st.markdown(f"**消息总数**: {session['message_count']}")
                    
                    st.markdown("---")
                    st.markdown("#### 💬 对话内容")
                    for msg in session['data']['messages']:
                        role_icon = "🤖" if msg['role'] == 'assistant' else "👤"
                        role_name = "AI助手" if msg['role'] == 'assistant' else "患者"
                        st.markdown(f"**{role_icon} {role_name}** <span style='color: #999; font-size: 0.85em;'>({msg.get('timestamp', '')})</span>", unsafe_allow_html=True)
                        st.markdown(f"> {msg['content']}")
                        
                        # 音频文件路径（仅显示，不播放避免缓存问题）
                        if 'audio_file' in msg and msg['audio_file']:
                            st.caption(f"🔊 音频: {os.path.basename(msg.get('audio_file', ''))}")
                        st.markdown("")

else:
    # 通话进行中 - 豆包风格UI
    
    # 顶部状态卡片 - 带粒子背景和动画小人
    
    # 1. 注入 CSS
    st.markdown("""
    <style>
    /* 全局字体优化 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    /* 粒子背景容器 */
    .call-header-container {
        position: relative;
        width: 100%;
        height: 300px;
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
        border-radius: 20px;
        overflow: hidden;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    /* 粒子画布 */
    #particles-js {
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        z-index: 1;
    }

    /* AI小人容器 */
    .ai-avatar-container {
        position: relative;
        z-index: 2;
        width: 100px;
        height: 100px;
        margin-bottom: 1rem;
        margin-left: auto;
        margin-right: auto;
    }

    /* 简单的CSS小人 */
    .ai-avatar {
        width: 100%;
        height: 100%;
        background: white;
        border-radius: 50%;
        position: relative;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.5);
        animation: breathe 3s ease-in-out infinite;
    }

    .ai-face {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 60%;
        height: 40%;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .ai-eye {
        width: 10px;
        height: 10px;
        background: #333;
        border-radius: 50%;
        animation: blink-eye 4s infinite;
    }

    .ai-mouth {
        position: absolute;
        bottom: 8px;
        left: 50%;
        transform: translateX(-50%);
        width: 16px;
        height: 8px;
        border-radius: 0 0 20px 20px;
        background: #333;
        transition: height 0.2s;
    }

    /* 说话时的嘴巴动画 */
    .speaking .ai-mouth {
        animation: speak 0.5s infinite alternate;
    }

    @keyframes breathe {
        0%, 100% { transform: scale(1); box-shadow: 0 0 20px rgba(255, 255, 255, 0.5); }
        50% { transform: scale(1.05); box-shadow: 0 0 30px rgba(255, 255, 255, 0.8); }
    }

    @keyframes blink-eye {
        0%, 48%, 52%, 100% { transform: scaleY(1); }
        50% { transform: scaleY(0.1); }
    }

    @keyframes speak {
        0% { height: 4px; width: 16px; }
        100% { height: 12px; width: 20px; }
    }

    .status-text {
        position: relative;
        z-index: 2;
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    
    .status-subtext {
        position: relative;
        z-index: 2;
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }

    .listening-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #52c41a;
        border-radius: 50%;
        box-shadow: 0 0 10px #52c41a;
        animation: pulse-green 1.5s infinite;
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(82, 196, 26, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(82, 196, 26, 0); }
        100% { box-shadow: 0 0 0 0 rgba(82, 196, 26, 0); }
    }

    /* =========================================
       聊天气泡美化
       ========================================= */
    
    /* 基础容器 */
    .stChatMessage {
        background-color: transparent;
        border: none;
        padding: 1rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    /* 用户消息 - 蓝色调 */
    .stChatMessage[data-testid="user-message"] {
        background-color: #E3F2FD; /* 浅蓝背景 */
        border-radius: 20px 20px 5px 20px; /* 右下角直角 */
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.1);
        border: 1px solid #BBDEFB;
        margin-left: 2rem; /* 向右偏移一点 */
    }

    /* AI消息 - 白色/灰色调 */
    .stChatMessage:not([data-testid="user-message"]) {
        background-color: #FFFFFF;
        border-radius: 20px 20px 20px 5px; /* 左下角直角 */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #F0F0F0;
        margin-right: 2rem; /* 向左偏移一点 */
    }

    /* 头像容器 */
    .stChatMessage .stAvatar {
        background-color: #fff;
        border: 2px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* 消息内容文本 */
    .stChatMessage div[data-testid="stMarkdownContainer"] p {
        font-size: 16px;
        line-height: 1.6;
        color: #2d3748;
    }

    /* =========================================
       控制面板美化
       ========================================= */
    .control-panel {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.5);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 2. 注入 HTML 和 JavaScript (使用 components.html 隔离上下文)
    import streamlit.components.v1 as components
    
    # 使用 components.html 可以避免 Markdown 解析干扰，并提供独立的 iframe 环境
    # 这样 CSS 和 JS 都能正常工作，不会被转义
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
        body {
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: "Source Sans Pro", sans-serif;
            overflow: hidden;
        }
        
        /* 粒子背景容器 - 优化高度和渐变 */
        .call-header-container {
            position: relative;
            width: 100%;
            height: 260px;
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
            border-radius: 20px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            box-shadow: 0 10px 30px rgba(74, 144, 226, 0.3);
        }

        /* 粒子画布 */
        #particles-js {
            position: absolute;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: 1;
        }

        /* AI小人容器 */
        .ai-avatar-container {
            position: relative;
            z-index: 2;
            width: 90px;
            height: 90px;
            margin-bottom: 1rem;
            margin-left: auto;
            margin-right: auto;
        }

        /* 简单的CSS小人 */
        .ai-avatar {
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 50%;
            position: relative;
            box-shadow: 0 0 25px rgba(255, 255, 255, 0.4);
            animation: breathe 3s ease-in-out infinite;
            backdrop-filter: blur(5px);
        }

        .ai-face {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 60%;
            height: 40%;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .ai-eye {
            width: 10px;
            height: 10px;
            background: #2d3748;
            border-radius: 50%;
            animation: blink-eye 4s infinite;
        }

        .ai-mouth {
            position: absolute;
            bottom: 8px;
            left: 50%;
            transform: translateX(-50%);
            width: 16px;
            height: 8px;
            border-radius: 0 0 20px 20px;
            background: #2d3748;
            transition: height 0.2s;
        }

        /* 说话时的嘴巴动画 */
        .speaking .ai-mouth {
            animation: speak 0.5s infinite alternate;
        }

        @keyframes breathe {
            0%, 100% { transform: scale(1); box-shadow: 0 0 20px rgba(255, 255, 255, 0.4); }
            50% { transform: scale(1.05); box-shadow: 0 0 35px rgba(255, 255, 255, 0.6); }
        }

        @keyframes blink-eye {
            0%, 48%, 52%, 100% { transform: scaleY(1); }
            50% { transform: scaleY(0.1); }
        }

        @keyframes speak {
            0% { height: 4px; width: 16px; }
            100% { height: 12px; width: 20px; }
        }

        .status-text {
            position: relative;
            z-index: 2;
            color: white;
            font-size: 1.6rem;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 0.5rem;
            letter-spacing: 0.5px;
        }
        
        .status-subtext {
            position: relative;
            z-index: 2;
            color: rgba(255,255,255,0.95);
            font-size: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            background: rgba(255, 255, 255, 0.15);
            padding: 0.4rem 1rem;
            border-radius: 20px;
            backdrop-filter: blur(4px);
        }

        .listening-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            background: #00ff88;
            border-radius: 50%;
            box-shadow: 0 0 15px #00ff88;
            animation: pulse-green 1.5s infinite;
        }
        
        @keyframes pulse-green {
            0% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); transform: scale(1); }
            70% { box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); transform: scale(1.2); }
            100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); transform: scale(1); }
        }
        </style>
    </head>
    <body>
        <div class="call-header-container">
            <div id="particles-js"></div>
            
            <div class="ai-avatar-container">
                <div class="ai-avatar" id="ai-avatar">
                    <div class="ai-face">
                        <div class="ai-eye"></div>
                        <div class="ai-mouth"></div>
                        <div class="ai-eye"></div>
                    </div>
                </div>
            </div>
            
            <div class="status-text">
                AI 语音助手在线
            </div>
            <div class="status-subtext">
                <span class="listening-indicator"></span>
                <span>正在聆听您的声音...</span>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
        <script>
            // 确保 DOM 加载完成后执行
            document.addEventListener('DOMContentLoaded', function() {
                if (typeof particlesJS !== 'undefined') {
                    particlesJS("particles-js", {
                        "particles": {
                            "number": { "value": 60, "density": { "enable": true, "value_area": 800 } },
                            "color": { "value": "#ffffff" },
                            "shape": { "type": "circle", "stroke": { "width": 0, "color": "#000000" } },
                            "opacity": { "value": 0.5, "random": false },
                            "size": { "value": 3, "random": true },
                            "line_linked": { "enable": true, "distance": 150, "color": "#ffffff", "opacity": 0.4, "width": 1 },
                            "move": { "enable": true, "speed": 2, "direction": "none", "random": false, "straight": false, "out_mode": "out", "bounce": false }
                        },
                        "interactivity": {
                            "detect_on": "canvas",
                            "events": { "onhover": { "enable": true, "mode": "repulse" }, "onclick": { "enable": true, "mode": "push" }, "resize": true },
                            "modes": { "repulse": { "distance": 100, "duration": 0.4 }, "push": { "particles_nb": 4 } }
                        },
                        "retina_detect": true
                    });
                }
                
                // 简单的状态同步
                const avatar = document.getElementById('ai-avatar');
                if (avatar) {
                    setInterval(() => {
                        if (Math.random() > 0.7) {
                            avatar.classList.add('speaking');
                            setTimeout(() => avatar.classList.remove('speaking'), 2000);
                        }
                    }, 3000);
                }
            });
        </script>
    </body>
    </html>
    """
    
    # 渲染组件，设置高度以适应内容
    components.html(html_content, height=320)
    
    # 对话历史 - 美化版
    for msg in st.session_state.call_history:
        if msg['role'] == 'assistant':
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(f"<div style='font-size: 1.05rem; line-height: 1.6;'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"<div style='font-size: 1.05rem; line-height: 1.6;'>{msg['content']}</div>", unsafe_allow_html=True)
    
    # 控制面板
    st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if st.session_state.input_mode == 'voice':
            if st.session_state.listening:
                st.markdown("""
                <div style='display: flex; align-items: center; padding: 0.8rem;'>
                    <div class='listening-indicator'></div>
                    <span style='color: #52c41a; font-weight: 600;'>🎤 正在监听</span>
                    <span style='color: #999; margin-left: 1rem; font-size: 0.9rem;'>请说话，我在听...</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("⏸️ 通话已暂停")
        else:
            st.info("⌨️ 文字输入模式")
    
    with col2:
        # 模式切换按钮
        if st.session_state.input_mode == 'voice':
            if st.button("💬 切换到文字", use_container_width=True):
                st.session_state.input_mode = 'text'
                st.session_state.listening = False  # 停止语音监听
                st.rerun()
        else:
            if st.button("🎤 切换到语音", use_container_width=True):
                st.session_state.input_mode = 'voice'
                st.rerun()
    
    with col3:
        pass  # 预留空间
    
    with col4:
        if st.button("结束通话", type="secondary", use_container_width=True):
            # 保存对话历史
            saved_dir = save_call_history()
            if saved_dir:
                st.success(f"✅ 对话记录已保存")
            st.session_state.call_active = False
            st.session_state.listening = False
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 文字输入模式
    if st.session_state.input_mode == 'text':
        # 使用 st.chat_input 提供类似聊天的输入体验
        user_input = st.chat_input("💬 输入你的回答...", key="text_input")
        
        if user_input and user_input.strip():
            # 添加用户消息
            st.session_state.call_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'audio_file': None
            })
            
            # AI 回复
            with st.spinner("🤔 AI 正在思考..."):
                dim = MMSE_DIMENSIONS[st.session_state.current_dimension]
                hist = [{"role": m['role'], "content": m['content']} for m in st.session_state.call_history]
                
                result = st.session_state.call_agent.process_turn(
                    user_input=user_input,
                    dimension=dim,
                    session_id=st.session_state.call_session_id,
                    patient_profile=st.session_state.patient_info,
                    chat_history=hist
                )
                
                response = result.get('output', '请继续')
            
            # 生成 AI 语音（可选）
            ai_audio_file = None
            if st.session_state.tts:
                try:
                    with st.spinner("🔊 正在生成语音..."):
                        temp_audio = asyncio.run(st.session_state.tts.text_to_speech_async(response))
                        ai_audio_dir = f"data/voice_calls/{st.session_state.call_session_id}/audio"
                        os.makedirs(ai_audio_dir, exist_ok=True)
                        ai_audio_file = os.path.join(ai_audio_dir, f"ai_{len(st.session_state.call_history):03d}.wav")
                        shutil.copy2(temp_audio, ai_audio_file)
                        os.unlink(temp_audio)
                except Exception as e:
                    print(f"[TTS] ❌ 文字输入AI语音生成失败: {e}")
            
            # 添加 AI 回复
            st.session_state.call_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'audio_file': ai_audio_file
            })
            
            st.rerun()
    
    # WebSocket 自动 VAD 录音（语音模式）
    elif st.session_state.listening and st.session_state.input_mode == 'voice':
        # 嵌入 WebSocket 客户端
        websocket_vad_component(
            session_id=st.session_state.call_session_id,
            dimension=MMSE_DIMENSIONS[st.session_state.current_dimension],
            patient_info=st.session_state.patient_info
        )
        
        # 检查是否有新的对话数据（通过 session_state 传递）
        audio_bytes = None
        if 'new_user_text' in st.session_state and st.session_state.new_user_text:
            # 有新的用户输入，手动处理
            handle_new_conversation(
                st.session_state.new_user_text,
                st.session_state.get('new_ai_response', '')
            )
            st.session_state.new_user_text = None
            st.session_state.new_ai_response = None
        
        if audio_bytes:
            # 去重：检查是否是新录音
            audio_hash = hash(audio_bytes)
            if 'last_audio_hash' not in st.session_state:
                st.session_state.last_audio_hash = None
            
            if audio_hash == st.session_state.last_audio_hash:
                # 相同录音，跳过处理
                pass
            else:
                st.session_state.last_audio_hash = audio_hash
                
                try:
                    # 保存录音到临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                        f.write(audio_bytes)
                        temp_file = f.name
                    
                    st.info("🗣️ 检测到语音，正在识别...")
                            
                    # 使用Whisper识别
                    with st.spinner("🔄 正在识别..."):
                        result = st.session_state.whisper_model.transcribe(temp_file, language='zh')
                        text = result['text'].strip()
                    
                    # 删除临时文件
                    os.unlink(temp_file)
                    
                    if text and len(text) > 1:  # 至少1个字符
                        st.success(f"✅ 识别: {text}")
                        
                        # 保存用户音频（持久化）
                        user_audio_dir = f"data/voice_calls/{st.session_state.call_session_id}/audio"
                        os.makedirs(user_audio_dir, exist_ok=True)
                        user_audio_file = os.path.join(user_audio_dir, f"user_{len(st.session_state.call_history):03d}.wav")
                        with open(user_audio_file, 'wb') as f:
                            f.write(audio_bytes)
                        
                        # 添加到对话历史
                        st.session_state.call_history.append({
                            'role': 'user',
                            'content': text,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'audio_file': user_audio_file
                        })
                        
                        # AI回复
                        st.session_state.listening = False  # 暂停监听
                        with st.spinner("🤖 AI思考中..."):
                            dim = MMSE_DIMENSIONS[st.session_state.current_dimension]
                            hist = [{"role": m['role'], "content": m['content']} for m in st.session_state.call_history]
                            
                            result = st.session_state.call_agent.process_turn(
                                user_input=text,
                                dimension=dim,
                                session_id=st.session_state.call_session_id,
                                patient_profile=st.session_state.patient_info,
                                chat_history=hist
                            )
                            
                            response = result.get('output', '请继续')
                            
                            # 生成AI语音并保存
                            ai_audio_file = None
                            if st.session_state.tts:
                                try:
                                    temp_audio = asyncio.run(st.session_state.tts.text_to_speech_async(response))
                                    ai_audio_dir = f"data/voice_calls/{st.session_state.call_session_id}/audio"
                                    os.makedirs(ai_audio_dir, exist_ok=True)
                                    ai_audio_file = os.path.join(ai_audio_dir, f"ai_{len(st.session_state.call_history):03d}.wav")
                                    shutil.copy2(temp_audio, ai_audio_file)
                                    os.unlink(temp_audio)
                                except Exception as e:
                                    print(f"[TTS] ❌ TTS生成失败: {e}")
                            
                            st.session_state.call_history.append({
                                'role': 'assistant',
                                'content': response,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'audio_file': ai_audio_file
                            })
                            
                            # 检查维度切换
                            if st.session_state.current_dimension < len(MMSE_DIMENSIONS) - 1:
                                current_dim_qa_count = sum(1 for msg in st.session_state.call_history if msg['role'] == 'user')
                                if current_dim_qa_count >= 3:
                                    st.session_state.current_dimension += 1
                        
                        st.success(f"🤖 AI: {response}")
                        
                        # 播放AI回复语音
                        if ai_audio_file and os.path.exists(ai_audio_file):
                            play_audio_sync(ai_audio_file)
                        
                        # 自动重新开始监听
                        st.session_state.listening = True
                        st.rerun()
                    else:
                        st.warning("⚠️ 识别内容太短，请重新说话")
                        
                except Exception as e:
                    st.error(f"❌ 处理失败: {e}")
    
    # 提示：如果需要文字输入，请点击上方的"切换到文字"按钮
    if st.session_state.input_mode == 'voice':
        st.markdown("---")
        st.info("💡 提示：如果麦克风无法使用，可以点击上方的 **💬 切换到文字** 按钮进行打字输入")
    

# 页面说明
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### 💡 使用提示
    
    **🎤 语音模式：**
    - 开始后自动监听
    - 检测到说话自动录音
    - 停顿后自动识别
    - AI自动回复
    - 自动进入下一轮
    
    **💬 文字模式：**
    - 点击"切换到文字"按钮
    - 在输入框中打字
    - 按回车或点击发送
    - AI立即回复
    - 适合麦克风无法使用的情况
    
    ### 🔧 技术说明
    
    - VAD语音活动检测
    - 本地Whisper识别
    - 分级推理（7B+0.5B）
    - 支持语音/文字双模式
    """)
