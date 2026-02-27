"""
FastAPI WebSocket 语音服务器
接收浏览器音频流 → VAD检测 → Whisper → AI Agent
"""
import asyncio
import base64
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# 创建 base64 模块别名，避免嵌套函数闭包问题
_base64 = base64

# 加载环境变量
load_dotenv()

# 强制设置 HuggingFace 镜像（防止 .env 未生效）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
# import whisper  # 替换为 SenseVoice
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import soundfile as sf

# 导入 Silero VAD（直接使用本地缓存，避免网络问题）
# 尝试多个可能的缓存路径
cache_paths = [
    '/root/autodl-tmp/.cache/torch/hub/snakers4_silero-vad_master',
    os.path.expanduser('~/.cache/torch/hub/snakers4_silero-vad_master')
]

vad_model = None
utils = None

for cache_dir in cache_paths:
    if os.path.exists(cache_dir):
        try:
            print(f"[VAD] 从本地缓存加载 Silero VAD: {cache_dir}")
            vad_model, utils = torch.hub.load(
                repo_or_dir=cache_dir,
                model='silero_vad',
                source='local',  # 使用本地源，不联网
                force_reload=False,
                onnx=False
            )
            print(f"[VAD] ✅ 成功加载 Silero VAD")
            break
        except Exception as e:
            print(f"[VAD] ⚠️  从 {cache_dir} 加载失败: {e}")
            continue

if vad_model is None or utils is None:
    raise RuntimeError(f"无法加载 Silero VAD。检查的缓存路径: {cache_paths}")

(get_speech_timestamps, _, read_audio, _, _) = utils

# 导入 Agent
import os
USE_FUNCTION_CALLING = os.getenv("USE_FUNCTION_CALLING", "true").lower() == "true"

if USE_FUNCTION_CALLING:
    from src.agents.screening_agent_function_calling import ADScreeningAgentFunctionCalling as ADScreeningAgent
    AGENT_TYPE = "FunctionCalling (快速)"
else:
    from src.agents.screening_agent import ADScreeningAgent
    AGENT_TYPE = "ReAct (标准)"

from src.domain.dimensions import MMSE_DIMENSIONS
# from src.tools.voice.tts_tool import VoiceTTS  # 旧的 Edge TTS
# from src.tools.voice.cosyvoice_tts import CosyVoiceTTS as VoiceTTS  # PaddleSpeech TTS
# from src.tools.voice.cosyvoice3_tts import CosyVoice3TTS as VoiceTTS  # CosyVoice3 TTS (太慢)
from src.tools.voice.zipvoice_tts import ZipVoiceTTS as VoiceTTS  # ZipVoice TTS (高质量流匹配)
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 📋 静态文件服务（用于 MMSE 图片等）
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# 全局实例
ASR_MODEL = None  # 改名为 ASR_MODEL，使用 SenseVoice
AGENT = None
TTS = None
SPEAKER_VERIFIER = None  # 声纹验证器


def clean_for_tts(text: str) -> str:
    """
    清理文本中的 markdown 符号，用于 TTS 输出
    例如：`**这是重点**` -> `这是重点`
    """
    import re
    if not text:
        return text
    # 移除 **, *, __, _ 等 markdown 标记
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
    text = re.sub(r'~~([^~]+)~~', r'\1', text)      # ~~strikethrough~~
    text = re.sub(r'`([^`]+)`', r'\1', text)        # `code`
    return text


def init_models():
    """初始化模型"""
    global ASR_MODEL, AGENT, TTS, SPEAKER_VERIFIER
    
    print("[初始化] 加载 SenseVoice-Small (多语言+情绪识别+事件检测)...")
    ASR_MODEL = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",  # 🔥 启用 VAD 自动切割长音频
        vad_kwargs={"max_single_segment_time": 30000},  # VAD 最大切割时长 30s
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        disable_pbar=True,
        disable_log=False,
        disable_update=True,  # 🔥 禁用更新检查，避免网络卡住
    )
    print("[初始化] ✅ SenseVoice-Small 加载完成（支持情绪识别+事件检测）")
    
    # 预热 SenseVoice ASR
    print("[初始化] 🔥 预热 SenseVoice ASR...")
    import tempfile
    import soundfile as sf
    dummy_audio = np.zeros(16000, dtype=np.float32)  # 1秒静音
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, dummy_audio, 16000)
    _ = ASR_MODEL.generate(input=temp_file.name, cache={}, language="auto", use_itn=True)
    os.unlink(temp_file.name)
    print("[初始化] ✅ SenseVoice ASR 预热完成")
    
    print("[初始化] 加载 Embedding 模型池 (BGE-M3)...")
    from src.tools.retrieval.embedding_pool import get_embedding_pool
    _ = get_embedding_pool()  # 触发 Embedding 模型加载和预热
    
    print(f"[初始化] 加载 Agent ({AGENT_TYPE})...")
    AGENT = ADScreeningAgent(use_local=True)
    print(f"[初始化] ✅ Agent 类型: {AGENT_TYPE}")
    
    print("[初始化] 加载 TTS...")
    TTS = VoiceTTS()
    
    print("[初始化] 加载声纹验证器 (ECAPA-TDNN)...")
    from src.tools.voice.speaker_verification import get_speaker_verifier
    SPEAKER_VERIFIER = get_speaker_verifier(threshold=0.25)
    print("[初始化] ✅ 声纹验证器加载完成")
    
    # 预加载地理位置信息（轻量）
    from src.utils.location_service import get_deployment_location
    location = get_deployment_location()
    print(f"[初始化] 📍 当前位置: {location.get('province', '未知')} {location.get('city', '未知')}")
    
    print("[初始化] ✅ 所有本地模型加载+预热完成")


class VADBuffer:
    """VAD 音频缓冲区"""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.buffer = []
        self.is_speaking = False
        self.silence_chunks = 0
        self.MAX_SILENCE_CHUNKS = 30  # 约2秒静音
        
    def add_chunk(self, audio_chunk):
        """添加音频块并检测VAD"""
        # Silero VAD 要求每次 512 个样本，需要分块
        VAD_CHUNK_SIZE = 512
        
        for i in range(0, len(audio_chunk), VAD_CHUNK_SIZE):
            chunk_512 = audio_chunk[i:i+VAD_CHUNK_SIZE]
            
            # 不足 512 个样本，保存到下次
            if len(chunk_512) < VAD_CHUNK_SIZE:
                self.buffer.append(chunk_512)
                continue
            
            # 转换为 torch tensor
            audio_tensor = torch.from_numpy(chunk_512.astype(np.float32))
            
            # VAD 检测
            speech_prob = vad_model(audio_tensor, self.sample_rate).item()
            
            if speech_prob > 0.5:  # 检测到语音
                if not self.is_speaking:
                    print(f"[VAD] 🎤 正在说话... (prob={speech_prob:.2f})")
                    self.is_speaking = True
                self.silence_chunks = 0
                self.buffer.append(chunk_512)
            else:  # 静音
                if self.is_speaking:
                    self.buffer.append(chunk_512)
                    self.silence_chunks += 1
                    
                    if self.silence_chunks >= self.MAX_SILENCE_CHUNKS:
                        # 静音足够长，返回完整语音
                        print(f"[VAD] ⏹️ 说话结束")
                        complete_audio = np.concatenate(self.buffer)
                        self.reset()
                        return complete_audio
        
        return None
    
    def has_speech(self, audio_chunk) -> float:
        """快速检测音频块是否包含语音，返回概率"""
        if len(audio_chunk) < 512:
            return 0.0
        
        # 取前512个样本检测
        chunk_512 = audio_chunk[:512]
        audio_tensor = torch.from_numpy(chunk_512.astype(np.float32))
        speech_prob = vad_model(audio_tensor, self.sample_rate).item()
        return speech_prob
    
    def reset(self):
        """重置缓冲区"""
        self.buffer = []
        self.is_speaking = False
        self.silence_chunks = 0


async def quick_asr(audio_data: np.ndarray) -> str:
    """
    快速ASR识别（用于打断检测）
    使用 SenseVoice-Small
    """
    try:
        print(f"[快速ASR] 开始识别，音频长度: {len(audio_data)/16000:.2f}秒")
        start_time = time.time()
        
        # 保存临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, audio_data, 16000)
        
        # SenseVoice 转录（按官方文档配置）
        result = await asyncio.to_thread(
            ASR_MODEL.generate,
            input=temp_file.name,
            cache={},
            language="auto",  # 自动检测语言：zh, en, yue, ja, ko, nospeech
            use_itn=True,  # 使用逆文本归一化（标点、数字转换）
            batch_size_s=60,  # 动态batch，总音频时长60s
            merge_vad=True,  # 合并VAD切割的短音频
            merge_length_s=15  # 合并后长度15s
        )
        
        # 提取文本和情绪（使用官方后处理工具）
        text = ""
        emotion = "neutral"
        event = "Speech"
        
        if result and len(result) > 0:
            # 使用官方后处理工具解析富文本
            raw_text = result[0].get("text", "")
            text = rich_transcription_postprocess(raw_text)
            
            # 解析情绪标签：<|HAPPY|>, <|SAD|>, <|ANGRY|>, <|NEUTRAL|> 等
            if "<|HAPPY|>" in raw_text:
                emotion = "happy"
            elif "<|SAD|>" in raw_text:
                emotion = "sad"
            elif "<|ANGRY|>" in raw_text:
                emotion = "angry"
            elif "<|FEARFUL|>" in raw_text:
                emotion = "fearful"
            elif "<|DISGUSTED|>" in raw_text:
                emotion = "disgusted"
            elif "<|SURPRISED|>" in raw_text:
                emotion = "surprised"
            
            # 解析事件标签：<|Speech|>, <|Applause|>, <|Laughter|>, <|Cry|> 等
            if "<|Applause|>" in raw_text:
                event = "Applause"
            elif "<|Laughter|>" in raw_text:
                event = "Laughter"
            elif "<|Cry|>" in raw_text:
                event = "Cry"
            elif "<|Cough|>" in raw_text:
                event = "Cough"
            elif "<|Sneeze|>" in raw_text:
                event = "Sneeze"
            
            print(f"[快速ASR] 情绪: {emotion}, 事件: {event}")
        
        os.unlink(temp_file.name)
        
        elapsed = time.time() - start_time
        print(f"[快速ASR] 识别完成: '{text}' (耗时 {elapsed:.2f}秒)")
        
        return text
    except Exception as e:
        print(f"[快速ASR] 错误: {e}")
        import traceback
        traceback.print_exc()
        return ""


async def judge_interrupt_intent(text: str) -> str:
    """
    使用LLM判断用户意图（用于打断检测）
    
    Returns:
        'backchannel' - 应答词（嗯、啊、对），不打断
        'complete' - 完整问题，打断
        'incomplete' - 不完整，暂不打断
    """
    # 简单规则优先（快速判断）
    backchannels = ["嗯", "啊", "哦", "对", "是的", "好的", "嗯嗯", "啊啊", "好", "嗯哼"]
    if text.strip() in backchannels:
        print(f"[语义判断] 应答词: {text}")
        return 'backchannel'
    
    # 常见打断词（明确的打断意图）
    interrupt_phrases = ["等一下", "等等", "停", "停一下", "等下", "慢着", "别说了", "停下", "打断一下"]
    if text.strip() in interrupt_phrases or any(phrase in text for phrase in interrupt_phrases):
        print(f"[语义判断] 打断词: {text}")
        return 'complete'
    
    # 太短，可能是噪音或应答词
    if len(text.strip()) < 2:
        print(f"[语义判断] 太短: {text}")
        return 'backchannel'
    
    # 使用本地LLM判断
    try:
        prompt = f"""判断下面这句话的意图，只输出一个字母：
B - 应答词（如"嗯"、"啊"、"对"等简短回应）
C - 完整的问题或陈述
I - 不完整的句子

句子："{text}"

只输出B、C或I其中一个字母："""
        
        # 使用 Agent 的 LLM 进行快速推理
        response = await asyncio.to_thread(
            AGENT.llm.invoke,
            prompt
        )
        
        # LocalQwenLLM 返回 str，ChatOpenAI 返回 AIMessage
        if hasattr(response, 'content'):
            result = response.content.strip().upper()
        else:
            result = str(response).strip().upper()
        
        # 提取首字母
        if 'B' in result:
            print(f"[语义判断] LLM判定为应答词: {text}")
            return 'backchannel'
        elif 'I' in result:
            print(f"[语义判断] LLM判定为不完整: {text}")
            return 'incomplete'
        else:
            print(f"[语义判断] LLM判定为完整: {text}")
            return 'complete'
            
    except Exception as e:
        print(f"[语义判断] LLM出错: {e}，默认为完整")
        return 'complete'


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    init_models()


@app.get("/")
async def index():
    """返回前端页面（禁缓存，确保每次加载最新版本）"""
    html_file = Path(__file__).parent / "static" / "voice_chat.html"
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(
                content=f.read(),
                headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
            )
    
    # 回退到简单页面
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>语音对话</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        #status { padding: 15px; margin: 20px 0; border-radius: 8px; background: #f0f0f0; }
        #status.active { background: #d4edda; color: #155724; }
        button { padding: 15px 30px; font-size: 18px; margin: 10px; border-radius: 8px; cursor: pointer; }
        #startBtn { background: #28a745; color: white; border: none; }
        #stopBtn { background: #dc3545; color: white; border: none; }
        #messages { margin-top: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; min-height: 300px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background: #e3f2fd; text-align: right; }
        .assistant { background: #f5f5f5; }
    </style>
</head>
<body>
    <h1>🎤 AD筛查语音对话</h1>
    <div id="status">未连接</div>
    
    <!-- 模式切换 -->
    <div style="margin: 20px 0; padding: 10px; background: #f8f9fa; border-radius: 8px;">
        <label style="margin-right: 20px;">
            <input type="radio" name="inputMode" value="voice" checked onchange="switchMode()"> 🎤 语音输入
        </label>
        <label>
            <input type="radio" name="inputMode" value="text" onchange="switchMode()"> 💬 文字输入
        </label>
    </div>
    
    <!-- 语音控制 -->
    <div id="voiceControls">
        <button id="startBtn" onclick="startRecording()">开始对话</button>
        <button id="stopBtn" onclick="stopRecording()" disabled>停止</button>
    </div>
    
    <!-- 文字输入 -->
    <div id="textControls" style="display: none;">
        <div style="display: flex; gap: 10px; align-items: center;">
            <input type="text" id="textInput" placeholder="💬 输入你的回答..." 
                   style="flex: 1; padding: 15px; font-size: 16px; border: 2px solid #ddd; border-radius: 8px;"
                   onkeypress="if(event.key==='Enter') sendText()">
            <button onclick="sendText()" style="background: #007bff; color: white; border: none; padding: 15px 30px;">发送</button>
        </div>
    </div>
    
    <div id="messages"></div>

    <script>
        let ws = null;
        let audioContext = null;
        let mediaStream = null;
        let scriptProcessor = null;
        let isRecording = false;

        function connect() {
            const wsUrl = `ws://${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('[WebSocket] 已连接');
                document.getElementById('status').textContent = '✅ 已连接，点击开始对话';
                document.getElementById('status').className = 'active';
            };
            
            ws.onclose = () => {
                console.log('[WebSocket] 断开');
                document.getElementById('status').textContent = '❌ 未连接';
                document.getElementById('status').className = '';
                setTimeout(connect, 3000);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
        }

        function handleMessage(data) {
            if (data.type === 'asr_result') {
                addMessage(data.text, 'user');
            } else if (data.type === 'ai_response') {
                addMessage(data.text, 'assistant');
            } else if (data.type === 'tts_audio') {
                playAudio(data.audio);
            }
        }

        async function startRecording() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                
                // 启用音频上下文（解决自动播放限制）
                if (audioContext.state === 'suspended') {
                    await audioContext.resume();
                }
                
                const source = audioContext.createMediaStreamSource(mediaStream);
                scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                
                scriptProcessor.onaudioprocess = (e) => {
                    if (!isRecording) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    const int16Array = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        int16Array[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                    }
                    ws.send(JSON.stringify({
                        type: 'audio',
                        data: Array.from(int16Array)
                    }));
                };
                
                source.connect(scriptProcessor);
                scriptProcessor.connect(audioContext.destination);
                
                isRecording = true;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('status').textContent = '🎤 正在监听...';
                console.log('[录音] 已开始');
                
            } catch (error) {
                console.error('[录音] 错误:', error);
                alert('无法访问麦克风');
            }
        }

        function stopRecording() {
            isRecording = false;
            if (scriptProcessor) scriptProcessor.disconnect();
            if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
            if (audioContext) audioContext.close();
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('status').textContent = '已停止';
            console.log('[录音] 已停止');
        }

        function playAudio(base64Audio) {
            const audio = new Audio('data:audio/wav;base64,' + base64Audio);
            audio.play().catch(err => {
                console.error('[音频播放] 错误:', err);
                // 如果错误是自动播放限制，提示用户
                if (err.name === 'NotAllowedError') {
                    console.log('[提示] 请点击页面任意处启用音频播放');
                }
            });
        }

        function addMessage(text, role) {
            const div = document.createElement('div');
            div.className = 'message ' + role;
            div.textContent = text;
            document.getElementById('messages').appendChild(div);
            div.scrollIntoView({ behavior: 'smooth' });
        }

        // 模式切换
        function switchMode() {
            const mode = document.querySelector('input[name="inputMode"]:checked').value;
            if (mode === 'voice') {
                document.getElementById('voiceControls').style.display = 'block';
                document.getElementById('textControls').style.display = 'none';
                if (isRecording) {
                    stopRecording();
                }
            } else {
                document.getElementById('voiceControls').style.display = 'none';
                document.getElementById('textControls').style.display = 'block';
                if (isRecording) {
                    stopRecording();
                }
                // 聚焦到输入框
                setTimeout(() => document.getElementById('textInput').focus(), 100);
            }
        }

        // 发送文字消息
        function sendText() {
            const input = document.getElementById('textInput');
            const text = input.value.trim();
            if (!text) return;
            
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                alert('未连接到服务器');
                return;
            }
            
            // 发送文字消息
            ws.send(JSON.stringify({
                type: 'text',
                data: text
            }));
            
            // 显示用户消息
            addMessage(text, 'user');
            
            // 清空输入框
            input.value = '';
            input.focus();
        }

        // 页面加载时连接
        connect();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点 - 全双工版本"""
    global SPEAKER_VERIFIER  # 🔥 声纹验证器懒加载需要写入全局变量
    await websocket.accept()
    client_id = str(id(websocket))
    print(f"[连接] 客户端 {client_id}")
    
    vad_buffer = VADBuffer()
    session_id = f"call_{int(time.time())}"

    history_file = f"data/voice_calls/{session_id}/messages.json"
    history_lock = asyncio.Lock()

    async def append_history(role: str, content: str):
        async with history_lock:
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            history = []
            if os.path.exists(history_file):
                try:
                    with open(history_file, 'r', encoding='utf-8') as f:
                        history = json.load(f)
                except Exception:
                    history = []

            if history:
                last = history[-1]
                if last.get('role') == role and (last.get('content') or '') == (content or ''):
                    return

            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            history.append({'role': role, 'content': content, 'timestamp': ts})

            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
    
    # 会话状态
    chat_history = []
    is_processing = False  # 处理锁
    patient_profile = {}  # 患者信息（等待前端发送）
    session_started = False  # 会话是否已开始
    has_sent_greeting = False  # 是否已发送开场问候
    speaker_enrolled = False  # 是否已注册声纹
    speaker_verify_enabled = False  # 是否启用声纹验证（默认关闭）
    
    # 🔑 全双工关键状态
    stop_generate = False  # 停止生成标志
    ai_speaking_until = 0.0  # AI预计说话结束时间
    interrupt_audio_buffer = []  # 打断检测音频缓冲
    
    # 打断检测配置
    INTERRUPT_MIN_DURATION = 0.7  # 至少累积1秒音频才判断（保证声纹识别准确性）
    AI_SPEAKING_BUFFER = 0.0  # 禁用保护时间，立即允许打断（提高实时性）
    last_speech_time = 0.0  # 上次检测到语音的时间
    waiting_for_complete = False  # 是否在等待不完整句子补全
    
    # 🎯 连接后等待患者信息
    print(f"[连接] 会话 {session_id} - 等待患者信息...")
    
    # 发送等待消息
    await websocket.send_json({
        'type': 'waiting_for_info',
        'message': '请先填写患者基本信息，然后点击「开始评估」'
    })
    
    async def process_speech(audio_data):
        """处理完整语音 - 内部函数，可访问外部状态"""
        nonlocal stop_generate, ai_speaking_until, is_processing, speaker_enrolled
        
        try:
            # 保存临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio_data, 16000)
            
            # 0. 声纹验证 - 过滤非目标说话人（只在启用时验证，不自动注册）
            # 🔍 调试：打印所有条件
            print(f"[声纹调试] SPEAKER_VERIFIER={SPEAKER_VERIFIER is not None}, "
                  f"speaker_enrolled={speaker_enrolled}, "
                  f"is_enrolled={SPEAKER_VERIFIER.is_enrolled if SPEAKER_VERIFIER else 'N/A'}, "
                  f"speaker_verify_enabled={speaker_verify_enabled}")
            
            if SPEAKER_VERIFIER and speaker_enrolled and SPEAKER_VERIFIER.is_enrolled and speaker_verify_enabled:
                is_target, similarity = SPEAKER_VERIFIER.verify(temp_file.name)
                print(f"[声纹] 🎯 验证结果: is_target={is_target}, similarity={similarity:.2f}")
                if not is_target:
                    print(f"[FLOW] ⏭️ 非目标说话人 (相似度: {similarity:.2f})，跳过处理")
                    os.unlink(temp_file.name)
                    return
                else:
                    print(f"[声纹] ✅ 目标说话人确认 (相似度: {similarity:.2f})")
            
            # 1. SenseVoice 转录（多语言+情绪+事件识别）
            print("\n[FLOW] 1️⃣ 语音转文字+情绪识别 (SenseVoice)...")
            result = ASR_MODEL.generate(
                input=temp_file.name,
                cache={},
                language="auto",  # 自动检测语言：zh, en, yue, ja, ko
                use_itn=True,  # 使用逆文本归一化（标点、数字转换）
                batch_size_s=60,  # 动态batch
                merge_vad=True,  # 合并VAD切割的短音频
                merge_length_s=15  # 合并后长度15s
            )
            
            # 提取文本、情绪、语言、事件
            text = ""
            emotion = "neutral"
            language = "zh"
            event = "Speech"
            
            if result and len(result) > 0:
                # 使用官方后处理工具解析富文本
                raw_text = result[0].get("text", "")
                text = rich_transcription_postprocess(raw_text)
                
                # 解析情绪标签（7种情绪）
                if "<|HAPPY|>" in raw_text:
                    emotion = "happy"
                elif "<|SAD|>" in raw_text:
                    emotion = "sad"
                elif "<|ANGRY|>" in raw_text:
                    emotion = "angry"
                elif "<|FEARFUL|>" in raw_text:
                    emotion = "fearful"
                elif "<|DISGUSTED|>" in raw_text:
                    emotion = "disgusted"
                elif "<|SURPRISED|>" in raw_text:
                    emotion = "surprised"
                
                # 解析语言标签
                if "<|zh|>" in raw_text:
                    language = "zh"
                elif "<|en|>" in raw_text:
                    language = "en"
                elif "<|yue|>" in raw_text:
                    language = "yue"
                elif "<|ja|>" in raw_text:
                    language = "ja"
                elif "<|ko|>" in raw_text:
                    language = "ko"
                
                # 解析事件标签（8种事件）
                if "<|Applause|>" in raw_text:
                    event = "Applause"
                elif "<|Laughter|>" in raw_text:
                    event = "Laughter"
                elif "<|Cry|>" in raw_text:
                    event = "Cry"
                elif "<|Cough|>" in raw_text:
                    event = "Cough"
                elif "<|Sneeze|>" in raw_text:
                    event = "Sneeze"
                elif "<|Breath|>" in raw_text:
                    event = "Breath"
                elif "<|BGM|>" in raw_text:
                    event = "BGM"
                
                print(f"[ASR] 语言: {language}, 情绪: {emotion}, 事件: {event}")
            
            os.unlink(temp_file.name)
            
            if not text:
                print(f"[ASR] ⚠️ 识别结果为空，跳过处理")
                return
            
            # 🔑 检查点1：ASR后检查打断
            if stop_generate:
                print(f"[FLOW] ⚠️ ASR后检测到打断，停止处理")
                return
            
            print(f"[ASR] ✅ 识别结果: \"{text}\" (情绪: {emotion})")
            if not await send_json_safe(websocket, {
                'type': 'asr_result', 
                'text': text,
                'emotion': emotion,  # 🔥 传递情绪信息
                'language': language
            }):
                return

            await append_history('user', text)
            
            chat_history.append({
                'role': 'user', 
                'content': text,
                'emotion': emotion  # 记录情绪到历史
            })
            
            # 2. Agent 流式处理 + 实时 TTS
            print(f"\n[FLOW] 2️⃣ 🚀 AI流式生成 + 实时TTS - 用户情绪: {emotion}...")
            
            # 根据用户情绪选择AI回复的情感
            tts_emotion = "gentle" if emotion in ["sad", "angry", "fearful"] else "neutral"
            
            # 🔥 提前发送 tts_start，前端只初始化播放器，不切换UI状态
            if not await send_json_safe(websocket, {
                'type': 'tts_start',
                'text': '正在生成...'
            }):
                return
            
            try:
                # 2. 调用 Agent 处理（同步，无流式）
                start_agent_time = time.time()
                result = AGENT.process_turn(
                    user_input=text,
                    session_id=session_id,
                    patient_profile={
                        'name': patient_profile.get('name', ''),
                        'age': int(patient_profile.get('age', 70)) if patient_profile.get('age') else 70, 
                        'gender': patient_profile.get('gender', ''),
                        'education_years': int(patient_profile.get('education_years', 6)) if patient_profile.get('education_years') else 6,
                    },
                    chat_history=chat_history,
                    current_emotion=emotion
                )
                print(f"[Agent] ✅ Agent 响应耗时: {time.time() - start_agent_time:.2f}s")
                
                response_text = result.get('output', '请继续')
                mmse_score = result.get('mmse_score')
                
                # 🔥 关键修复：将 AI 回复写回 chat_history，保持对话记忆
                chat_history.append({
                    'role': 'assistant',
                    'content': response_text
                })
                await append_history('assistant', response_text)
                
                # 🔥 并行触发：后台映射话题到任务
                # 这是本次优化的核心：利用 TTS 生成时间进行后台计算
                selected_topic = result.get('selected_topic')
                if selected_topic:
                    print(f"[FLOW] 🚀 触发后台映射任务: {selected_topic}")
                    asyncio.create_task(AGENT._background_global_analysis(selected_topic, chat_history))
                
                # 发送文字回复
                await websocket.send_json({
                    'type': 'ai_response',
                    'text': response_text
                })
                
                # 3. TTS 流式合成
                if response_text:
                    if stop_generate:
                        print("[TTS] ⚠️ 生成前检测到打断，跳过")
                    else:
                        tts_start_time = time.time()
                        chunk_count = 0
                        total_samples = 0
                        first_chunk_sent = False
                        
                        _last_tts_send = tts_start_time
                        async for audio_chunk in TTS.text_to_speech_streaming(clean_for_tts(response_text)):
                            if stop_generate:
                                print("[TTS] ⚠️ 流式生成中检测到打断，停止")
                                break
                            
                            _t_recv = time.time()
                            _gap = _t_recv - _last_tts_send
                            
                            chunk_bytes = audio_chunk.tobytes()
                            _t_enc_s = time.time()
                            chunk_base64 = _base64.b64encode(chunk_bytes).decode()
                            _t_enc = time.time() - _t_enc_s
                            _size_kb = len(chunk_base64) / 1024
                            
                            _t_send_s = time.time()
                            await send_json_safe(websocket, {
                                'type': 'tts_chunk',
                                'chunk': chunk_base64,
                                'sample_rate': 24000,
                                'dtype': 'float32'
                            })
                            _t_send = time.time() - _t_send_s
                            await asyncio.sleep(0)
                            
                            chunk_count += 1
                            total_samples += len(audio_chunk)
                            _last_tts_send = time.time()
                            
                            if not first_chunk_sent:
                                first_latency = time.time() - tts_start_time
                                print(f"[TTS] 🎵 首块! 延迟={first_latency:.2f}s")
                                first_chunk_sent = True
                            
                            print(f"[TTS] 📤 块{chunk_count} | t={time.time()-tts_start_time:.3f}s | "
                                  f"等yield={_gap:.3f}s | 编码={_t_enc*1000:.0f}ms({_size_kb:.0f}KB) | "
                                  f"发送={_t_send*1000:.0f}ms | 音频={len(audio_chunk)/24000:.1f}s")
                            
                        # 发送 tts_end + 设置说话时间
                        if total_samples > 0:
                            duration = total_samples / 24000.0
                            ai_speaking_until = time.time() + duration
                            await send_json_safe(websocket, {
                                'type': 'tts_end',
                                'duration': duration,
                                'chunks': chunk_count
                            })
                            print(f"[TTS] ✅ 流式发送完成: {chunk_count}块, {duration:.1f}s, 首块延迟{first_latency:.2f}s, AI说话至 {time.strftime('%H:%M:%S', time.localtime(ai_speaking_until))}")
                        else:
                            print(f"[TTS] ❌ 没有生成任何音频块")

            except Exception as e:
                print(f"[FLOW] ❌ 处理失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 📊 获取并发送最新MMSE评分
            try:
                mmse_tool = getattr(AGENT, 'mmse_tool', None)
                if not mmse_tool:
                    from src.tools.agent_tools import MMSEScoringTool
                    mmse_tool = MMSEScoringTool()
                
                summary_json = mmse_tool._run(
                    session_id=session_id, 
                    action="summary",
                    dimension_id="orientation", 
                    score=0
                )
                summary_data = json.loads(summary_json)
                if summary_data.get('success'):
                    await send_json_safe(websocket, {
                        'type': 'update_score',
                        'data': summary_data
                    })
            except Exception as e:
                print(f"[评分] 发送评分失败: {e}")

            # 标记完成
            print(f"[FLOW] 🏁 处理完成\n")
            
            # 清理状态
            is_processing = False
            
        except Exception as e:
            print(f"[错误] {e}")
            import traceback
            traceback.print_exc()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理心跳消息
            if message.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
                continue
            
            # 🔄 处理重置声纹请求
            if message.get('type') == 'reset_speaker':
                if SPEAKER_VERIFIER:
                    SPEAKER_VERIFIER.reset()
                    speaker_enrolled = False
                    print("[声纹] 🔄 用户请求重置声纹")
                    await websocket.send_json({
                        'type': 'speaker_reset',
                        'message': '声纹已重置，请重新录入'
                    })
                continue
                
            # 🔊 处理声纹验证开关请求
            if message.get('type') == 'toggle_speaker_verify':
                speaker_verify_enabled = message.get('enabled', False)
                # 🔥 懒加载声纹验证器：仅在用户首次开启时加载
                if speaker_verify_enabled and SPEAKER_VERIFIER is None:
                    print("[声纹] 🚀 首次开启，加载声纹验证器 (ECAPA-TDNN)...")
                    from src.tools.voice.speaker_verification import get_speaker_verifier
                    SPEAKER_VERIFIER = get_speaker_verifier(threshold=0.25)
                    print("[声纹] ✅ 声纹验证器加载完成")
                status = "开启" if speaker_verify_enabled else "关闭"
                print(f"[声纹] 🔊 用户{status}了声纹验证")
                await websocket.send_json({
                    'type': 'speaker_verify_status',
                    'enabled': speaker_verify_enabled,
                    'message': f'声纹验证已{status}'
                })
                continue
            
            # 🎚️ 更新声纹验证阈值
            if message.get('type') == 'update_speaker_threshold':
                new_threshold = message.get('threshold', 0.7)
                if SPEAKER_VERIFIER:
                    SPEAKER_VERIFIER.threshold = new_threshold
                    print(f"[声纹] 🎚️ 阈值已更新为: {new_threshold}")
                    await websocket.send_json({
                        'type': 'threshold_updated',
                        'threshold': new_threshold,
                        'message': f'阈值已更新为 {new_threshold:.2f}'
                    })
                continue
            
            # 🌍 更新地理位置（来自浏览器）
            if message.get('type') == 'update_location':
                lat = message.get('latitude')
                lon = message.get('longitude')
                if lat and lon:
                    print(f"[位置] 🌍 收到浏览器位置: {lat}, {lon}")
                    
                    # 使用反向地理编码获取城市名（使用 OpenStreetMap Nominatim）
                    import httpx
                    city_name = message.get('city')  # 前端可能已经传了城市名
                    province_name = message.get('province')
                    
                    print(f"[位置] 📦 浏览器传来的数据: city={city_name}, province={province_name}")
                    
                    # 如果前端没传城市名，用 Nominatim 反向编码
                    if not city_name:
                        print(f"[位置] ⚠️ 浏览器未传城市名，尝试反向编码...")
                        try:
                            with httpx.Client(timeout=5.0) as client:
                                # Nominatim 反向地理编码（免费，需要 User-Agent）
                                geo_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=zh"
                                headers = {"User-Agent": "ADScreeningApp/1.0"}
                                response = client.get(geo_url, headers=headers)
                                if response.status_code == 200:
                                    geo_data = response.json()
                                    address = geo_data.get('address', {})
                                    # 中国地址结构：state=省, city=市
                                    province_name = address.get('state', address.get('province', ''))
                                    city_name = address.get('city', address.get('county', address.get('town', '')))
                                    print(f"[位置] 🗺️ Nominatim反向编码: {province_name} {city_name}")
                        except Exception as e:
                            print(f"[位置] ⚠️ 反向地理编码失败: {e}")
                    else:
                        print(f"[位置] ✅ 使用浏览器传来的城市: {province_name} {city_name}")
                    
                    # 更新配置文件
                    from pathlib import Path
                    location_file = Path("config/deployment.json")
                    if location_file.exists():
                        with open(location_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        config['location']['lat'] = lat
                        config['location']['lon'] = lon
                        if province_name:
                            config['location']['province'] = province_name
                        if city_name:
                            config['location']['city'] = city_name
                        config['location']['source'] = 'browser-geolocation'
                        with open(location_file, 'w', encoding='utf-8') as f:
                            json.dump(config, f, ensure_ascii=False, indent=4)
                    
                    # 🔥 关键：刷新内存缓存（从配置文件读取，不重新获取）
                    from src.utils.location_service import get_location_from_config
                    import src.utils.location_service as loc_service
                    
                    # 从配置文件重新读取（已包含浏览器刚传来的数据）
                    updated_location = get_location_from_config()
                    if updated_location:
                        # 更新全局缓存
                        loc_service._cached_location = updated_location
                        print(f"[位置] ✅ 已更新并刷新缓存: {updated_location.get('province')} {updated_location.get('city')}")
                    else:
                        print(f"[位置] ⚠️ 配置文件读取失败")
                continue
            
            # 💾 保存声纹
            if message.get('type') == 'save_speaker':
                speaker_name = message.get('name', '')
                if speaker_name and SPEAKER_VERIFIER and SPEAKER_VERIFIER.is_enrolled:
                    success = SPEAKER_VERIFIER.save(speaker_name)
                    if success:
                        await websocket.send_json({
                            'type': 'speaker_saved',
                            'name': speaker_name,
                            'message': f'✅ 声纹已保存为「{speaker_name}」'
                        })
                    else:
                        await websocket.send_json({
                            'type': 'speaker_error',
                            'message': '保存失败，请重试'
                        })
                else:
                    await websocket.send_json({
                        'type': 'speaker_error',
                        'message': '请先完成声纹注册'
                    })
                continue
            
            # 📥 加载声纹
            if message.get('type') == 'load_speaker':
                speaker_name = message.get('name', '')
                if speaker_name and SPEAKER_VERIFIER:
                    success = SPEAKER_VERIFIER.load(speaker_name)
                    if success:
                        speaker_enrolled = True
                        await websocket.send_json({
                            'type': 'speaker_loaded',
                            'name': speaker_name,
                            'message': f'✅ 已加载声纹「{speaker_name}」'
                        })
                    else:
                        await websocket.send_json({
                            'type': 'speaker_error',
                            'message': f'加载失败，找不到「{speaker_name}」的声纹'
                        })
                continue
            
            # 📋 获取已保存的声纹列表
            if message.get('type') == 'list_speakers':
                from src.tools.voice.speaker_verification import SpeakerVerifier
                speakers = SpeakerVerifier.list_saved_speakers()
                await websocket.send_json({
                    'type': 'speakers_list',
                    'speakers': speakers
                })
                continue
            
            # 🎤 处理声纹样本注册
            if message.get('type') == 'enroll_speaker_sample':
                audio_base64 = message.get('audio')
                if audio_base64 and SPEAKER_VERIFIER:
                    import base64
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # 保存临时文件
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    sf.write(temp_file.name, audio_array, 16000)
                    
                    # 添加样本
                    success, current, needed = SPEAKER_VERIFIER.add_sample(temp_file.name)
                    os.unlink(temp_file.name)
                    
                    if success:
                        if SPEAKER_VERIFIER.is_enrolled:
                            speaker_enrolled = True
                            await websocket.send_json({
                                'type': 'speaker_enrolled',
                                'message': f'✅ 声纹注册完成 ({current}个样本)',
                                'current': current,
                                'needed': needed
                            })
                        else:
                            await websocket.send_json({
                                'type': 'speaker_sample_added',
                                'message': f'已添加 {current}/{needed} 个样本',
                                'current': current,
                                'needed': needed
                            })
                    else:
                        await websocket.send_json({
                            'type': 'speaker_error',
                            'message': '样本添加失败，请重试'
                        })
                continue
            
            # 🎯 处理开始会话请求（可选，用于更新患者信息）
            if message.get('type') == 'start_session':
                profile = message.get('profile', {})
                patient_profile = profile  # 保存到会话状态
                print(f"\n[开始] 收到用户信息: {profile}")
                
                # 根据患者信息生成个性化开场白
                if not has_sent_greeting and profile:
                    name = profile.get('name', '')
                    age = int(profile.get('age', 70)) if profile.get('age') else 70
                    gender = profile.get('gender', '女')
                    
                    # 生成称呼
                    if name:
                        if gender == '女':
                            if age >= 60:
                                greeting_name = f"{name}阿姨"
                            else:
                                greeting_name = f"{name}女士"
                        else:
                            if age >= 60:
                                greeting_name = f"{name}叔叔"
                            else:
                                greeting_name = f"{name}先生"
                    else:
                        greeting_name = "您"
                    
                    # 个性化开场白
                    welcome_message = f"{greeting_name}，您好呀！我是陪您聊天的小助手。咱们就随便聊聊，您想到什么就说什么就行～有时候我也会请您帮我确认一下时间、帮我算个小账，您别嫌我笨哈。那您今天感觉怎么样？心情还好吗？"
                    
                    # 发送文字和语音
                    try:
                        # 发送文字
                        await websocket.send_json({
                            'type': 'ai_response',
                            'text': welcome_message
                        })
                        
                        # 流式生成和发送语音
                        print(f"[开场] 🎵 流式语音合成 (TTS) - 情感: neutral...")
                        await websocket.send_json({'type': 'tts_start'})
                        
                        total_duration = 0.0
                        chunk_count = 0
                        
                        # 🔥 清理 markdown 符号
                        tts_text = clean_for_tts(welcome_message)
                        
                        import time as _time
                        tts_stream_start = _time.time()
                        last_send_time = tts_stream_start
                        async for audio_chunk in TTS.text_to_speech_streaming(tts_text, emotion="neutral"):
                            chunk_count += 1
                            chunk_duration = len(audio_chunk) / 24000.0
                            total_duration += chunk_duration
                            
                            t_recv = _time.time()
                            gap_from_last = t_recv - last_send_time
                            
                            # base64 编码计时
                            t_enc_start = _time.time()
                            chunk_b64 = _base64.b64encode(audio_chunk).decode('utf-8')
                            t_enc = _time.time() - t_enc_start
                            chunk_size_kb = len(chunk_b64) / 1024
                            
                            # WebSocket 发送计时
                            t_send_start = _time.time()
                            await websocket.send_json({
                                'type': 'tts_chunk',
                                'chunk': chunk_b64,
                                'sample_rate': 24000
                            })
                            t_send = _time.time() - t_send_start
                            
                            await asyncio.sleep(0)
                            
                            now = _time.time()
                            last_send_time = now
                            print(f"[开场] 📤 块 {chunk_count} | t={now-tts_stream_start:.3f}s | "
                                  f"等yield={gap_from_last:.3f}s | 编码={t_enc*1000:.0f}ms({chunk_size_kb:.0f}KB) | "
                                  f"发送={t_send*1000:.0f}ms | 音频={chunk_duration:.1f}s")
                        
                        # 发送结束信号
                        await websocket.send_json({
                            'type': 'tts_end',
                            'duration': total_duration
                        })
                        
                        # 设置AI说话状态
                        ai_speaking_until = time.time() + total_duration
                        print(f"[开场] ✅ 个性化开场问候已发送给{greeting_name}，AI将说话 {total_duration:.1f}s")
                        
                        has_sent_greeting = True
                        session_started = True
                        
                        # 🔥 关键修复：将开场白写入 chat_history，让 Agent 知道自己说过什么
                        chat_history.append({
                            'role': 'assistant',
                            'content': welcome_message
                        })
                        await append_history('assistant', welcome_message)
                        print(f"[开场] 📝 开场白已写入 chat_history (当前历史长度: {len(chat_history)})")
                        
                    except Exception as e:
                        print(f"[错误] 发送开场问候失败: {e}")
                        import traceback
                        traceback.print_exc()
                
                continue
            
            # 🔑 处理手动打断请求
            if message.get('type') == 'interrupt':
                print(f"\n[打断] 用户手动请求打断")
                stop_generate = True
                await websocket.send_json({'type': 'stop_tts'})
                # 等待处理完成
                await asyncio.sleep(0.1)
                is_processing = False
                stop_generate = False
                ai_speaking_until = 0.0
                interrupt_audio_buffer = []
                print(f"[打断] ✅ 手动打断完成")
                continue
            
            # 📝 处理文字消息
            if message.get('type') == 'text':
                text = message.get('data', '').strip()
                if text:
                    print(f"\n[文字输入] 用户: {text}")
                    
                    # 等待上一次处理完成
                    while is_processing:
                        await asyncio.sleep(0.1)
                    
                    # 直接处理文字（跳过 ASR）
                    is_processing = True
                    
                    # 🔥 通知前端进入处理阶段
                    await send_json_safe(websocket, {
                        'type': 'processing_status',
                        'stage': 'agent',
                        'text': '🧠 正在理解您的回答...'
                    })
                    
                    try:
                        # 添加到历史
                        chat_history.append({'role': 'user', 'content': text})
                        await append_history('user', text)
                        
                        # 调用 Agent
                        # 不传dimension，让Agent自己管理
                        result = AGENT.process_turn(
                            user_input=text,
                            session_id=session_id,
                            patient_profile=patient_profile if patient_profile else {'name': '测试', 'age': 70, 'gender': '女', 'education_years': 6},
                            chat_history=chat_history
                        )
                        
                        response = result.get('output', '请继续')
                        
                        # 🔥 并行触发：后台映射话题到任务
                        selected_topic = result.get('selected_topic')
                        if selected_topic:
                            print(f"[文字输入] 🚀 触发后台映射任务: {selected_topic}")
                            asyncio.create_task(AGENT._background_global_analysis(selected_topic, chat_history))
                        
                        print(f"[文字输入] AI: {response[:50]}...")
                        
                        # 发送AI回复
                        await websocket.send_json({
                            'type': 'ai_response',
                            'text': response
                        })
                        await append_history('assistant', response)
                        
                        # 📊 获取并发送最新MMSE评分
                        try:
                            # 1. 尝试从Agent获取评分工具（兼容不同Agent实现）
                            mmse_tool = getattr(AGENT, 'mmse_tool', None)
                            if not mmse_tool and hasattr(AGENT, 'score_tool'):
                                # 如果是fast agent，可能在score_tool里或者独立
                                pass
                            
                            # 2. 如果Agent没有直接暴露，重新实例化一个工具来读取（只要session_id一致即可）
                            if not mmse_tool:
                                from src.tools.agent_tools import MMSEScoringTool
                                mmse_tool = MMSEScoringTool()
                            
                            # 3. 获取评分汇总
                            # action='summary' 不需要 score/dimension_id，但参数校验需要
                            summary_json = mmse_tool._run(
                                session_id=session_id, 
                                action="summary",
                                dimension_id="orientation", # dummy
                                score=0 # dummy
                            )
                            
                            # 4. 发送给前端
                            summary_data = json.loads(summary_json)
                            if summary_data.get('success'):
                                await websocket.send_json({
                                    'type': 'update_score',
                                    'data': summary_data
                                })
                                print(f"[评分] 已发送MMSE更新: {summary_data.get('total_score')}/30")
                        except Exception as e:
                            print(f"[评分] 发送评分失败: {e}")
                        
                        # 生成并发送语音（文字模式默认使用温和语气）
                        audio_file = await TTS.text_to_speech_async(response, emotion="gentle")
                        with open(audio_file, 'rb') as f:
                            audio_bytes = f.read()
                        audio_base64 = _base64.b64encode(audio_bytes).decode('utf-8')
                        
                        # 获取音频时长
                        audio_duration = 3.0
                        try:
                            import wave
                            wav_file = audio_file.replace('.mp3', '.wav') if audio_file.endswith('.mp3') else audio_file
                            if os.path.exists(wav_file):
                                with wave.open(wav_file, 'rb') as wf:
                                    frames = wf.getnframes()
                                    rate = wf.getframerate()
                                    audio_duration = frames / float(rate)
                        except:
                            pass
                        
                        await websocket.send_json({
                            'type': 'tts_audio',
                            'audio': audio_base64,
                            'duration': audio_duration
                        })
                        os.unlink(audio_file)
                        
                        # 添加到历史
                        chat_history.append({'role': 'assistant', 'content': response})
                        ai_speaking_until = time.time() + audio_duration
                        
                    except Exception as e:
                        print(f"[文字输入] ❌ 处理失败: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        is_processing = False
                    
                    continue
            
            if message['type'] == 'audio':
                # 接收音频块
                audio_data = np.array(message['data'], dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # ⭐ 如果正在处理，静默丢弃所有音频（避免并发和重复日志）
                if is_processing and not waiting_for_complete:
                    # 🔥 修复：处理期间不再向 VAD buffer 送入音频，防止累积导致重复触发
                    continue
                
                # 🎯 VAD 检测（仅在非处理期间运行）
                complete_audio_from_vad = vad_buffer.add_chunk(audio_float)
                
                if complete_audio_from_vad is not None:
                    # VAD 检测到说话结束，立即发送 vad_end 给前端显示思考指示器
                    print(f"[VAD] 🎯 发送 vad_end 到前端")
                    await websocket.send_json({'type': 'vad_end'})
                
                # 🔑 全双工关键：AI说话时检测打断，或等待不完整句子补全
                if time.time() < ai_speaking_until or waiting_for_complete:
                    # AI正在说话，检查是否需要打断
                    
                    # 1. VAD检测：有语音吗？
                    speech_prob = vad_buffer.has_speech(audio_float)
                    current_time = time.time()
                    
                    # ⚠️ 提高VAD阈值，过滤噪音（只在明确的人声时触发）
                    if speech_prob > 0.7:
                        # 有语音，累积到缓冲
                        interrupt_audio_buffer.append(audio_float)
                        last_speech_time = current_time  # ⭐ 更新最后语音时间
                        
                        # 计算累积时长
                        total_duration = sum(len(a) for a in interrupt_audio_buffer) / 16000
                        
                        # 只在首次达到最小判断时长时判断（不在等待补全状态）
                        # 如果已经在等待补全，依靠停顿检测再次判断
                        should_judge = total_duration >= INTERRUPT_MIN_DURATION and not waiting_for_complete
                        
                        if should_judge:
                            # 声纹验证已启用，不再需要保护时间，直接判断
                            print(f"\n[全双工] 检测到用户声音，累积 {total_duration:.2f}s，进行意图判断...")
                            
                            # 拼接音频
                            full_audio = np.concatenate(interrupt_audio_buffer)
                            
                            # 声纹验证 - 过滤非目标说话人
                            if SPEAKER_VERIFIER and speaker_enrolled and SPEAKER_VERIFIER.is_enrolled and speaker_verify_enabled:
                                is_target, similarity = SPEAKER_VERIFIER.verify_from_audio(full_audio)
                                if not is_target:
                                    print(f"[全双工] ⏭️ 非目标说话人 (相似度: {similarity:.2f})，忽略打断")
                                    # 清空缓冲区但不打断AI的发言
                                    interrupt_audio_buffer = []
                                    waiting_for_complete = False
                                    continue
                            
                            # 快速ASR识别
                            text = await quick_asr(full_audio)
                            
                            # ⭐ 过滤噪音：只有识别出有效文本才认为是人在说话
                            if not text or len(text.strip()) < 2:
                                print(f"[全双工] ⚠️ 识别失败或文本过短，可能是噪音，忽略")
                                interrupt_audio_buffer = []
                                waiting_for_complete = False
                                continue
                            
                            if text:
                                print(f"[全双工] 识别到: \"{text}\"")
                                
                                # LLM判断意图
                                intent = await judge_interrupt_intent(text)
                                
                                # 处理不同意图
                                if intent == 'backchannel':
                                    # 应答词：不打断，清空缓冲
                                    print(f"[全双工] ❌ 应答词，不打断")
                                    interrupt_audio_buffer = []
                                    waiting_for_complete = False
                                    
                                elif intent == 'incomplete':
                                    # 不完整句子：先打断AI，但不处理输入，继续等待
                                    print(f"\n[全双工打断] 🛑 检测到用户说话（不完整），先停止AI播放...")
                                    
                                    # 停止AI播放
                                    stop_generate = True
                                    await websocket.send_json({'type': 'stop_tts'})
                                    await websocket.send_json({'type': 'interrupt'})
                                    await asyncio.sleep(0.1)
                                    
                                    # 重置AI说话状态，但保持音频缓冲和等待标志
                                    stop_generate = False
                                    ai_speaking_until = 0.0  # 清除AI说话状态
                                    waiting_for_complete = True  # 标记为等待补全
                                    # ⭐ 不清空 interrupt_audio_buffer，继续累积
                                    
                                    print(f"[全双工] ⏳ AI已停止，等待用户说完整...")
                                    
                                elif intent == 'complete':
                                    # 完整句子：打断AI并处理输入
                                    print(f"\n[全双工打断] 🛑 检测到有效打断（完整句子）！")
                                    
                                    # 🎯 发送 vad_end 消息给前端，触发思考指示器
                                    await websocket.send_json({'type': 'vad_end'})
                                    
                                    # 停止AI播放
                                    stop_generate = True
                                    await websocket.send_json({'type': 'stop_tts'})
                                    await websocket.send_json({'type': 'interrupt'})
                                    await asyncio.sleep(0.2)
                                    
                                    # 检查是否是明确的打断词
                                    interrupt_phrases = ["等一下", "等等", "停", "停一下", "等下", "慢着", "别说了", "停下", "打断一下"]
                                    is_interrupt_request = any(phrase in text for phrase in interrupt_phrases)
                                    
                                    if is_interrupt_request:
                                        # 是打断请求，立即回复确认
                                        print(f"[全双工打断] 📢 检测到打断请求，发送确认回复")
                                        
                                        # 发送简短的确认回复
                                        confirm_response = "好的，请说。"
                                        await websocket.send_json({
                                            'type': 'ai_response',
                                            'text': confirm_response
                                        })
                                        
                                        # 合成并发送语音（快速版）
                                        try:
                                            audio_file = await TTS.text_to_speech_async(confirm_response, emotion="gentle")
                                            with open(audio_file, 'rb') as f:
                                                audio_bytes = f.read()
                                            audio_base64 = _base64.b64encode(audio_bytes).decode('utf-8')
                                            
                                            await websocket.send_json({
                                                'type': 'tts_audio',
                                                'audio': audio_base64,
                                                'duration': 1.5  # 简短回复，大概1.5秒
                                            })
                                            os.unlink(audio_file)
                                            
                                            # 记录到历史
                                            chat_history.append({'role': 'assistant', 'content': confirm_response})
                                            
                                        except Exception as e:
                                            print(f"[全双工打断] ⚠️ 发送确认语音失败: {e}")
                                    
                                    # 重置所有状态，允许处理新输入
                                    stop_generate = False
                                    ai_speaking_until = 0.0
                                    is_processing = False
                                    waiting_for_complete = False
                                    interrupt_audio_buffer = []
                                    vad_buffer.reset()
                                    
                                    print(f"[全双工打断] ✅ 打断完成，等待用户新输入...")
                    else:
                        # 没有语音
                        if len(interrupt_audio_buffer) > 0:
                            # 如果在等待补全，检测停顿时间
                            if waiting_for_complete:
                                silence_duration = current_time - last_speech_time
                                if silence_duration > 0.5:  # 停顿超过0.5秒，认为说完了
                                    print(f"[全双工] 检测到停顿 {silence_duration:.2f}s，再次判断...")
                                    
                                    # 再次ASR + 判断
                                    full_audio = np.concatenate(interrupt_audio_buffer)
                                    text = await quick_asr(full_audio)
                                    
                                    if text:
                                        intent = await judge_interrupt_intent(text)
                                        
                                        if intent == 'complete':
                                            print(f"\n[全双工] ✅ 停顿后判定为完整句子，准备处理: \"{text}\"")
                                            
                                            # 🎯 发送 vad_end 消息给前端，触发思考指示器
                                            await websocket.send_json({'type': 'vad_end'})
                                            
                                            # AI已经停止（在incomplete时已停止），现在可以处理输入
                                            waiting_for_complete = False
                                            
                                            # 检查是否是明确的打断词
                                            interrupt_phrases = ["等一下", "等等", "停", "停一下", "等下", "慢着", "别说了", "停下", "打断一下"]
                                            is_interrupt_request = any(phrase in text for phrase in interrupt_phrases)
                                            
                                            if is_interrupt_request:
                                                # 是打断请求，立即回复确认
                                                print(f"[全双工] 📢 检测到打断请求，发送确认回复")
                                                
                                                # 发送简短的确认回复
                                                confirm_response = "好的，请说。"
                                                await websocket.send_json({
                                                    'type': 'ai_response',
                                                    'text': confirm_response
                                                })
                                                
                                                # 合成并发送语音（快速版）
                                                try:
                                                    audio_file = await TTS.text_to_speech_async(confirm_response, emotion="gentle")
                                                    with open(audio_file, 'rb') as f:
                                                        audio_bytes = f.read()
                                                    audio_base64 = _base64.b64encode(audio_bytes).decode('utf-8')
                                                    
                                                    await websocket.send_json({
                                                        'type': 'tts_audio',
                                                        'audio': audio_base64,
                                                        'duration': 1.5
                                                    })
                                                    os.unlink(audio_file)
                                                    
                                                    # 记录到历史
                                                    chat_history.append({'role': 'assistant', 'content': confirm_response})
                                                    
                                                except Exception as e:
                                                    print(f"[全双工] ⚠️ 发送确认语音失败: {e}")
                                                
                                                # 清空缓冲，等待用户新输入
                                                interrupt_audio_buffer = []
                                                is_processing = False
                                                # 🔥 修复：同步清空 VAD buffer
                                                vad_buffer.reset()
                                                complete_audio_from_vad = None
                                            else:
                                                # 不是打断请求，正常处理用户输入
                                                # 处理完整输入
                                                if not is_processing:
                                                    is_processing = True
                                                    try:
                                                        print(f"[全双工] 🎯 开始处理补全后的完整语音")
                                                        await process_speech(full_audio)
                                                    except Exception as e:
                                                        print(f"[全双工] ❌ 处理失败: {e}")
                                                    finally:
                                                        is_processing = False
                                                
                                                # 清空缓冲
                                                interrupt_audio_buffer = []
                                                # 🔥 修复双路径重复处理：全双工已处理，同步清空 VAD buffer
                                                vad_buffer.reset()
                                                complete_audio_from_vad = None
                                                print(f"[全双工] 🧹 已同步清空 VAD buffer，防止重复处理")
                                        else:
                                            # 仍然不完整或是应答词，放弃这段输入
                                            print(f"[全双工] ❌ 停顿后仍然不完整或是应答词，放弃: \"{text}\"")
                                            interrupt_audio_buffer = []
                                            waiting_for_complete = False
                            else:
                                # 不在等待状态，直接清空
                                interrupt_audio_buffer = []
                
                # 使用前面已经检测到的 complete_audio（避免重复调用 VAD）
                complete_audio = complete_audio_from_vad
                
                if complete_audio is not None:
                    # vad_end 已经在前面发送过了，这里不需要重复发送
                    
                    # ⭐ 如果正在等待补全，VAD检测到完整语音说明用户说完了
                    if waiting_for_complete:
                        print(f"[VAD] ✅ 等待补全时检测到完整语音，用户说完了")
                        
                        # 使用全双工累积的音频（如果有）
                        if len(interrupt_audio_buffer) > 0:
                            # 合并之前累积的音频和当前VAD的完整音频
                            full_audio_combined = np.concatenate(interrupt_audio_buffer + [complete_audio])
                            print(f"[VAD] 🎯 合并音频（累积 + VAD），总时长: {len(full_audio_combined)/16000:.2f}s")
                            
                            # 清空状态
                            waiting_for_complete = False
                            interrupt_audio_buffer = []
                            
                            # 处理合并后的完整音频
                            is_processing = True
                            try:
                                await process_speech(full_audio_combined)
                            except Exception as e:
                                print(f"[VAD] ❌ 处理失败: {e}")
                            finally:
                                is_processing = False
                            
                            continue
                        else:
                            # 没有累积音频，直接处理VAD的完整音频
                            print(f"[VAD] 🎯 无累积音频，直接处理VAD检测到的完整语音")
                            waiting_for_complete = False
                    
                    # ⭐ 如果AI正在说话，需要进行声纹验证再决定是否打断
                    if time.time() < ai_speaking_until:
                        print(f"[VAD] �️ 检测到完整语音（AI正在说话），用户主动打断或回答")
                        
                        # 声纹验证 - 非目标说话人不应该打断AI
                        should_interrupt = True
                        if SPEAKER_VERIFIER and speaker_enrolled and SPEAKER_VERIFIER.is_enrolled and speaker_verify_enabled:
                            # 使用精确的WAV格式验证
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                            sf.write(temp_file.name, complete_audio, 16000)
                            
                            is_target, similarity = SPEAKER_VERIFIER.verify(temp_file.name)
                            os.unlink(temp_file.name)
                            
                            if not is_target:
                                print(f"[VAD] ⏭️ 非目标说话人 (相似度: {similarity:.3f})，跳过处理")
                                should_interrupt = False
                        
                        # 只有目标说话人才打断AI
                        if should_interrupt:
                            # 停止AI播放
                            stop_generate = True
                            await websocket.send_json({'type': 'stop_tts'})
                            await websocket.send_json({'type': 'interrupt'})
                            await asyncio.sleep(0.1)
                            
                            # 清除AI说话状态
                            ai_speaking_until = 0.0
                            stop_generate = False
                            print(f"[VAD] ✅ AI已停止，准备处理用户输入")
                        else:
                            # 非目标说话人，不打断AI，跳过处理
                            is_processing = False
                            continue
                    
                    # 检测到完整语音，开始处理
                    is_processing = True
                    stop_generate = False
                    interrupt_audio_buffer = []  # 清空打断缓冲
                    
                    try:
                        # 调用内部定义的 process_speech 函数
                        await process_speech(complete_audio)
                    except WebSocketDisconnect:
                        print(f"[断开] 客户端在处理过程中断开 {client_id}")
                        break
                    except Exception as e:
                        print(f"[错误] 处理语音时出错: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        is_processing = False
                    
    except WebSocketDisconnect:
        print(f"[断开] 客户端 {client_id}")
    except RuntimeError as e:
        if "WebSocket is not connected" in str(e):
            print(f"[断开] WebSocket 连接已断开 {client_id}")
        else:
            raise
    except Exception as e:
        print(f"[错误] WebSocket 端点异常: {e}")
        import traceback
        traceback.print_exc()


async def send_json_safe(websocket: WebSocket, data: dict) -> bool:
    """安全发送 JSON 数据，如果连接断开则返回 False"""
    try:
        await websocket.send_json(data)
        return True
    except (WebSocketDisconnect, RuntimeError, Exception) as e:
        print(f"[WebSocket] 发送失败（连接可能已断开）: {e}")
        return False


if __name__ == "__main__":
    import uvicorn
    # 增加 WebSocket 超时配置，防止长时间处理时断开连接
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8502,
        ws_ping_interval=30.0,  # 每30秒发送 ping
        ws_ping_timeout=60.0,   # ping 超时时间60秒
        timeout_keep_alive=120  # 保持连接120秒
    )
