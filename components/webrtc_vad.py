"""
WebRTC 实时语音检测 + 录音
支持全自动 VAD（语音活动检测）
"""
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import numpy as np
import threading
from collections import deque
import io
import wave


class AudioVADProcessor(AudioProcessorBase):
    """实时 VAD 音频处理器"""
    
    def __init__(self):
        self.is_speaking = False
        self.silence_frames = 0
        self.audio_buffer = deque(maxlen=500)  # 最多保存 10 秒（假设 16kHz, 20ms/frame）
        self.recording = False
        self.recorded_audio = []
        
        # VAD 参数
        self.ENERGY_THRESHOLD = 500  # 能量阈值
        self.SILENCE_DURATION = 100  # 静音帧数（约2秒）
        self.MIN_SPEECH_DURATION = 10  # 最少说话帧数
        
        self.lock = threading.Lock()
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """处理音频帧"""
        # 转换为 numpy 数组
        audio_array = frame.to_ndarray()
        
        # 计算能量
        energy = np.abs(audio_array).mean()
        
        with self.lock:
            if energy > self.ENERGY_THRESHOLD:
                # 检测到声音
                if not self.is_speaking:
                    self.is_speaking = True
                    self.recording = True
                    print(f"[VAD] 🎤 检测到声音 (能量: {energy:.2f})")
                
                self.silence_frames = 0
                self.recorded_audio.append(audio_array)
            else:
                # 静音
                if self.is_speaking:
                    self.silence_frames += 1
                    self.recorded_audio.append(audio_array)
                    
                    if self.silence_frames >= self.SILENCE_DURATION:
                        # 静音超时，停止录音
                        if len(self.recorded_audio) >= self.MIN_SPEECH_DURATION:
                            print(f"[VAD] ⏹️ 检测到静音，录音完成 (帧数: {len(self.recorded_audio)})")
                            self.recording = False
                        else:
                            print("[VAD] ⚠️ 录音太短，忽略")
                            self.recorded_audio.clear()
                        
                        self.is_speaking = False
                        self.silence_frames = 0
        
        return frame
    
    def get_recorded_audio(self):
        """获取录音数据"""
        with self.lock:
            if self.recorded_audio and not self.recording:
                audio_data = np.concatenate(self.recorded_audio, axis=1)
                self.recorded_audio.clear()
                return audio_data
            return None


def auto_voice_input_webrtc(key="webrtc_vad"):
    """
    全自动语音输入（WebRTC + VAD）
    
    Returns:
        audio_data: numpy array 或 None
    """
    
    # 配置多个 STUN 服务器提高连接成功率
    rtc_config = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun.stunprotocol.org:3478"]},
        ]
    }
    
    webrtc_ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioVADProcessor,
        media_stream_constraints={
            "video": False, 
            "audio": {
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
            }
        },
        rtc_configuration=rtc_config,
        async_processing=True,
    )
    
    if webrtc_ctx.audio_processor:
        # 检查是否有新录音
        audio_data = webrtc_ctx.audio_processor.get_recorded_audio()
        
        if audio_data is not None:
            # 保存为 WAV 格式
            return audio_to_wav_bytes(audio_data, sample_rate=48000)
    
    return None


def audio_to_wav_bytes(audio_array, sample_rate=48000):
    """将音频数组转换为 WAV 字节"""
    # 转换为 int16
    if audio_array.dtype != np.int16:
        audio_int16 = (audio_array * 32767).astype(np.int16)
    else:
        audio_int16 = audio_array
    
    # 如果是多声道，转为单声道
    if len(audio_int16.shape) > 1:
        audio_int16 = audio_int16.mean(axis=0).astype(np.int16)
    
    # 写入 WAV
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    return wav_buffer.getvalue()
