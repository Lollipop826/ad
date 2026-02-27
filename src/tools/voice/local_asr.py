"""
本地流式ASR工具 - 使用Whisper模型
完全本地运行，无需API
"""

import os
import sys
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from typing import Optional, Callable, Generator
import tempfile
import torch
import whisper
from datetime import datetime

class LocalASR:
    """本地流式语音识别器"""
    
    def __init__(self, model_size: str = "base", language: str = "zh"):
        """
        初始化本地ASR
        
        Args:
            model_size: Whisper模型大小 (tiny, base, small, medium, large)
            language: 识别语言 (zh, en, etc.)
        """
        self.model_size = model_size
        self.language = language
        self.model = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.processing_thread = None
        self.callback = None
        
        # 音频参数
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # 每0.5秒处理一次
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # 流式识别参数
        self.vad_threshold = 0.5  # 语音活动检测阈值
        self.silence_duration = 2.0  # 静音持续时间（秒）
        self.min_audio_length = 1.0  # 最小音频长度（秒）
        
        print(f"[ASR] 初始化本地ASR，模型: {model_size}, 语言: {language}")
        
    def load_model(self):
        """加载Whisper模型"""
        try:
            print(f"[ASR] 正在加载Whisper模型: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            print(f"[ASR] 模型加载完成")
            return True
        except Exception as e:
            print(f"[ASR] 模型加载失败: {e}")
            return False
    
    def start_recording(self, callback: Optional[Callable[[str, bool], None]] = None):
        """
        开始录音和流式识别
        
        Args:
            callback: 识别结果回调函数 callback(text, is_final)
        """
        if self.is_recording:
            print("[ASR] 已经在录音中")
            return False
            
        if not self.model:
            if not self.load_model():
                return False
        
        self.callback = callback
        self.is_recording = True
        
        # 启动录音线程
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("[ASR] 开始录音和识别")
        return True
    
    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join(timeout=1)
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
            
        print("[ASR] 停止录音")
    
    def _record_audio(self):
        """录音线程"""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ):
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as e:
            print(f"[ASR] 录音错误: {e}")
    
    def _audio_callback(self, indata, frames, time, status):
        """音频数据回调"""
        if status:
            print(f"[ASR] 音频状态: {status}")
        
        # 将音频数据放入队列
        audio_data = indata[:, 0]  # 取单声道
        self.audio_queue.put(audio_data.copy())
    
    def _process_audio(self):
        """音频处理线程 - 流式识别"""
        audio_buffer = []
        last_voice_time = time.time()
        
        while self.is_recording or not self.audio_queue.empty():
            try:
                # 获取音频数据
                if not self.audio_queue.empty():
                    chunk = self.audio_queue.get(timeout=0.1)
                    audio_buffer.extend(chunk)
                    last_voice_time = time.time()
                else:
                    time.sleep(0.1)
                    continue
                
                # 检查是否有足够的音频数据
                if len(audio_buffer) < self.sample_rate * self.min_audio_length:
                    continue
                
                # 检查是否应该处理（有语音活动或超时）
                current_time = time.time()
                has_voice = self._detect_voice_activity(audio_buffer)
                
                should_process = (
                    has_voice or 
                    (current_time - last_voice_time) > self.silence_duration or
                    len(audio_buffer) > self.sample_rate * 10  # 最大10秒
                )
                
                if should_process and len(audio_buffer) > 0:
                    # 转换为numpy数组
                    audio_array = np.array(audio_buffer, dtype=np.float32)
                    
                    # 执行识别
                    try:
                        result = self._recognize_audio(audio_array)
                        if result and self.callback:
                            # 判断是否为最终结果
                            is_final = not has_voice or (current_time - last_voice_time) > self.silence_duration
                            self.callback(result, is_final)
                    except Exception as e:
                        print(f"[ASR] 识别错误: {e}")
                    
                    # 清空缓冲区
                    audio_buffer = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ASR] 处理错误: {e}")
                time.sleep(0.1)
    
    def _detect_voice_activity(self, audio_data):
        """简单的语音活动检测"""
        if len(audio_data) == 0:
            return False
        
        # 计算RMS能量
        rms = np.sqrt(np.mean(audio_data ** 2))
        return rms > self.vad_threshold
    
    def _recognize_audio(self, audio_data):
        """使用Whisper识别音频"""
        try:
            # 确保音频长度合适
            if len(audio_data) < self.sample_rate * 0.5:  # 至少0.5秒
                return None
            
            # 使用Whisper识别
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                fp16=False,  # 避免精度问题
                verbose=False
            )
            
            text = result.get("text", "").strip()
            return text if text else None
            
        except Exception as e:
            print(f"[ASR] Whisper识别错误: {e}")
            return None
    
    def recognize_file(self, audio_file_path: str) -> str:
        """
        识别音频文件
        
        Args:
            audio_file_path: 音频文件路径
            
        Returns:
            识别结果文本
        """
        if not self.model:
            if not self.load_model():
                return ""
        
        try:
            result = self.model.transcribe(
                audio_file_path,
                language=self.language,
                fp16=False
            )
            return result.get("text", "").strip()
        except Exception as e:
            print(f"[ASR] 文件识别错误: {e}")
            return ""

class StreamlitASR:
    """Streamlit集成的本地ASR"""
    
    def __init__(self, model_size: str = "base"):
        self.asr = LocalASR(model_size=model_size)
        self.recognition_result = ""
        self.is_final = False
    
    def start_recording(self):
        """开始录音"""
        def callback(text, is_final):
            if text:
                self.recognition_result = text
                self.is_final = is_final
                print(f"[ASR] 识别结果: {text} ({'最终' if is_final else '临时'})")
        
        return self.asr.start_recording(callback)
    
    def stop_recording(self):
        """停止录音"""
        self.asr.stop_recording()
    
    def get_result(self):
        """获取识别结果"""
        result = self.recognition_result
        is_final = self.is_final
        
        # 重置结果
        if is_final:
            self.recognition_result = ""
            self.is_final = False
        
        return result, is_final

# 使用示例
if __name__ == "__main__":
    # 测试本地ASR
    asr = LocalASR(model_size="base")
    
    def test_callback(text, is_final):
        print(f"识别结果: {text} ({'最终' if is_final else '临时'})")
    
    print("开始录音，按Ctrl+C停止...")
    try:
        asr.start_recording(test_callback)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        asr.stop_recording()
        print("录音已停止")
