"""
音频录制工具
用于录制用户语音并保存为文件
"""

import pyaudio
import wave
import tempfile
import os
import time
from pathlib import Path


class AudioRecorder:
    """音频录制器"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 format: int = pyaudio.paInt16):
        """
        初始化音频录制器
        
        Args:
            sample_rate: 采样率，默认16000Hz（Whisper推荐）
            channels: 声道数，1=单声道，2=立体声
            chunk_size: 每次读取的音频块大小
            format: 音频格式
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format
        self.audio = None
        self.stream = None
        self.is_recording = False
        self.frames = []
    
    def start_recording(self):
        """开始录制"""
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.is_recording = True
            self.frames = []
            print("[RECORDER] 🎤 开始录制...")
            return True
        except Exception as e:
            print(f"[RECORDER] ❌ 录制启动失败: {e}")
            return False
    
    def record_chunk(self):
        """录制一个音频块"""
        if not self.is_recording:
            return None
        
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            self.frames.append(data)
            return data
        except Exception as e:
            print(f"[RECORDER] ❌ 录制块失败: {e}")
            return None
    
    def stop_recording(self) -> str:
        """
        停止录制并保存文件
        
        Returns:
            str: 保存的音频文件路径
        """
        if not self.is_recording:
            return ""
        
        self.is_recording = False
        
        try:
            # 停止流
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            # 关闭音频
            if self.audio:
                self.audio.terminate()
            
            # 保存音频文件 - 使用外部存储避免系统盘满
            temp_dir = Path(os.path.dirname(__file__)).parent.parent.parent / "tmp" / "ad_screening_voice"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = int(time.time() * 1000)
            audio_path = str(temp_dir / f"recording_{timestamp}.wav")
            
            with wave.open(audio_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
            
            duration = len(self.frames) * self.chunk_size / self.sample_rate
            print(f"[RECORDER] ✅ 录制完成: {audio_path} (时长: {duration:.2f}秒)")
            
            return audio_path
            
        except Exception as e:
            print(f"[RECORDER] ❌ 保存失败: {e}")
            return ""
    
    def get_recording_duration(self) -> float:
        """获取当前录制时长（秒）"""
        if not self.frames:
            return 0.0
        return len(self.frames) * self.chunk_size / self.sample_rate


# 全局录制器实例
_recorder_instance = None


def get_recorder() -> AudioRecorder:
    """获取录制器实例（单例）"""
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = AudioRecorder()
    return _recorder_instance


if __name__ == "__main__":
    # 测试录制
    print("🎤 音频录制测试")
    
    recorder = AudioRecorder()
    
    if recorder.start_recording():
        print("录制5秒...")
        time.sleep(5)
        
        audio_path = recorder.stop_recording()
        if audio_path:
            print(f"✅ 录制完成: {audio_path}")
        else:
            print("❌ 录制失败")
    else:
        print("❌ 无法启动录制")









