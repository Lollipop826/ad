"""
本地语音识别工具 (Whisper)
使用OpenAI Whisper进行本地语音识别，速度快且离线运行
"""

import whisper
import tempfile
import os
import time
from pathlib import Path
import numpy as np
import io
import base64


class LocalASR:
    """本地Whisper语音识别"""
    
    def __init__(self, model_size: str = "base"):
        """
        初始化本地ASR
        
        Args:
            model_size: 模型大小，可选：
            - "tiny": 最快，准确度较低 (~39MB)
            - "base": 平衡速度和准确度 (~74MB) ✅ 推荐
            - "small": 更准确，稍慢 (~244MB)
            - "medium": 高准确度，较慢 (~769MB)
            - "large": 最高准确度，最慢 (~1550MB)
        """
        self.model_size = model_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载Whisper模型"""
        print(f"[ASR] 正在加载Whisper模型: {self.model_size}")
        start_time = time.time()
        
        try:
            self.model = whisper.load_model(self.model_size)
            load_time = time.time() - start_time
            print(f"[ASR] ✅ 模型加载完成，耗时: {load_time:.2f}秒")
        except Exception as e:
            print(f"[ASR] ❌ 模型加载失败: {e}")
            raise
    
    def transcribe_audio_file(self, audio_path: str, language: str = "zh") -> str:
        """
        识别音频文件
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码，默认"zh"（中文）
            
        Returns:
            str: 识别结果文本
        """
        if not self.model:
            raise RuntimeError("模型未加载")
        
        start_time = time.time()
        
        try:
            # 使用Whisper进行识别
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False,  # 避免某些系统上的精度问题
                verbose=False
            )
            
            text = result["text"].strip()
            transcribe_time = time.time() - start_time
            
            print(f"[ASR] ✅ 识别完成: '{text}' (耗时: {transcribe_time:.2f}秒)")
            return text
            
        except Exception as e:
            print(f"[ASR] ❌ 识别失败: {e}")
            return ""
    
    def transcribe_audio_data(self, audio_data: bytes, language: str = "zh") -> str:
        """
        识别音频数据（字节）
        
        Args:
            audio_data: 音频字节数据
            language: 语言代码
            
        Returns:
            str: 识别结果文本
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name
        
        try:
            result = self.transcribe_audio_file(temp_path, language)
            return result
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def transcribe_base64_audio(self, base64_audio: str, language: str = "zh") -> str:
        """
        识别base64编码的音频
        
        Args:
            base64_audio: base64编码的音频数据
            language: 语言代码
            
        Returns:
            str: 识别结果文本
        """
        try:
            audio_data = base64.b64decode(base64_audio)
            return self.transcribe_audio_data(audio_data, language)
        except Exception as e:
            print(f"[ASR] ❌ Base64解码失败: {e}")
            return ""


# 全局ASR实例（单例模式）
_asr_instance = None


def get_asr(model_size: str = "base") -> LocalASR:
    """获取ASR实例（单例）"""
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = LocalASR(model_size=model_size)
    return _asr_instance


def transcribe_audio(audio_path: str, language: str = "zh", model_size: str = "base") -> str:
    """
    快捷函数：音频转文字
    
    Args:
        audio_path: 音频文件路径
        language: 语言代码
        model_size: 模型大小
        
    Returns:
        str: 识别结果
    """
    asr = get_asr(model_size=model_size)
    return asr.transcribe_audio_file(audio_path, language)


if __name__ == "__main__":
    # 测试
    print("🎤 本地ASR测试")
    
    # 创建测试音频（这里需要实际的音频文件）
    test_audio_path = "test_audio.wav"
    
    if os.path.exists(test_audio_path):
        asr = LocalASR(model_size="base")
        result = asr.transcribe_audio_file(test_audio_path)
        print(f"✅ 识别结果: {result}")
    else:
        print("⚠️ 请提供测试音频文件")









