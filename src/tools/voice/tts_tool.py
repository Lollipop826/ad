"""
高质量语音合成工具 (Edge TTS)
使用Microsoft Edge TTS，云端高质量合成，完美支持中文
"""

import asyncio
import tempfile
import os
import numpy as np
from pathlib import Path
import soundfile as sf
import hashlib
import time
import edge_tts


class VoiceTTS:
    """Edge TTS语音合成（Microsoft云端高质量合成，完美支持中文）"""
    
    # Edge TTS支持的声音（中文）
    VOICES = {
        "xiaoxiao": "zh-CN-XiaoxiaoNeural",     # 温柔中文女声
        "yunxi": "zh-CN-YunxiNeural",           # 平和中文男声  
        "xiaoyi": "zh-CN-XiaoyiNeural",         # 温暖中文女声
        "yunjian": "zh-CN-YunjianNeural",       # 稳重中文男声
    }
    
    def __init__(self, voice: str = "xiaoxiao", rate: str = "+0%", volume: str = "+0%"):
        """
        初始化TTS
        
        Args:
            voice: 音色名称，默认"xiaoxiao"（温柔中文女声）
            rate: 语速，Edge TTS格式如 "+0%", "-10%", "+20%"
            volume: 音量，Edge TTS格式如 "+0%", "-10%", "+20%"
        """
        self.voice = voice
        self.voice_name = self.VOICES.get(voice, self.VOICES["xiaoxiao"])
        self.rate = rate  # Edge TTS直接使用字符串格式
        self.volume = volume  # Edge TTS直接使用字符串格式
        
    def _get_cache_path(self, text: str) -> str:
        """基于文本和参数生成稳定的缓存文件路径"""
        version_tag = "edge_tts_v1"
        key = f"{version_tag}|{self.voice_name}|{self.rate}|{self.volume}|{text}".encode("utf-8")
        digest = hashlib.sha1(key).hexdigest()
        # 使用外部存储避免系统盘满
        temp_dir = Path(os.path.dirname(__file__)).parent.parent.parent / "tmp" / "ad_screening_voice"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return str(temp_dir / f"tts_{digest}.wav")

    @staticmethod
    def _is_valid_audio(path: str) -> bool:
        """快速校验音频文件是否有效（存在、非空、时长>0.1s）"""
        try:
            if not os.path.exists(path):
                return False
            if os.path.getsize(path) < 1024:  # 小于1KB基本无效
                return False
            info = sf.info(path)
            duration_s = info.frames / float(info.samplerate)
            return duration_s > 0.1
        except Exception:
            return False

    @staticmethod
    def _print_audio_stats(tag: str, path: str):
        try:
            size = os.path.getsize(path) if os.path.exists(path) else -1
            info = sf.info(path)
            duration_s = info.frames / float(info.samplerate)
            amp = 0.0
            try:
                data, sr = sf.read(path)
                import numpy as _np
                amp = float(_np.max(_np.abs(data))) if len(data) else 0.0
            except Exception:
                pass
            print(f"[TTS][{tag}] file={path} size={size}B sr={info.samplerate}Hz frames={info.frames} dur={duration_s:.3f}s max_amp={amp:.3f}")
        except Exception as e:
            print(f"[TTS][{tag}] stat error for {path}: {e}")
        
    async def text_to_speech_async(self, text: str, output_path: str = None) -> str:
        """
        异步将文本转为语音
        
        Args:
            text: 要转换的文本
            output_path: 输出音频文件路径，默认为None（自动创建临时文件）
            
        Returns:
            str: 音频文件路径
        """
        if not output_path:
            output_path = self._get_cache_path(text)
        
        # 缓存命中
        if self._is_valid_audio(output_path):
            print(f"[TTS] cache hit -> {output_path}")
            self._print_audio_stats("cache", output_path)
            return output_path
        
        try:
            # 使用Edge TTS生成语音
            temp_out = output_path + ".tmp.mp3"  # Edge TTS生成mp3
            result_path = await self._edge_tts_async(text, temp_out)
            
            # 转换为WAV格式
            if result_path and os.path.exists(temp_out):
                # 读取mp3并转换为wav
                import subprocess
                try:
                    # 使用ffmpeg转换（如果有）
                    subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-ar', '24000', output_path],
                                 check=True, capture_output=True)
                    os.unlink(temp_out)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # ffmpeg不可用，直接使用soundfile读写
                    try:
                        import pydub
                        audio = pydub.AudioSegment.from_mp3(temp_out)
                        audio = audio.set_frame_rate(24000)
                        audio.export(output_path, format='wav')
                        os.unlink(temp_out)
                    except:
                        # 都不可用，直接重命名
                        os.replace(temp_out, output_path)
                
                self._print_audio_stats("write_final", output_path)
                return output_path
            return result_path
                
        except Exception as e:
            print(f"[TTS] ❌ Edge TTS生成失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def _edge_tts_async(self, text: str, output_path: str) -> str:
        """使用Edge TTS生成语音"""
        try:
            print(f"[TTS] 正在使用Edge TTS生成语音...")
            print(f"[TTS] 文本: {text[:50]}{'...' if len(text) > 50 else ''}")
            print(f"[TTS] 音色: {self.voice_name}, 语速: {self.rate}, 音量: {self.volume}")
            
            t0 = time.time()
            
            # 创建Edge TTS通信对象
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice_name,
                rate=self.rate,
                volume=self.volume
            )
            
            # 生成并保存音频
            await communicate.save(output_path)
            
            print(f"[TTS] Edge TTS生成完成，耗时: {(time.time()-t0)*1000:.1f}ms")
            
            # 验证文件
            if not os.path.exists(output_path):
                raise ValueError("Edge TTS未生成音频文件")
            
            file_size = os.path.getsize(output_path)
            if file_size < 1024:
                raise ValueError(f"生成的音频文件过小: {file_size}B")
            
            print(f"[TTS] ✅ Edge TTS高质量语音生成成功: {output_path} ({file_size}B)")
            return output_path
            
        except Exception as e:
            print(f"[TTS] ❌ Edge TTS生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def text_to_speech(self, text: str, output_path: str = None) -> str:
        """
        同步接口：将文本转为语音
        
        Args:
            text: 要转换的文本
            output_path: 输出音频文件路径
            
        Returns:
            str: 音频文件路径
        """
        try:
            return asyncio.run(self.text_to_speech_async(text, output_path))
        except Exception as e:
            print(f"[TTS] ❌ 同步TTS生成失败: {e}")
            return None


# 全局TTS实例（单例模式，提高效率）
_tts_instance = None


def get_tts(voice: str = "xiaoxiao", rate: str = "+0%", volume: str = "100%") -> VoiceTTS:
    """获取TTS实例（单例）"""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = VoiceTTS(voice=voice, rate=rate, volume=volume)
    return _tts_instance


async def text_to_speech(text: str, voice: str = "xiaoxiao") -> str:
    """
    快捷函数：文本转语音
    
    Args:
        text: 要转换的文本
        voice: 音色
        
    Returns:
        str: 音频文件路径
    """
    tts = get_tts(voice=voice)
    return await tts.text_to_speech_async(text)


if __name__ == "__main__":
    # 测试
    print("[TTS] 测试Edge TTS（Microsoft云端高质量合成）...")
    tts = VoiceTTS(voice="xiaoxiao", rate="-10%")
    audio_path = tts.text_to_speech("您好！咱们聊聊天，顺便做个简单的评估。您知道今天是几号吗？")
    if audio_path:
        print(f"✅ Edge TTS高质量语音生成成功: {audio_path}")
        print(f"📂 可以播放此文件测试效果")
        # 显示音频统计信息
        VoiceTTS._print_audio_stats("test", audio_path)
    else:
        print("❌ 语音生成失败")

