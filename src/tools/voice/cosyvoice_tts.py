"""
高质量语音合成工具 (PaddleSpeech)
使用百度PaddleSpeech，支持中文TTS
使用 fastspeech2_cnndecoder_csmsc 实现流式推理
"""

import asyncio
import os
import sys
import numpy as np
from pathlib import Path
import hashlib
import time
import paddle
import yaml
from yacs.config import CfgNode

class CosyVoiceTTS:
    """PaddleSpeech 语音合成（保持类名兼容性）"""
    
    # PaddleSpeech 支持的发音人（am=fastspeech2）
    SPEAKERS = {
        "neutral": "aistudio_zh",  # 标准女声
        "gentle": "aistudio_zh",   # 温柔（用同一个）
        "professional": "aistudio_zh",  # 专业
        "happy": "aistudio_zh",
        "sad": "aistudio_zh",
        "angry": "aistudio_zh",
        "fearful": "aistudio_zh"
    }
    
    def __init__(self, model_dir: str = None, device: str = None):
        """初始化 PaddleSpeech TTS
        
        Args:
            model_dir: 不使用（保留兼容性）
            device: GPU设备，例如 'gpu:0' 或 'cpu'
        """
        # 添加 PaddleSpeech 源码路径
        ps_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'PaddleSpeech')
        if ps_path not in sys.path:
            sys.path.insert(0, ps_path)
        
        # 使用 GPU（已安装 CUDA 11.8）
        self.device = device or ('gpu:0' if paddle.is_compiled_with_cuda() else 'cpu')
        if 'gpu' in self.device:
            paddle.set_device(self.device.replace('gpu:', 'gpu:'))
        else:
            paddle.set_device('cpu')
        
        self.am = None  # 声学模型 (Inference 包装)
        self.voc = None  # 声码器 (Inference 包装)
        self.frontend = None  # 前端文本处理
        
        self._load_model()
        
    def _load_model(self):
        """加载 PaddleSpeech 模型"""
        try:
            print(f"[PaddleSpeech] 🚀 初始化流式 TTS 模型...")
            
            from paddlespeech.t2s.models.fastspeech2 import FastSpeech2, FastSpeech2Inference
            from paddlespeech.t2s.models.melgan import MelGANGenerator, MelGANInference
            from paddlespeech.t2s.modules.normalizer import ZScore
            from paddlespeech.t2s.frontend.zh_frontend import Frontend
            from paddlespeech.resource import CommonTaskResource
            
            # 初始化资源管理器
            task_resource = CommonTaskResource(task='tts', model_format='dynamic')
            
            # 1. 加载声学模型 (AM)
            print(f"[PaddleSpeech] 📥 加载声学模型 fastspeech2_cnndecoder_csmsc...")
            am_tag = 'fastspeech2_cnndecoder_csmsc-zh'
            task_resource.set_task_model(
                model_tag=am_tag,
                model_type=0,  # am
                skip_download=False,
                version=None
            )
            
            am_res_path = task_resource.res_dir
            am_ckpt = os.path.join(am_res_path, task_resource.res_dict['ckpt'])
            am_config = os.path.join(am_res_path, task_resource.res_dict['config'])
            am_stat = os.path.join(am_res_path, task_resource.res_dict['speech_stats'])
            phones_dict = os.path.join(am_res_path, task_resource.res_dict['phones_dict'])
            
            # 加载配置
            with open(am_config) as f:
                am_config = CfgNode(yaml.safe_load(f))
            
            # 创建模型
            with open(phones_dict, 'r') as f:
                phn_id = [line.strip().split() for line in f.readlines()]
            vocab_size = len(phn_id)
            
            odim = am_config.n_mels
            am_model = FastSpeech2(
                idim=vocab_size,
                odim=odim,
                **am_config["model"]
            )
            am_model.set_state_dict(paddle.load(am_ckpt)["main_params"])
            am_model.eval()
            
            # 加载统计信息并创建 normalizer
            am_mu, am_std = np.load(am_stat)
            am_mu = paddle.to_tensor(am_mu)
            am_std = paddle.to_tensor(am_std)
            am_normalizer = ZScore(am_mu, am_std)
            
            # 使用 Inference 包装
            self.am = FastSpeech2Inference(am_normalizer, am_model)
            
            # 2. 加载声码器 (VOC)
            print(f"[PaddleSpeech] 📥 加载声码器 mb_melgan_csmsc...")
            voc_tag = 'mb_melgan_csmsc-zh'
            task_resource.set_task_model(
                model_tag=voc_tag,
                model_type=1,  # vocoder
                skip_download=False,
                version=None
            )
            
            voc_res_path = task_resource.voc_res_dir
            voc_ckpt = os.path.join(voc_res_path, task_resource.voc_res_dict['ckpt'])
            voc_config = os.path.join(voc_res_path, task_resource.voc_res_dict['config'])
            voc_stat = os.path.join(voc_res_path, task_resource.voc_res_dict['speech_stats'])
            
            with open(voc_config) as f:
                voc_config = CfgNode(yaml.safe_load(f))
            
            voc_model = MelGANGenerator(**voc_config["generator_params"])
            voc_model.set_state_dict(paddle.load(voc_ckpt)["generator_params"])
            voc_model.remove_weight_norm()
            voc_model.eval()
            
            # 加载统计信息并创建 normalizer
            voc_mu, voc_std = np.load(voc_stat)
            voc_mu = paddle.to_tensor(voc_mu)
            voc_std = paddle.to_tensor(voc_std)
            voc_normalizer = ZScore(voc_mu, voc_std)
            
            # 使用 Inference 包装
            self.voc = MelGANInference(voc_normalizer, voc_model)
            
            # 3. 加载前端
            print(f"[PaddleSpeech] 📥 加载文本前端处理器...")
            self.frontend = Frontend(phone_vocab_path=phones_dict, tone_vocab_path=None)
            
            print(f"[PaddleSpeech] ✅ 模型加载完成（设备: {self.device}）")
            
            # 预热模型
            self._warmup()
            
        except Exception as e:
            print(f"[PaddleSpeech] ❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _warmup(self):
        """预热模型（首次推理会慢，预热后加速）"""
        try:
            print(f"[PaddleSpeech] 🔥 开始预热模型...")
            
            import tempfile
            
            warmup_text = "你好，这是预热测试。"
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # 执行一次推理
            self._generate_speech(warmup_text, tmp_path, 1.0)
            
            # 删除临时文件
            try:
                os.remove(tmp_path)
            except:
                pass
            
            print(f"[PaddleSpeech] ✅ 预热完成")
            print(f"[PaddleSpeech] 💡 后续推理将快 2-3 倍")
            
        except Exception as e:
            print(f"[PaddleSpeech] ⚠️ 预热失败（不影响使用）: {e}")
    
    def _get_cache_path(self, text: str, emotion: str = "neutral") -> str:
        """基于文本和情感生成缓存文件路径"""
        version_tag = "paddlespeech_v1"
        key = f"{version_tag}|{emotion}|{text}".encode("utf-8")
        digest = hashlib.sha1(key).hexdigest()
        temp_dir = Path(os.path.dirname(__file__)).parent.parent.parent / "tmp" / "ad_screening_voice"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return str(temp_dir / f"tts_{digest}.wav")
    
    async def text_to_speech_streaming(
        self,
        text: str,
        emotion: str = "neutral"
    ):
        """流式文本转语音（逐句生成并返回）
        
        Args:
            text: 要合成的文本
            emotion: 情感类型
            
        Yields:
            音频数据块（numpy数组）
        """
        print(f"[PaddleSpeech] 🎵 流式生成语音...")
        print(f"[PaddleSpeech] 文本: {text[:80]}...")
        
        try:
            # 按句子分割
            import re
            sentences = re.split(r'([。！？；;,.!?])', text)
            # 重新组合句子和标点
            merged = []
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences):
                    merged.append(sentences[i] + sentences[i+1])
                elif sentences[i].strip():
                    merged.append(sentences[i])
            
            # 逐句生成
            for idx, sentence in enumerate(merged):
                if not sentence.strip():
                    continue
                    
                print(f"[PaddleSpeech] 句子 {idx+1}/{len(merged)}: {sentence[:30]}...")
                
                # 生成临时文件
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                # 生成音频
                await asyncio.to_thread(
                    self._generate_speech,
                    sentence,
                    tmp_path,
                    1.0
                )
                
                # 读取并返回
                import soundfile as sf
                audio_data, sr = sf.read(tmp_path)
                
                # 删除临时文件
                try:
                    os.remove(tmp_path)
                except:
                    pass
                
                # 返回音频块
                yield audio_data.astype(np.float32)
                
        except Exception as e:
            print(f"[PaddleSpeech] ❌ 流式生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    async def text_to_speech_async(
        self, 
        text: str, 
        emotion: str = "neutral",
        speed: float = 1.0
    ) -> str:
        """异步文本转语音
        
        Args:
            text: 要合成的文本
            emotion: 情感类型
            speed: 语速
        
        Returns:
            生成的音频文件路径
        """
        # 检查缓存
        cache_path = self._get_cache_path(text, emotion)
        if os.path.exists(cache_path):
            print(f"[PaddleSpeech] 💾 使用缓存: {cache_path}")
            return cache_path
        
        start_time = time.time()
        print(f"[PaddleSpeech] 🎙️ 正在生成语音...")
        print(f"[PaddleSpeech] 文本: {text[:80]}...")
        
        try:
            # 使用asyncio.to_thread避免阻塞
            await asyncio.to_thread(
                self._generate_speech,
                text,
                cache_path,
                speed
            )
            
            elapsed = (time.time() - start_time) * 1000
            print(f"[PaddleSpeech] ✅ 生成完成，耗时: {elapsed:.1f}ms")
            
            return cache_path
            
        except Exception as e:
            print(f"[PaddleSpeech] ❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_speech(self, text: str, output_path: str, speed: float = 1.0):
        """生成语音（同步方法，由async方法调用）
        
        使用 PaddleSpeech 底层 API
        """
        # 1. 文本前端处理（merge_sentences=False 返回多个句子）
        input_ids = self.frontend.get_input_ids(text, merge_sentences=False)
        phone_ids = input_ids["phone_ids"]  # 列表，每个元素是一句的 phone_ids
        
        # 2. 逐句推理并拼接
        wav_list = []
        
        for part_phone_ids in phone_ids:
            # 转换为 Paddle Tensor（模型需要的格式）
            if isinstance(part_phone_ids, list):
                part_phone_ids = np.array(part_phone_ids, dtype=np.int64)
            if not isinstance(part_phone_ids, paddle.Tensor):
                part_phone_ids = paddle.to_tensor(part_phone_ids)
            
            # 声学模型推理（生成 mel 谱，Inference 会自动处理归一化）
            with paddle.no_grad():
                mel = self.am(part_phone_ids)  # Inference 的 __call__ 方法
            
            # 声码器推理（生成波形，Inference 会自动处理归一化）
            with paddle.no_grad():
                part_wav = self.voc(mel)  # Inference 的 __call__ 方法
            
            # 转换为 numpy 并确保形状为 1D
            part_wav_np = part_wav.numpy()
            if part_wav_np.ndim > 1:
                part_wav_np = part_wav_np.squeeze()  # 去掉所有长度为1的维度
            
            # 收集音频片段
            wav_list.append(part_wav_np)
        
        # 3. 拼接所有句子的音频
        if len(wav_list) > 0:
            # 确保所有片段都是 1D 数组再拼接
            wav_list_1d = [w.flatten() if w.ndim > 1 else w for w in wav_list]
            wav = np.concatenate(wav_list_1d, axis=0)
        else:
            wav = np.array([])
        
        # 4. 保存音频
        import soundfile as sf
        sf.write(output_path, wav, samplerate=24000)
        
        return wav
    
    def text_to_speech(self, text: str, emotion: str = "neutral", speed: float = 1.0) -> str:
        """
        同步文本转语音（兼容旧接口）
        
        Args:
            text: 要合成的文本
            emotion: 情感类型
            speed: 语速
        
        Returns:
            生成的音频文件路径
        """
        return asyncio.run(self.text_to_speech_async(text, emotion, speed))


# 兼容旧代码的别名
VoiceTTS = CosyVoiceTTS
