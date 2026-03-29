"""
声纹验证模块 - 使用 SpeechBrain ECAPA-TDNN 实现 Personal VAD
相比 Resemblyzer (GE2E)，ECAPA-TDNN 在噪声、说话内容变化等环境下更稳定
支持多样本注册提高准确率
支持声纹持久化保存/加载
"""
import numpy as np
import os
import json
from typing import Optional, Tuple, List, Dict
from pathlib import Path
try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

# 声纹数据保存目录
SPEAKER_DATA_DIR = Path("data/speakers")


class SpeakerVerifier:
    def __init__(self, threshold: float = 0.25, device: str = None):
        """
        初始化声纹验证器
        
        Args:
            threshold: 相似度阈值，ECAPA-TDNN 使用余弦相似度，推荐 0.25-0.35
            device: 设备，默认自动选择
        """
        self.threshold = threshold
        self.target_embedding = None
        self.embeddings: List[np.ndarray] = []  # 存储多个样本的embedding
        self.is_enrolled = False
        self.min_samples = 8  # 需要至少8个样本才算注册完成
        self.speaker_name = None  # 说话人名称
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[SpeakerVerifier] 🎤 初始化 ECAPA-TDNN 声纹模型 ({self.device})...")
        
        # 直接从本地加载 ECAPA-TDNN 模型（不使用 HuggingFace API）
        self._load_ecapa_model()
        print("[SpeakerVerifier] ✅ ECAPA-TDNN 声纹模型加载完成")
    
    def _load_ecapa_model(self):
        """手动加载本地 ECAPA-TDNN 模型"""
        import speechbrain
        from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
        from speechbrain.lobes.features import Fbank
        from speechbrain.processing.features import InputNormalization
        
        # 尝试多个可能的模型路径
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'spkrec-ecapa-voxceleb'),
            "/root/autodl-tmp/models/spkrec-ecapa-voxceleb",
        ]
        
        model_dir = None
        for path in possible_paths:
            path = os.path.normpath(path)
            if os.path.exists(path) and os.path.exists(os.path.join(path, "embedding_model.ckpt")):
                model_dir = path
                break
        
        if model_dir is None:
            # 如果本地没有模型，使用 SpeechBrain 的预训练模型接口
            print("[SpeakerVerifier] 本地模型未找到，使用 SpeechBrain 预训练模型...")
            from speechbrain.inference.speaker import EncoderClassifier
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'spkrec-ecapa-voxceleb'),
                run_opts={"device": str(self.device)}
            )
            self.use_pretrained = True
            SPEAKER_DATA_DIR.mkdir(parents=True, exist_ok=True)
            return
        
        self.use_pretrained = False
        print(f"[SpeakerVerifier] 使用本地模型: {model_dir}")
        
        # 创建特征提取器
        self.compute_features = Fbank(n_mels=80)
        
        # 输入特征归一化（sentence 级别，不需要预训练参数）
        self.mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)
        
        # Embedding 归一化（global 级别，需要加载预训练参数）
        self.mean_var_norm_emb = InputNormalization(norm_type="global", std_norm=False)
        mean_var_emb_path = os.path.join(model_dir, "mean_var_norm_emb.ckpt")
        if os.path.exists(mean_var_emb_path):
            # 手动加载 checkpoint（非标准 state_dict 格式）
            ckpt = torch.load(mean_var_emb_path, map_location=self.device)
            self.mean_var_norm_emb.count = ckpt.get("count", 0)
            self.mean_var_norm_emb.glob_mean = ckpt.get("glob_mean")
            self.mean_var_norm_emb.glob_std = ckpt.get("glob_std")
        
        # 创建 ECAPA-TDNN 模型
        self.embedding_model = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=192
        )
        
        # 加载预训练权重
        embedding_path = os.path.join(model_dir, "embedding_model.ckpt")
        self.embedding_model.load_state_dict(torch.load(embedding_path, map_location=self.device))
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        
        # 确保数据目录存在
        SPEAKER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def _extract_embedding(self, audio_path: str) -> np.ndarray:
        """从音频文件提取 ECAPA-TDNN embedding"""
        # 加载音频
        signal, sr = torchaudio.load(audio_path)
        
        # 重采样到 16kHz（ECAPA-TDNN 要求）
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            signal = resampler(signal)
        
        # 转为单声道
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        
        # 确保是 [batch, time] 格式
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
        
        signal = signal.to(self.device)
        
        # 提取 embedding
        with torch.no_grad():
            # 计算 Fbank 特征
            feats = self.compute_features(signal)
            # 归一化
            feats = self.mean_var_norm(feats, torch.tensor([1.0]).to(self.device))
            # 提取 embedding
            embedding = self.embedding_model(feats)
        
        return embedding.squeeze().cpu().numpy()
    
    def add_sample(self, audio_path: str) -> Tuple[bool, int, int]:
        """
        添加一个声纹样本
        Returns: (是否成功, 当前样本数, 需要样本数)
        """
        try:
            embedding = self._extract_embedding(audio_path)
            self.embeddings.append(embedding)
            
            current = len(self.embeddings)
            needed = self.min_samples
            
            print(f"[SpeakerVerifier] 📝 添加样本 {current}/{needed}")
            
            # 达到最小样本数，计算平均embedding
            if current >= needed:
                self.target_embedding = np.mean(self.embeddings, axis=0)
                self.is_enrolled = True
                print(f"[SpeakerVerifier] ✅ 声纹注册完成 ({current}个样本)")
            
            return True, current, needed
        except Exception as e:
            print(f"[SpeakerVerifier] ❌ 添加样本失败: {e}")
            import traceback
            traceback.print_exc()
            return False, len(self.embeddings), self.min_samples
    
    def enroll(self, audio_path: str) -> bool:
        """兼容旧接口：单样本注册"""
        success, current, needed = self.add_sample(audio_path)
        return success and self.is_enrolled
    
    def verify(self, audio_path: str) -> Tuple[bool, float]:
        """
        验证音频是否属于已注册的说话人
        
        Returns:
            (是否匹配, 相似度分数)
        """
        if not self.is_enrolled:
            return True, 1.0
        try:
            test_emb = self._extract_embedding(audio_path)
            
            # 计算余弦相似度
            sim = float(np.dot(self.target_embedding, test_emb) / 
                       (np.linalg.norm(self.target_embedding) * np.linalg.norm(test_emb)))
            
            is_target = sim >= self.threshold
            print(f"[SpeakerVerifier] {'✅' if is_target else '❌'} 相似度: {sim:.3f} (阈值: {self.threshold})")
            return is_target, sim
        except Exception as e:
            print(f"[SpeakerVerifier] ⚠️ 验证出错: {e}")
            return True, 1.0
    
    def verify_from_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float]:
        if not self.is_enrolled:
            return True, 1.0
        import tempfile, soundfile as sf
        try:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(tf.name, audio_data, sample_rate)
            result = self.verify(tf.name)
            os.unlink(tf.name)
            return result
        except:
            return True, 1.0
    
    def get_status(self) -> dict:
        """获取注册状态"""
        return {
            "is_enrolled": self.is_enrolled,
            "current_samples": len(self.embeddings),
            "min_samples": self.min_samples,
            "threshold": self.threshold
        }
    
    def reset(self):
        self.target_embedding = None
        self.embeddings = []
        self.is_enrolled = False
        self.speaker_name = None
        print("[SpeakerVerifier] 🔄 声纹已重置")
    
    def save(self, speaker_name: str) -> bool:
        """
        保存声纹数据到文件
        
        Args:
            speaker_name: 说话人名称
            
        Returns:
            是否保存成功
        """
        if not self.is_enrolled or self.target_embedding is None:
            print("[SpeakerVerifier] ⚠️ 未注册声纹，无法保存")
            return False
        
        try:
            self.speaker_name = speaker_name
            # 文件名使用说话人名称
            safe_name = speaker_name.replace("/", "_").replace("\\", "_")
            file_path = SPEAKER_DATA_DIR / f"{safe_name}.npz"
            
            # 保存embedding和元数据
            np.savez(
                file_path,
                target_embedding=self.target_embedding,
                embeddings=np.array(self.embeddings),
                threshold=self.threshold,
                speaker_name=speaker_name
            )
            
            print(f"[SpeakerVerifier] 💾 声纹已保存: {file_path}")
            return True
        except Exception as e:
            print(f"[SpeakerVerifier] ❌ 保存失败: {e}")
            return False
    
    def load(self, speaker_name: str) -> bool:
        """
        从文件加载声纹数据
        
        Args:
            speaker_name: 说话人名称
            
        Returns:
            是否加载成功
        """
        try:
            safe_name = speaker_name.replace("/", "_").replace("\\", "_")
            file_path = SPEAKER_DATA_DIR / f"{safe_name}.npz"
            
            if not file_path.exists():
                print(f"[SpeakerVerifier] ⚠️ 声纹文件不存在: {file_path}")
                return False
            
            data = np.load(file_path, allow_pickle=True)
            self.target_embedding = data['target_embedding']
            self.embeddings = list(data['embeddings'])
            # 注意：不覆盖当前阈值，让用户可以动态调节
            # self.threshold = float(data['threshold'])
            self.speaker_name = str(data['speaker_name'])
            self.is_enrolled = True
            
            print(f"[SpeakerVerifier] ✅ 已加载声纹: {self.speaker_name} ({len(self.embeddings)}个样本, 阈值: {self.threshold})")
            return True
        except Exception as e:
            print(f"[SpeakerVerifier] ❌ 加载失败: {e}")
            return False
    
    @staticmethod
    def list_saved_speakers() -> List[str]:
        """获取所有已保存的说话人列表"""
        SPEAKER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        speakers = []
        for f in SPEAKER_DATA_DIR.glob("*.npz"):
            try:
                data = np.load(f, allow_pickle=True)
                name = str(data['speaker_name'])
                speakers.append(name)
            except:
                pass
        return speakers


_verifier = None

def get_speaker_verifier(threshold: float = 0.25) -> SpeakerVerifier:
    """
    获取声纹验证器单例
    
    Args:
        threshold: 相似度阈值，ECAPA-TDNN 推荐 0.25-0.35
    """
    global _verifier
    if _verifier is None:
        _verifier = SpeakerVerifier(threshold=threshold)
    return _verifier
