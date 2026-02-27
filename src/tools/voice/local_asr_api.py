"""
本地ASR API接口
提供HTTP接口供前端调用Whisper进行语音识别
"""

import whisper
import tempfile
import os
import time
import base64
from flask import Flask, request, jsonify
import threading
import queue
import json
import subprocess


class LocalASRServer:
    """本地ASR服务器"""
    
    def __init__(self, model_size: str = "base", port: int = 5001):
        self.model_size = model_size
        self.port = port
        self.model = None
        self.app = Flask(__name__)
        self._setup_routes()
        self._load_model()
    
    def _load_model(self):
        """加载Whisper模型"""
        print(f"[ASR] 正在加载Whisper模型: {self.model_size}")
        start_time = time.time()
        
        # 设置ffmpeg路径
        ffmpeg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'ffmpeg')
        if os.path.exists(ffmpeg_path):
            os.environ['FFMPEG_BINARY'] = ffmpeg_path
            print(f"[ASR] 设置ffmpeg路径: {ffmpeg_path}")
            # 同时设置whisper的ffmpeg路径
            import whisper
            whisper.ffmpeg_path = ffmpeg_path
            # 设置PATH环境变量
            current_path = os.environ.get('PATH', '')
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            os.environ['PATH'] = f"{project_dir}:{current_path}"
            print(f"[ASR] 已更新PATH: {project_dir}")
        else:
            print(f"[ASR] ⚠️ 未找到ffmpeg，使用系统默认")
        
        try:
            self.model = whisper.load_model(self.model_size)
            load_time = time.time() - start_time
            print(f"[ASR] ✅ 模型加载完成，耗时: {load_time:.2f}秒")
        except Exception as e:
            print(f"[ASR] ❌ 模型加载失败: {e}")
            raise
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.route('/transcribe', methods=['POST'])
        def transcribe():
            """语音识别接口"""
            try:
                data = request.get_json()
                audio_base64 = data.get('audio')
                language = data.get('language', 'zh')
                
                if not audio_base64:
                    return jsonify({'error': 'No audio data'}), 400
                
                # 解码音频
                audio_data = base64.b64decode(audio_base64)
                
                # 保存临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_path = temp_file.name
                
                try:
                    # Whisper识别
                    start_time = time.time()
                    
                    # 确保ffmpeg在PATH中
                    ffmpeg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'ffmpeg')
                    if os.path.exists(ffmpeg_path):
                        os.environ['PATH'] = f"{os.path.dirname(ffmpeg_path)}:{os.environ.get('PATH', '')}"
                    
                    result = self.model.transcribe(
                        temp_path,
                        language=language,
                        fp16=False,
                        verbose=False
                    )
                    
                    text = result["text"].strip()
                    transcribe_time = time.time() - start_time
                    
                    print(f"[ASR] ✅ 识别完成: '{text}' (耗时: {transcribe_time:.2f}秒)")
                    
                    return jsonify({
                        'text': text,
                        'confidence': 0.95,  # Whisper不直接提供置信度
                        'duration': transcribe_time
                    })
                    
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e:
                print(f"[ASR] ❌ 识别失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """健康检查"""
            return jsonify({'status': 'ok', 'model': self.model_size})
    
    def start(self):
        """启动服务器"""
        print(f"[ASR] 🚀 启动本地ASR服务器，端口: {self.port}")
        self.app.run(host='127.0.0.1', port=self.port, debug=False, threaded=True)


# 全局服务器实例
_asr_server = None


def start_asr_server(model_size: str = "base", port: int = 5001):
    """启动ASR服务器"""
    global _asr_server
    if _asr_server is None:
        _asr_server = LocalASRServer(model_size=model_size, port=port)
    
    # 在后台线程中启动
    server_thread = threading.Thread(target=_asr_server.start, daemon=True)
    server_thread.start()
    
    # 等待服务器启动
    time.sleep(2)
    print(f"[ASR] ✅ 本地ASR服务器已启动: http://127.0.0.1:{port}")


if __name__ == "__main__":
    # 测试启动
    start_asr_server(model_size="base", port=5001)
    
    # 保持运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[ASR] 服务器已停止")
