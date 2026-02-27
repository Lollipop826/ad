"""
自动语音输入组件 - 持续监听版本
使用 JavaScript 实现浏览器端 VAD + 录音
"""
import streamlit as st
import streamlit.components.v1 as components

def auto_voice_input(key="auto_voice"):
    """
    自动语音输入组件
    
    Returns:
        audio_data: 录音的音频数据（base64编码的WAV）
    """
    
    # HTML + JavaScript 实现自动监听
    component_html = """
    <div style="text-align: center; padding: 20px;">
        <div id="status" style="font-size: 18px; margin-bottom: 20px;">
            <span id="status-text">🎤 准备监听...</span>
        </div>
        
        <button id="startBtn" onclick="startListening()" 
                style="background: #52c41a; color: white; border: none; 
                       padding: 15px 30px; border-radius: 25px; 
                       font-size: 16px; cursor: pointer; margin: 10px;">
            开始监听
        </button>
        
        <button id="stopBtn" onclick="stopListening()" disabled
                style="background: #f5222d; color: white; border: none; 
                       padding: 15px 30px; border-radius: 25px; 
                       font-size: 16px; cursor: pointer; margin: 10px;">
            停止监听
        </button>
        
        <div id="volume-meter" style="margin-top: 20px;">
            <div style="background: #f0f0f0; height: 20px; border-radius: 10px; overflow: hidden;">
                <div id="volume-bar" style="background: linear-gradient(90deg, #52c41a, #faad14); 
                     height: 100%; width: 0%; transition: width 0.1s;"></div>
            </div>
        </div>
    </div>
    
    <script>
    let mediaRecorder;
    let audioChunks = [];
    let isListening = false;
    let silenceTimeout;
    let audioContext;
    let analyser;
    let volumeCheckInterval;
    
    const SILENCE_THRESHOLD = 0.01;  // 音量阈值
    const SILENCE_DURATION = 2000;   // 2秒静音
    const MIN_RECORDING_TIME = 500;  // 最短录音时间
    let recordingStartTime = 0;
    
    async function startListening() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // 设置音频分析器
            audioContext = new AudioContext();
            analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);
            analyser.fftSize = 256;
            
            // 开始录音
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64Audio = reader.result.split(',')[1];
                    // 发送到 Streamlit
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        data: base64Audio
                    }, '*');
                };
                reader.readAsDataURL(audioBlob);
                
                // 清理
                stream.getTracks().forEach(track => track.stop());
                if (audioContext) audioContext.close();
            };
            
            mediaRecorder.start();
            recordingStartTime = Date.now();
            isListening = true;
            
            document.getElementById('status-text').textContent = '🎤 监听中...请说话';
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            
            // 实时检测音量
            checkVolume();
            
        } catch (error) {
            document.getElementById('status-text').textContent = '❌ 麦克风权限被拒绝';
            console.error('Error:', error);
        }
    }
    
    function checkVolume() {
        if (!isListening) return;
        
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);
        
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
        const volume = average / 255;
        
        // 更新音量条
        document.getElementById('volume-bar').style.width = (volume * 100) + '%';
        
        // 检测说话/静音
        if (volume > SILENCE_THRESHOLD) {
            // 正在说话
            document.getElementById('status-text').textContent = '🗣️ 检测到声音...';
            clearTimeout(silenceTimeout);
        } else {
            // 静音
            if (!silenceTimeout && Date.now() - recordingStartTime > MIN_RECORDING_TIME) {
                silenceTimeout = setTimeout(() => {
                    if (audioChunks.length > 0) {
                        stopRecording();
                    }
                }, SILENCE_DURATION);
            }
        }
        
        requestAnimationFrame(checkVolume);
    }
    
    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            isListening = false;
            mediaRecorder.stop();
            document.getElementById('status-text').textContent = '⏹️ 录音结束，处理中...';
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
    }
    
    function stopListening() {
        stopRecording();
    }
    </script>
    """
    
    # 渲染组件
    audio_data = components.html(component_html, height=200)
    
    return audio_data
