"""
简单的浏览器端 VAD + 录音组件
无需 WebRTC，直接用 MediaRecorder API
"""
import streamlit as st
import streamlit.components.v1 as components


def simple_vad_recorder(key="simple_vad"):
    """
    简单的 VAD 录音组件
    
    Returns:
        audio_bytes: 录音数据（WAV格式的 base64）
    """
    
    component_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: sans-serif;
                text-align: center;
                padding: 20px;
                background: #f0f2f6;
            }}
            #status {{
                font-size: 20px;
                margin: 20px;
                padding: 15px;
                border-radius: 10px;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .recording {{
                background: #ffebee !important;
                color: #c62828;
            }}
            .listening {{
                background: #e8f5e9 !important;
                color: #2e7d32;
            }}
            #volumeMeter {{
                width: 80%;
                height: 30px;
                margin: 20px auto;
                background: #e0e0e0;
                border-radius: 15px;
                overflow: hidden;
            }}
            #volumeBar {{
                height: 100%;
                width: 0%;
                background: linear-gradient(90deg, #4caf50, #ff9800, #f44336);
                transition: width 0.1s;
            }}
            .button {{
                padding: 12px 30px;
                font-size: 16px;
                margin: 10px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
            }}
            .start-btn {{
                background: #4caf50;
                color: white;
            }}
            .stop-btn {{
                background: #f44336;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div id="status">准备就绪，点击开始</div>
        <div id="volumeMeter">
            <div id="volumeBar"></div>
        </div>
        <button id="startBtn" class="button start-btn" onclick="startListening()">🎤 开始监听</button>
        <button id="stopBtn" class="button stop-btn" onclick="stopListening()" disabled>⏹️ 停止</button>
        
        <script>
        let mediaRecorder;
        let audioChunks = [];
        let audioContext;
        let analyser;
        let isRecording = false;
        let isSpeaking = false;
        let silenceTimeout;
        let animationId;
        
        const ENERGY_THRESHOLD = 0.02;  // 能量阈值（0-1）
        const SILENCE_DURATION = 2000;  // 静音持续时间（毫秒）
        const MIN_RECORDING_DURATION = 500;  // 最短录音时间
        let recordingStartTime = 0;
        
        async function startListening() {{
            try {{
                const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                
                // 设置音频分析器（用于 VAD）
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                const source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);
                analyser.fftSize = 2048;
                
                // 设置录音器
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = (event) => {{
                    if (event.data.size > 0) {{
                        audioChunks.push(event.data);
                    }}
                }};
                
                mediaRecorder.onstop = () => {{
                    const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                    const reader = new FileReader();
                    reader.onloadend = () => {{
                        const base64Audio = reader.result.split(',')[1];
                        console.log('[VAD] 音频已录制，大小:', audioBlob.size, 'bytes');
                        
                        // 发送给 Streamlit - 使用正确的格式
                        if (window.parent && window.parent.Streamlit) {{
                            window.parent.Streamlit.setComponentValue(base64Audio);
                        }} else {{
                            // 回退方案：使用 postMessage
                            window.parent.postMessage({{
                                type: 'streamlit:setComponentValue',
                                value: base64Audio
                            }}, '*');
                        }}
                        
                        console.log('[VAD] 音频已发送');
                        
                        // 录音完成后触发页面刷新，让Streamlit获取数据
                        setTimeout(() => {{
                            window.location.reload();
                        }}, 100);
                    }};
                    reader.readAsDataURL(audioBlob);
                    audioChunks = [];
                }};
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                updateStatus('🎧 正在监听...', 'listening');
                
                // 开始 VAD 检测
                detectVoiceActivity();
                
            }} catch (error) {{
                console.error('[VAD] 错误:', error);
                updateStatus('❌ 麦克风权限被拒绝', '');
            }}
        }}
        
        function detectVoiceActivity() {{
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            
            function check() {{
                analyser.getByteFrequencyData(dataArray);
                
                // 计算平均能量
                const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
                const normalizedEnergy = average / 255;
                
                // 更新音量条
                document.getElementById('volumeBar').style.width = (normalizedEnergy * 100) + '%';
                
                // VAD 判断
                if (normalizedEnergy > ENERGY_THRESHOLD) {{
                    // 检测到声音
                    if (!isSpeaking) {{
                        console.log('[VAD] 🎤 检测到声音');
                        isSpeaking = true;
                        startRecording();
                        updateStatus('🔴 正在录音...', 'recording');
                    }}
                    
                    // 清除静音计时器
                    clearTimeout(silenceTimeout);
                    
                }} else {{
                    // 静音
                    if (isSpeaking) {{
                        // 设置静音计时器
                        clearTimeout(silenceTimeout);
                        silenceTimeout = setTimeout(() => {{
                            const duration = Date.now() - recordingStartTime;
                            if (duration >= MIN_RECORDING_DURATION) {{
                                console.log('[VAD] ⏹️ 检测到静音，停止录音');
                                isSpeaking = false;
                                stopRecording();
                                updateStatus('🎧 正在监听...', 'listening');
                            }} else {{
                                console.log('[VAD] ⚠️ 录音太短，继续监听');
                                isSpeaking = false;
                                mediaRecorder.stop();
                            }}
                        }}, SILENCE_DURATION);
                    }}
                }}
                
                animationId = requestAnimationFrame(check);
            }}
            
            check();
        }}
        
        function startRecording() {{
            if (mediaRecorder && mediaRecorder.state === 'inactive') {{
                audioChunks = [];
                mediaRecorder.start();
                recordingStartTime = Date.now();
                console.log('[VAD] 开始录音');
            }}
        }}
        
        function stopRecording() {{
            if (mediaRecorder && mediaRecorder.state === 'recording') {{
                mediaRecorder.stop();
                console.log('[VAD] 停止录音');
            }}
        }}
        
        function stopListening() {{
            cancelAnimationFrame(animationId);
            clearTimeout(silenceTimeout);
            
            if (mediaRecorder && mediaRecorder.stream) {{
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }}
            if (audioContext) {{
                audioContext.close();
            }}
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            updateStatus('已停止', '');
            document.getElementById('volumeBar').style.width = '0%';
        }}
        
        function updateStatus(text, className) {{
            const statusEl = document.getElementById('status');
            statusEl.textContent = text;
            statusEl.className = className;
        }}
        </script>
    </body>
    </html>
    """
    
    # 渲染组件，返回音频数据
    audio_base64 = components.html(component_html, height=250)
    
    return audio_base64
