"""
阿尔茨海默病初筛对话系统 - 流式输出版本
支持实时流式显示，大幅提升用户体验
"""

import streamlit as st
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置环境变量
os.environ['PATH'] = f"{os.path.expanduser('~/bin')}:{os.environ.get('PATH', '')}"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="AD认知评估系统（流式版）",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 使用流式Agent
from src.agents.screening_agent_streaming import ADScreeningAgentStreaming
from src.domain.dimensions import MMSE_DIMENSIONS

def get_agent():
    """创建并返回流式Agent实例"""
    agent = ADScreeningAgentStreaming()
    print(f"[INFO] Agent版本: Streaming（流式输出）")
    return agent

# ChatGPT风格界面CSS
st.markdown("""
<style>
    /* 隐藏Streamlit默认元素 */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* 对话消息容器 */
    .msg {
        display: flex;
        gap: 1.5rem;
        padding: 1.5rem 1rem;
        align-items: flex-start;
    }
    
    /* AI消息背景 */
    .msg:has(.avatar-ai) {
        background: #f7f7f8;
    }
    
    /* 头像样式 */
    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 2px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        font-weight: 600;
        color: white;
    }
    
    .avatar-user {
        background: #16a085;
    }
    
    .avatar-ai {
        background: #27ae60;
    }
    
    /* 消息内容 */
    .msg-content {
        flex: 1;
        line-height: 1.75;
        font-size: 16px;
        color: #353740;
        padding-top: 2px;
    }
    
    /* 流式光标 */
    .streaming-cursor {
        display: inline-block;
        width: 8px;
        height: 20px;
        background: #27ae60;
        margin-left: 2px;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

# Session State初始化
defaults = {
    'agent': None,
    'session_id': None,
    'chat_history': [],
    'current_dimension_index': 0,
    'patient_profile': {'name': '', 'age': 70, 'education_years': 6, 'sex': '女'},
    'session_started': False,
    'waiting_for_answer': False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def get_current_dimension():
    if st.session_state.current_dimension_index < len(MMSE_DIMENSIONS):
        return MMSE_DIMENSIONS[st.session_state.current_dimension_index]
    return None

def generate_first_question():
    dimension = get_current_dimension()
    questions = {
        "定向力": "您知道今天是几月几号吗？",
        "记忆力 - 即刻记忆": "现在我要说三样东西，请仔细听后重复：苹果、桌子、外套。",
        "注意力与计算力": "请帮我算一下：100减7等于多少？",
        "记忆力 - 延迟回忆": "还记得刚才让您记住的三样东西吗？",
        "语言能力": "请看着这支笔，告诉我这是什么？",
        "视空间能力": "请按照我说的画一个五边形。"
    }
    return questions.get(dimension['name'], f"我们聊聊{dimension['name']}。")

# 侧边栏
with st.sidebar:
    st.markdown("### 患者信息")
    
    disabled = st.session_state.session_started
    
    name = st.text_input("姓名", value=st.session_state.patient_profile['name'], disabled=disabled, placeholder="请输入")
    
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("年龄", 40, 100, st.session_state.patient_profile['age'], disabled=disabled)
    with c2:
        sex = st.selectbox("性别", ["女", "男"], index=0 if st.session_state.patient_profile['sex'] == '女' else 1, disabled=disabled)
    
    edu = st.number_input("教育年限", 0, 20, st.session_state.patient_profile['education_years'], disabled=disabled)
    
    if not disabled:
        st.session_state.patient_profile = {'name': name, 'age': age, 'sex': sex, 'education_years': edu}
    
    st.markdown("### 评估进度")
    
    for i, dim in enumerate(MMSE_DIMENSIONS):
        if i < st.session_state.current_dimension_index:
            st.markdown(f'<div style="color:#27ae60">✓ {dim["name"]}</div>', unsafe_allow_html=True)
        elif i == st.session_state.current_dimension_index:
            st.markdown(f'<div style="color:#3498db">• {dim["name"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:#95a5a6">○ {dim["name"]}</div>', unsafe_allow_html=True)
    
    st.progress(st.session_state.current_dimension_index / len(MMSE_DIMENSIONS))
    
    st.markdown("---")
    st.markdown("### 💡 流式输出特性")
    st.info("✨ 边生成边显示\n⚡ 首字延迟 < 0.5秒\n🎯 总耗时不变，体验翻倍")

# 主界面标题
st.title(f"🏥 AD认知评估系统")
st.caption("🌊 流式输出版本 - 实时响应")

# 显示对话历史
for msg in st.session_state.chat_history:
    if msg['role'] == 'ai':
        st.markdown(f'''
        <div class="msg">
            <div class="avatar avatar-ai">AI</div>
            <div class="msg-content">{msg['content']}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="msg">
            <div class="avatar avatar-user">U</div>
            <div class="msg-content">{msg['content']}</div>
        </div>
        ''', unsafe_allow_html=True)

# 输入区
if st.session_state.session_started and st.session_state.waiting_for_answer:
    with st.form(key="f", clear_on_submit=True):
        answer = st.text_area("患者回答", placeholder="输入患者回答...", height=52, key="input", label_visibility="collapsed")
        submit = st.form_submit_button("发送")
        
        if submit and answer:
            st.session_state.chat_history.append({'role': 'user', 'content': answer})
            st.session_state.waiting_for_answer = False
            
            # 判断是否需要切换维度
            ai_msgs = [m for m in st.session_state.chat_history if m['role'] == 'ai']
            should_switch_dimension = len(ai_msgs) >= 3 and st.session_state.current_dimension_index < len(MMSE_DIMENSIONS) - 1
            dimension_switched = False
            
            if should_switch_dimension:
                st.session_state.current_dimension_index += 1
                dimension_switched = True
            
            # 获取当前维度
            dim = get_current_dimension()
            hist = [{"role": "assistant" if m['role'] == 'ai' else "user", "content": m['content']} for m in st.session_state.chat_history]
            
            # 🌊 流式生成开始
            st.markdown("---")
            response_placeholder = st.empty()
            full_response = ""
            
            # 显示流式响应
            try:
                for chunk in st.session_state.agent.process_turn_streaming(
                    user_input=answer,
                    dimension=dim,
                    session_id=st.session_state.session_id,
                    patient_profile=st.session_state.patient_profile,
                    chat_history=hist
                ):
                    if chunk['type'] == 'token':
                        full_response = chunk['full_text']
                        # 实时显示（带流式光标）
                        response_placeholder.markdown(f'''
                        <div class="msg">
                            <div class="avatar avatar-ai">AI</div>
                            <div class="msg-content">{full_response}<span class="streaming-cursor"></span></div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    elif chunk['type'] == 'done':
                        # 完成，移除光标
                        full_response = chunk['content']
                        response_placeholder.markdown(f'''
                        <div class="msg">
                            <div class="avatar avatar-ai">AI</div>
                            <div class="msg-content">{full_response}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # 显示性能信息
                        metadata = chunk.get('metadata', {})
                        st.caption(f"⚡ 总耗时: {metadata.get('total_time', 0):.2f}秒 | 流式耗时: {metadata.get('stream_time', 0):.2f}秒")
                
                # 如果切换了维度，添加提示
                if dimension_switched:
                    full_response = f"很好。现在我们评估一下{dim['name']}。\n\n{full_response}"
                
                # 保存到历史
                st.session_state.chat_history.append({'role': 'ai', 'content': full_response})
                st.session_state.waiting_for_answer = True
                
            except Exception as e:
                st.error(f"处理失败: {e}")
                import traceback
                st.code(traceback.format_exc())
            
            st.rerun()

# 开始按钮
if not st.session_state.session_started:
    if st.button("🚀 开始评估（流式版）", type="primary"):
        if not st.session_state.patient_profile['name']:
            st.error("请输入患者姓名")
        else:
            with st.spinner("初始化流式Agent..."):
                st.session_state.agent = get_agent()
                st.session_state.session_id = f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.session_state.session_started = True
                st.session_state.waiting_for_answer = True
                
                # 生成第一个问题
                first_q = generate_first_question()
                st.session_state.chat_history.append({'role': 'ai', 'content': first_q})
                
                st.success("✅ 已启动流式对话模式")
                st.rerun()
else:
    # 控制按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重新开始", use_container_width=True):
            st.session_state.session_started = False
            st.session_state.chat_history = []
            st.session_state.current_dimension_index = 0
            st.session_state.waiting_for_answer = False
            st.rerun()
    
    with col2:
        if st.button("⏹️ 结束评估", use_container_width=True):
            st.session_state.session_started = False
            st.session_state.waiting_for_answer = False
            st.rerun()
