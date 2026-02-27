"""
现代化医疗风格 UI 样式库
"""

def get_modern_css():
    return """
    <style>
        /* ================= 字体引入 ================= */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* ================= 全局样式 ================= */
        html, body, [class*="st-"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #1a202c;
        }

        /* 背景色优化 */
        .stApp {
            background-color: #f4f7fa;
            background-image: 
                radial-gradient(at 0% 0%, rgba(74, 144, 226, 0.05) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(0, 168, 168, 0.05) 0px, transparent 50%);
        }
        
        /* 隐藏 Streamlit 默认头部和脚部 */
        #MainMenu, footer, header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* 隐藏 Streamlit 原生侧边栏导航 */
        [data-testid="stSidebarNav"] {display: none !important;}

        /* ================= 侧边栏优化 ================= */
        [data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
            box-shadow: 4px 0 24px rgba(0,0,0,0.02);
        }
        
        [data-testid="stSidebar"] .stMarkdown h1, 
        [data-testid="stSidebar"] .stMarkdown h2, 
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #2d3748;
        }
        
        /* 侧边栏链接/按钮容器 */
        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        /* ================= 组件样式优化 ================= */
        
        /* 按钮通用样式 */
        .stButton > button {
            border-radius: 12px;
            font-weight: 600;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }
        
        /* Primary 按钮 */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
            box-shadow: 0 4px 12px rgba(74, 144, 226, 0.3);
            border: none;
        }
        
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(74, 144, 226, 0.4);
        }
        
        /* Secondary 按钮 */
        .stButton > button[kind="secondary"] {
            background-color: #ffffff;
            color: #4a5568;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        
        .stButton > button[kind="secondary"]:hover {
            border-color: #cbd5e0;
            color: #2d3748;
            background-color: #f7fafc;
        }

        /* 卡片效果 (用于包裹内容) */
        div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.01), 0 2px 4px -1px rgba(0, 0, 0, 0.01);
        }

        /* 标题样式 */
        h1 {
            font-weight: 800 !important;
            letter-spacing: -0.025em;
            color: #1a202c;
        }
        
        h2 {
            font-weight: 700 !important;
            letter-spacing: -0.02em;
            color: #2d3748;
        }
        
        h3 {
            font-weight: 600 !important;
            color: #4a5568;
        }

        /* 输入框优化 */
        .stTextInput > div > div > input {
            border-radius: 10px;
            border-color: #e2e8f0;
            padding: 0.5rem 1rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #4A90E2;
            box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
        }

        /* 聊天消息样式优化 */
        .stChatMessage {
            background-color: #ffffff;
            border-radius: 16px;
            border: 1px solid #f0f0f0;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        }
        
        /* 用户消息 */
        .stChatMessage[data-testid="user-message"] {
            background-color: #f0f7ff;
            border-color: #e0efff;
        }

        /* 警告框/Info框优化 */
        .stAlert {
            border-radius: 12px;
            border: none;
        }
        
        /* 进度条 */
        .stProgress > div > div > div > div {
            background-color: #4A90E2;
            border-radius: 10px;
        }

        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 28px;
            font-weight: 700;
            color: #2d3748;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #718096;
            font-weight: 500;
        }

        /* 分割线 */
        hr {
            border-color: #e2e8f0;
            margin: 2rem 0;
        }
        
        /* Expander 详情展开 */
        .streamlit-expanderHeader {
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        }
        
        /* 调整特定元素的间距 */
        .block-container {
            padding-top: 3rem;
            padding-bottom: 5rem;
            max-width: 1200px;
        }

    </style>
    """

def get_sidebar_content(current_page="main"):
    """获取现代化侧边栏内容"""
    
    # Logo / 标题区域
    html = f"""
    <div style='text-align: center; padding: 2rem 0; margin-bottom: 2rem;'>
        <div style='
            width: 64px; 
            height: 64px; 
            margin: 0 auto 1rem; 
            background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            box-shadow: 0 10px 20px rgba(74, 144, 226, 0.2);
            color: white;
        '>
            🧠
        </div>
        <div style='font-size: 20px; font-weight: 800; color: #1a202c; margin-bottom: 0.25rem;'>
            AD 认知初筛
        </div>
        <div style='font-size: 12px; font-weight: 500; color: #718096; letter-spacing: 1px; text-transform: uppercase;'>
            Smart Screening
        </div>
    </div>
    """
    
    # 当前页面指示器
    page_map = {
        "main": ("🏠", "系统主页", "System Home"),
        "assessment": ("🩺", "认知评估", "Assessment"),
        "knowledge": ("📚", "医学知识库", "Knowledge Base"),
        "builder": ("🔧", "知识库构建", "KB Builder"),
        "voice": ("📞", "语音通话", "Voice Call")
    }
    
    icon, title, subtitle = page_map.get(current_page, ("📄", "未知页面", "Unknown"))
    
    html += f"""
    <div style='
        background-color: #f7fafc; 
        border-radius: 12px; 
        padding: 1rem; 
        margin-bottom: 2rem;
        border: 1px solid #edf2f7;
    '>
        <div style='font-size: 12px; color: #a0aec0; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase;'>
            CURRENT MODULE
        </div>
        <div style='display: flex; align-items: center; gap: 10px;'>
            <div style='font-size: 20px;'>{icon}</div>
            <div>
                <div style='font-size: 15px; font-weight: 700; color: #2d3748;'>{title}</div>
                <div style='font-size: 11px; color: #718096;'>{subtitle}</div>
            </div>
        </div>
    </div>
    """
    
    # 帮助/说明区域
    html += """
    <div style='margin-top: auto; padding-top: 2rem; border-top: 1px solid #edf2f7;'>
        <div style='font-size: 13px; color: #4a5568; margin-bottom: 1rem; font-weight: 600;'>
            <span style='margin-right: 6px;'>💡</span> 操作指南
        </div>
        <div style='font-size: 12px; color: #718096; line-height: 1.6;'>
            请使用主界面的卡片或上方导航栏切换不同的功能模块。
        </div>
    </div>
    
    <div style='margin-top: 2rem; text-align: center; font-size: 11px; color: #cbd5e0;'>
        v1.0.0 • Build 2025
    </div>
    """
    
    return html
