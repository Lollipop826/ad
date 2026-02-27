"""
统一的医疗绿侧边栏样式
"""

def get_medical_sidebar_css():
    return """
    <style>
        /* 隐藏Streamlit默认元素 */
        #MainMenu, footer, header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* 全局字体 */
        * {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }
        
        /* 医疗绿侧边栏样式 */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2d5a27 0%, #1a3d1a 100%);
            padding-top: 2rem;
        }
        
        /* 侧边栏内所有按钮 - 医疗绿主题 */
        [data-testid="stSidebar"] .stButton button {
            width: 100%;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            font-size: 16px;
            font-weight: 600;
            border: 2px solid transparent;
            transition: all 0.3s ease;
            margin: 0.3rem 0;
            text-align: left;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        
        /* 主要按钮（选中状态）- 医疗绿 */
        [data-testid="stSidebar"] .stButton button[kind="primary"] {
            background: linear-gradient(135deg, #4a7c59 0%, #2d5a27 100%);
            color: white;
            box-shadow: 0 4px 12px rgba(74, 124, 89, 0.4);
            border: 2px solid rgba(255, 255, 255, 0.3);
        }
        
        /* 次要按钮（未选中状态）- 医疗绿 */
        [data-testid="stSidebar"] .stButton button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border: 2px solid rgba(255, 255, 255, 0.2);
        }
        
        [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.15);
            color: white;
            border-color: rgba(255, 255, 255, 0.3);
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(74, 124, 89, 0.2);
        }
        
        /* 侧边栏文字 - 医疗绿主题强制白色 */
        [data-testid="stSidebar"] {
            color: white !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: white !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown div,
        [data-testid="stSidebar"] .stMarkdown span,
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown h4,
        [data-testid="stSidebar"] .stMarkdown h5,
        [data-testid="stSidebar"] .stMarkdown h6,
        [data-testid="stSidebar"] .stMarkdown ul,
        [data-testid="stSidebar"] .stMarkdown li {
            color: white !important;
        }
        
        /* Streamlit原生导航链接强制白色 - 最强覆盖 */
        [data-testid="stSidebar"] .css-1d391kg,
        [data-testid="stSidebar"] .css-1n76uvr,
        [data-testid="stSidebar"] .css-10trblm,
        [data-testid="stSidebar"] section[data-testid="stSidebar"] > div > div > div *,
        [data-testid="stSidebar"] .css-1d391kg a,
        [data-testid="stSidebar"] .css-1n76uvr a,
        [data-testid="stSidebar"] .css-10trblm a,
        [data-testid="stSidebar"] .css-1d391kg div,
        [data-testid="stSidebar"] .css-1n76uvr div,
        [data-testid="stSidebar"] .css-10trblm div {
            color: white !important;
            font-weight: 600 !important;
        }
        
        /* 强制所有链接文字白色 */
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] label {
            color: white !important;
        }
        
        /* 侧边栏导航链接样式 - 增强版 */
        [data-testid="stSidebar"] a {
            color: white !important;
            text-decoration: none;
            padding: 0.8rem 1.2rem;
            border-radius: 12px;
            transition: all 0.3s ease;
            display: block;
            margin: 0.4rem 0.5rem;
            font-size: 16px;
            font-weight: 600;
            border: 2px solid transparent;
            position: relative;
            text-align: center;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
        }
        
        [data-testid="st-sidebar"] a::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255 0.05);
            border-radius: 12px;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        [data-testid="stSidebar"] a:hover {
            background: rgba(74, 124, 89, 0.4);
            color: white !important;
            transform: translateX(5px) scale(1.02);
            border-color: rgba(255, 255, 255, 0.3);
            box-shadow: 0 4px 12px rgba(74, 124, 89, 0.4);
        }
        
        [data-testid="stSidebar"] a:hover::before {
            opacity: 1;
        }
        
        [data-testid="stSidebar"] a[aria-current="page"] {
            background: linear-gradient(135deg, #4a7c59 0%, #2d5a27 100%);
            color: white !important;
            box-shadow: 0 4px 12px rgba(74, 124, 89, 0.5);
            border: 2px solid rgba(255, 255, 255, 0.4);
            font-weight: 700;
            transform: scale(1.05);
        }
        
        [data-testid="stSidebar"] a[aria-current="page"]::after {
            content: "▶";
            position: absolute;
            right: 1rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 14px;
            opacity: 0.8;
        }
        
        /* 信息提示框 - 医疗绿 */
        [data-testid="stSidebar"] .stAlert {
            background: rgba(74, 124, 89, 0.2);
            border-left: 4px solid #4a7c59;
            color: white !important;
        }
        
        /* 侧边栏滚动条样式 */
        [data-testid="stSidebar"]::-webkit-scrollbar {
            width: 6px;
        }
        
        [data-testid="stSidebar"]::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }
        
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
            background: rgba(74, 124, 89, 0.5);
            border-radius: 3px;
        }
        
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
            background: rgba(74, 124, 89, 0.7);
        }
    </style>
    """

def get_medical_sidebar_content(current_page="main"):
    """获取统一的侧边栏内容"""
    
    # 系统标题
    sidebar_html = """
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <div style='font-size: 36px; margin-bottom: 0.5rem;'>🏥</div>
        <div style='font-size: 24px; font-weight: 700; color: white; margin-bottom: 0.3rem;'>
            AD初筛系统
        </div>
        <div style='font-size: 12px; color: #95a5a6; letter-spacing: 2px;'>
            ALZHEIMER'S SCREENING
        </div>
    </div>
    """
    
    # 分隔线
    sidebar_html += "<div style='height: 1px; background: rgba(255,255,255,0.1); margin: 2rem 0;'></div>"
    
    # 页面信息
    if current_page == "assessment":
        sidebar_html += """
        <div style='color: white; font-size: 14px; padding: 0 0.5rem; line-height: 1.6;'>
            <p style='margin-bottom: 0.5rem; color: white;'>🩺 <strong style='color: white;'>当前页面：患者评估</strong></p>
            <p style='font-size: 12px; color: white;'>智能化的认知功能评估，基于MMSE标准</p>
        </div>
        """
    elif current_page == "knowledge":
        sidebar_html += """
        <div style='color: white; font-size: 14px; padding: 0 0.5rem; line-height: 1.6;'>
            <p style='margin-bottom: 0.5rem; color: white;'>📚 <strong style='color: white;'>当前页面：知识库浏览</strong></p>
            <p style='font-size: 12px; color: white;'>专业医学知识检索与浏览</p>
        </div>
        """
    else:
        sidebar_html += """
        <div style='color: white; font-size: 14px; padding: 0 0.5rem; line-height: 1.6;'>
            <p style='margin-bottom: 0.5rem; color: white;'>🏠 <strong style='color: white;'>当前页面：系统主页</strong></p>
            <p style='font-size: 12px; color: white;'>欢迎使用AD初筛系统</p>
        </div>
        """
    
    # 分隔线
    sidebar_html += "<div style='height: 1px; background: rgba(255,255,255,0.1); margin: 2rem 0;'></div>"
    
    # 使用说明
    sidebar_html += """
    <div style='color: white; font-size: 12px; padding: 0 0.5rem; line-height: 1.5;'>
        <p style='margin-bottom: 0.5rem; color: white;'>💡 <strong style='color: white;'>使用提示:</strong></p>
        <p style='margin: 0; font-size: 11px; color: white;'>使用顶部导航切换页面功能</p>
    </div>
    """
    
    # 分隔线和版本信息
    sidebar_html += "<div style='height: 1px; background: rgba(255,255,255,0.1); margin: 2rem 0;'></div>"
    sidebar_html += """
    <div style='text-align: center; margin-top: 3rem; padding: 1rem; color: white;'>
        <div style='font-size: 11px; margin-bottom: 0.3rem; color: white;'>Version 1.0.0</div>
        <div style='font-size: 10px; opacity: 0.7; color: white;'>© 2024 AD System</div>
    </div>
    """
    
    return sidebar_html

