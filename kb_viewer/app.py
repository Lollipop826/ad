"""
知识库可视化应用
"""

import streamlit as st
import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径（必须在最前面）
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 现在导入项目模块
try:
    from src.tools.retrieval.paragraph_retrieval import paragraph_retrieval
    from src.tools.retrieval.sentence_filter import SentenceFilter, split_sentences
    import torch
except ImportError as e:
    st.error(f"导入错误: {e}")
    st.error(f"项目根目录: {project_root}")
    st.error(f"sys.path: {sys.path[:3]}")
    st.stop()

st.set_page_config(
    page_title="阿尔茨海默病知识库浏览器",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式
st.markdown("""
<style>
    .main {
        background: #f8f9fa;
    }
    
    .doc-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #27ae60;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .doc-title {
        font-size: 18px;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .doc-content {
        font-size: 14px;
        line-height: 1.8;
        color: #555;
    }
    
    .doc-meta {
        font-size: 12px;
        color: #7f8c8d;
        margin-top: 0.5rem;
    }
    
    .highlight {
        background: #fff3cd;
        padding: 2px 4px;
        border-radius: 3px;
    }
    
    .score-badge {
        display: inline-block;
        background: #27ae60;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    
    .score-medium {
        background: #f39c12;
    }
    
    .score-low {
        background: #95a5a6;
    }
    
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stats-number {
        font-size: 32px;
        font-weight: 700;
        color: #27ae60;
    }
    
    .stats-label {
        font-size: 14px;
        color: #7f8c8d;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 初始化
@st.cache_resource
def init_sentence_filter():
    return SentenceFilter()

def load_kb_stats():
    """加载知识库统计信息"""
    kb_dir = project_root / "kb" / "chunks_semantic_per_file"
    
    total_docs = 0
    total_chunks = 0
    
    if kb_dir.exists():
        jsonl_files = list(kb_dir.glob("*.jsonl"))
        total_docs = len(jsonl_files)
        
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        total_chunks += 1
    
    return {
        "total_docs": total_docs,
        "total_chunks": total_chunks
    }

def search_knowledge(query: str, top_k: int = 10):
    """搜索知识库"""
    try:
        docs = paragraph_retrieval(
            query=query,
            persist_dir=str(project_root / "kb" / ".chroma_semantic"),
            collection_name="ad_kb_semantic",
            embedding_model="BAAI/bge-m3",
            device="cpu",
            k=top_k * 2
        )
        
        # 使用句子过滤器计算相关性
        sentence_filter = init_sentence_filter()
        results = []
        
        for i, doc in enumerate(docs[:top_k], 1):
            sentences = split_sentences(doc.page_content)
            if not sentences:
                continue
            
            # 计算句子相关性
            pairs = [[query, s] for s in sentences]
            with torch.inference_mode():
                scores = sentence_filter.model.predict(pairs).tolist()
            
            # 计算平均分数
            avg_score = sum(scores) / len(scores) if scores else 0
            max_score = max(scores) if scores else 0
            
            # 高亮相关句子
            highlighted_sentences = [
                (s, score) for s, score in zip(sentences, scores) if score >= 0.4
            ]
            
            results.append({
                "rank": i,
                "content": doc.page_content,
                "sentences": sentences,
                "highlighted_sentences": highlighted_sentences,
                "avg_score": avg_score,
                "max_score": max_score,
                "source": doc.metadata.get("filename", "未知"),
                "metadata": doc.metadata
            })
        
        return results
    except Exception as e:
        st.error(f"搜索出错: {str(e)}")
        return []

def display_search_result(result, query):
    """显示搜索结果"""
    score = result['max_score']
    score_class = "score-badge"
    if score < 0.5:
        score_class += " score-low"
    elif score < 0.7:
        score_class += " score-medium"
    
    st.markdown(f"""
    <div class="doc-card">
        <div class="doc-title">
            <span class="{score_class}">#{result['rank']} 相关度: {score:.2%}</span>
            {result['source']}
        </div>
    """, unsafe_allow_html=True)
    
    # 显示高亮句子
    if result['highlighted_sentences']:
        st.markdown("**相关内容：**")
        for sentence, sent_score in result['highlighted_sentences'][:3]:
            st.markdown(f"""
            <div class="doc-content">
                <span class="highlight">{sentence}</span>
                <span style="color: #27ae60; font-size: 12px;"> ({sent_score:.2%})</span>
            </div>
            """, unsafe_allow_html=True)
    
    # 完整内容（可展开）
    with st.expander("查看完整内容"):
        st.text_area(
            "内容", 
            result['content'], 
            height=200, 
            disabled=True,
            label_visibility="collapsed"
        )
        
        # 元数据
        if result['metadata']:
            st.json(result['metadata'])
    
    st.markdown("</div>", unsafe_allow_html=True)

# 主界面
st.title("📚 阿尔茨海默病知识库浏览器")

# 侧边栏 - 统计和设置
with st.sidebar:
    st.header("📊 知识库统计")
    
    stats = load_kb_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats['total_docs']}</div>
            <div class="stats-label">文档数量</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{stats['total_chunks']}</div>
            <div class="stats-label">知识片段</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("⚙️ 搜索设置")
    top_k = st.slider("返回结果数量", 1, 20, 5)
    
    st.markdown("---")
    
    st.header("💡 使用说明")
    st.markdown("""
    1. 在搜索框输入查询内容
    2. 查看相关度最高的知识
    3. 展开查看完整内容
    4. 高亮部分是最相关的句子
    """)

# 主内容区域
st.markdown("### 🔍 搜索知识库")

query = st.text_input(
    "输入查询内容",
    placeholder="例如：阿尔茨海默病的早期症状、定向力评估、记忆力测试...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 4])
with col1:
    search_button = st.button("🔍 搜索", type="primary", use_container_width=True)

if search_button and query:
    with st.spinner("搜索中..."):
        results = search_knowledge(query, top_k=top_k)
    
    if results:
        st.markdown(f"### 📋 搜索结果（共 {len(results)} 条）")
        
        for result in results:
            display_search_result(result, query)
    else:
        st.warning("没有找到相关内容，请尝试其他关键词。")

elif not query and search_button:
    st.info("请输入搜索内容")

# 示例查询
st.markdown("---")
st.markdown("### 💡 示例查询")

example_queries = [
    "阿尔茨海默病的早期症状",
    "定向力评估方法",
    "记忆力测试",
    "MMSE量表",
    "认知功能筛查",
    "非药物干预"
]

cols = st.columns(3)
for i, example in enumerate(example_queries):
    with cols[i % 3]:
        if st.button(f"🔖 {example}", key=f"example_{i}"):
            st.rerun()

