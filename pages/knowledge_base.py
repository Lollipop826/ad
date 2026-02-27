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




# CSS样式 - 适配 Modern 主题
st.markdown("""
<style>
    .doc-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #4A90E2; /* 医疗蓝 */
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #edf2f7;
    }
    
    .doc-title {
        font-size: 16px;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .doc-content {
        font-size: 14px;
        line-height: 1.8;
        color: #4a5568;
        margin-top: 0.5rem;
    }
    
    .highlight {
        background: #ebf8ff;
        color: #2b6cb0;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .score-badge {
        display: inline-block;
        background: #edf2f7;
        color: #4a5568;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    
    .score-high {
        background: #c6f6d5;
        color: #22543d;
    }
    
    .score-medium {
        background: #feebc8;
        color: #744210;
    }
    
    .score-low {
        background: #fed7d7;
        color: #822727;
    }
    
    /* 统计卡片优化 */
    .stats-container {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    .stats-number {
        font-size: 32px;
        font-weight: 800;
        color: #2d3748;
        line-height: 1.2;
    }
    
    .stats-label {
        font-size: 12px;
        color: #718096;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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

def load_all_chunks():
    """加载所有知识分块"""
    kb_dir = project_root / "kb" / "chunks_semantic_per_file"
    all_chunks = []
    
    if kb_dir.exists():
        jsonl_files = sorted(kb_dir.glob("*.jsonl"))
        
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "").strip()
                        if text:
                            all_chunks.append({
                                "content": text,
                                "source": jsonl_file.stem,
                                "line_no": line_no,
                                "metadata": record.get("metadata", {})
                            })
                    except:
                        continue
    
    return all_chunks

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
    score_class = "score-badge score-high"
    if score < 0.5:
        score_class = "score-badge score-low"
    elif score < 0.7:
        score_class = "score-badge score-medium"
    
    st.markdown(f"""
    <div class="doc-card">
        <div class="doc-title">
            <span class="{score_class}">#{result['rank']} 相关度: {score:.2%}</span>
            <span style="margin-left: auto;">📄 {result['source']}</span>
        </div>
    """, unsafe_allow_html=True)
    
    # 显示高亮句子
    if result['highlighted_sentences']:
        st.markdown("**相关内容：**")
        for sentence, sent_score in result['highlighted_sentences'][:3]:
            st.markdown(f"""
            <div class="doc-content">
                <span class="highlight">{sentence}</span>
                <span style="color: #718096; font-size: 11px;"> ({sent_score:.2%})</span>
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
st.title("📚 阿尔茨海默病知识库")

# 侧边栏 - 统计和设置
with st.sidebar:
    st.markdown("### 📊 知识库概览")
    
    stats = load_kb_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="stats-container">
            <div class="stats-number">{stats['total_docs']}</div>
            <div class="stats-label">文档总数</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stats-container">
            <div class="stats-number">{stats['total_chunks']}</div>
            <div class="stats-label">知识片段</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 浏览模式")
    view_mode = st.radio(
        "选择模式",
        ["🔍 智能搜索", "📖 浏览全部"],
        label_visibility="collapsed"
    )
    
    st.markdown("### ⚙️ 显示设置")
    if view_mode == "🔍 智能搜索":
        top_k = st.slider("返回结果数量", 1, 20, 5)
    else:
        chunks_per_page = st.slider("每页显示数量", 5, 50, 20)
    
    st.info("""
    **💡 提示**
    
    **智能搜索**:
    输入问题，系统会自动匹配最相关的医学知识。
    
    **浏览全部**:
    查看所有已入库的知识片段。
    """)


# 主内容区域
if view_mode == "🔍 智能搜索":
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

else:  # 浏览全部模式
    st.markdown("### 📖 浏览所有知识分块")
    
    # 加载所有分块
    with st.spinner("加载知识库..."):
        all_chunks = load_all_chunks()
    
    if not all_chunks:
        st.warning("未找到知识库内容")
    else:
        # 分页设置
        total_chunks = len(all_chunks)
        total_pages = (total_chunks + chunks_per_page - 1) // chunks_per_page
        
        # 初始化page（确保是整数类型）
        if 'kb_current_page' not in st.session_state:
            st.session_state.kb_current_page = 1
        
        # 确保当前页是整数
        st.session_state.kb_current_page = int(st.session_state.kb_current_page)
        
        # 显示统计
        st.info(f"📊 共 {total_chunks} 个知识分块 | 每页 {chunks_per_page} 个 | 共 {total_pages} 页")
        
        # 分页控制
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ 首页", disabled=(st.session_state.kb_current_page == 1)):
                st.session_state.kb_current_page = 1
                st.rerun()
        
        with col2:
            if st.button("◀️ 上页", disabled=(st.session_state.kb_current_page == 1)):
                st.session_state.kb_current_page -= 1
                st.rerun()
        
        with col3:
            st.markdown(f"<div style='text-align: center; padding: 8px;'><b>第 {st.session_state.kb_current_page} / {total_pages} 页</b></div>", unsafe_allow_html=True)
        
        with col4:
            if st.button("▶️ 下页", disabled=(st.session_state.kb_current_page == total_pages)):
                st.session_state.kb_current_page += 1
                st.rerun()
        
        with col5:
            if st.button("⏭️ 末页", disabled=(st.session_state.kb_current_page == total_pages)):
                st.session_state.kb_current_page = total_pages
                st.rerun()
        
        st.markdown("---")
        
        # 显示当前页的分块
        start_idx = (st.session_state.kb_current_page - 1) * chunks_per_page
        end_idx = min(start_idx + chunks_per_page, total_chunks)
        
        current_chunks = all_chunks[start_idx:end_idx]
        
        for idx, chunk in enumerate(current_chunks, start=start_idx + 1):
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-title">
                    <span class="score-badge">#{idx}</span>
                    {chunk['source']} (第 {chunk['line_no']} 行)
                </div>
            """, unsafe_allow_html=True)
            
            # 显示内容预览（前200字）
            preview = chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            st.markdown(f"""
            <div class="doc-content">
                {preview}
            </div>
            """, unsafe_allow_html=True)
            
            # 完整内容（可展开）
            with st.expander("查看完整内容"):
                st.text_area(
                    "内容",
                    chunk['content'],
                    height=200,
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"chunk_{idx}"
                )
                
                # 元数据
                if chunk['metadata']:
                    st.json(chunk['metadata'])
            
            st.markdown("</div>", unsafe_allow_html=True)

