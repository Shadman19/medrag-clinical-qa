"""
app.py
-------
MedRAG — Interactive Medical Question Answering Web App
Run with: streamlit run app.py
"""

import streamlit as st
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="MedRAG — Clinical QA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f4c81 0%, #1a7abf 50%, #0d6ebd 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .main-header p { font-size: 1.05rem; opacity: 0.9; margin: 0.5rem 0 0 0; }

    .answer-box {
        background: #f0f7ff;
        border-left: 5px solid #1a7abf;
        border-radius: 0 12px 12px 0;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .source-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #0f4c81;
        border: 1px solid #b8d9f5;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0.2rem;
    }

    .metric-card {
        background: white;
        border: 1px solid #e0e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a7abf; }
    .metric-label { font-size: 0.82rem; color: #666; margin-top: 0.2rem; }

    .chunk-card {
        background: #fafbfc;
        border: 1px solid #e4e8ed;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
    }
    .score-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #1a7abf, #4db8ff);
        margin-top: 0.4rem;
    }

    .sample-q-btn {
        background: #f0f7ff;
        border: 1px solid #b8d9f5;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        cursor: pointer;
        font-size: 0.85rem;
        color: #0f4c81;
    }

    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-weight: 500;
    }

    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Pipeline ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    from src.rag.pipeline import MedRAGPipeline
    from src.data.build_index import build
    from src.rag.retriever import MedicalRetriever
    import config

    pipeline = MedRAGPipeline()
    if not pipeline.load():
        with st.spinner("🔨 Building knowledge base index (first run only)..."):
            build()
        pipeline.load()
    return pipeline


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=5, value=3,
                       help="How many document chunks to retrieve per query")

    show_chunks = st.checkbox("Show retrieved chunks", value=True,
                               help="Display the raw text passages used to generate the answer")

    show_scores = st.checkbox("Show similarity scores", value=True)

    st.markdown("---")
    st.markdown("## 📚 Knowledge Base")
    st.markdown("""
    **5 Medical Topics:**
    - 🩺 Type 2 Diabetes
    - ❤️ Cardiovascular Disease
    - 🔬 Cancer Biology
    - 💊 Hypertension
    - 🧠 Mental Health
    """)

    st.markdown("---")
    st.markdown("## 🏗️ Architecture")
    st.markdown("""
    **Embedding:** `all-MiniLM-L6-v2`
    **Vector Store:** FAISS (IndexFlatIP)
    **Generator:** `flan-t5-base`
    **Chunking:** Overlapping windows + section splits
    """)

    st.markdown("---")
    if st.button("📊 Run Evaluation Suite"):
        st.session_state.run_eval = True

    st.markdown("---")
    st.markdown("Built by **Shadman Mahmood Khan Pathan**")
    st.markdown("[GitHub](https://github.com/Shadman19) · [LinkedIn](https://linkedin.com/in/shadmanmahmood9)")


# ── Main UI ──────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 MedRAG</h1>
    <p>Clinical Question Answering · Retrieval-Augmented Generation · Source-Cited Answers</p>
</div>
""", unsafe_allow_html=True)

# Load pipeline
with st.spinner("Loading MedRAG..."):
    pipeline = load_rag_pipeline()

# Sample questions
st.markdown("#### 💡 Try a Sample Question")
sample_questions = [
    "What are the symptoms of Type 2 diabetes?",
    "How is hypertension classified and treated?",
    "What causes cardiovascular disease?",
    "What is the Gleason score in prostate cancer?",
    "What medications treat depression?",
    "How is cancer diagnosed and staged?",
]

cols = st.columns(3)
for i, q in enumerate(sample_questions):
    if cols[i % 3].button(q, key=f"sample_{i}", use_container_width=True):
        st.session_state.selected_question = q

st.markdown("---")

# Query input
default_q = st.session_state.get("selected_question", "")
question = st.text_area(
    "**Ask a medical question:**",
    value=default_q,
    placeholder="e.g. What are the symptoms and treatment options for Type 2 diabetes?",
    height=90,
)

col1, col2, col3 = st.columns([2, 1, 4])
ask_btn = col1.button("🔍 Ask MedRAG", type="primary", use_container_width=True)
clear_btn = col2.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state.selected_question = ""
    st.rerun()

# ── Query Execution ──────────────────────────────────────────
if ask_btn and question.strip():
    with st.spinner("🔍 Retrieving relevant passages and generating answer..."):
        result = pipeline.query(question.strip(), top_k=top_k)

    # Answer
    st.markdown("### 💬 Answer")
    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

    # Sources
    if result["sources"]:
        st.markdown("**📄 Sources used:**")
        source_html = "".join(f'<span class="source-badge">📖 {s}</span>' for s in result["sources"])
        st.markdown(source_html, unsafe_allow_html=True)

    st.markdown("---")

    # Metrics row
    st.markdown("### 📊 Retrieval Analytics")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result['latency_ms']:.0f}ms</div>
            <div class="metric-label">Query Latency</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{result['num_chunks_retrieved']}</div>
            <div class="metric-label">Chunks Retrieved</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        top_score = result['retrieval_scores'][0] if result['retrieval_scores'] else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{top_score:.3f}</div>
            <div class="metric-label">Top Similarity Score</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        avg_score = sum(result['retrieval_scores']) / len(result['retrieval_scores']) if result['retrieval_scores'] else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_score:.3f}</div>
            <div class="metric-label">Avg Similarity</div>
        </div>""", unsafe_allow_html=True)

    # Retrieved chunks
    if show_chunks and result["chunks"]:
        st.markdown("---")
        st.markdown("### 🔎 Retrieved Passages")

        for i, chunk in enumerate(result["chunks"]):
            score = chunk["similarity_score"]
            score_pct = int(score * 100)
            bar_width = min(100, score_pct)

            with st.expander(f"Chunk {i+1}: {chunk['doc_title']} — Score: {score:.4f}", expanded=(i == 0)):
                if show_scores:
                    st.markdown(f"""
                    <div style="margin-bottom:0.8rem;">
                        <span style="font-size:0.82rem;color:#666;">Similarity: {score:.4f}</span>
                        <div class="score-bar" style="width:{bar_width}%"></div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f'<div class="chunk-card">{chunk["text"]}</div>', unsafe_allow_html=True)
                st.caption(f"Source: {chunk['doc_source']} | File: {chunk['doc_filename']}")

    # Save to history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, {
        "question": question,
        "answer": result["answer"],
        "sources": result["sources"],
        "latency_ms": result["latency_ms"],
    })

elif ask_btn and not question.strip():
    st.warning("Please enter a question first.")


# ── Evaluation Tab ───────────────────────────────────────────
if st.session_state.get("run_eval", False):
    st.markdown("---")
    st.markdown("## 🧪 Evaluation Suite Results")

    with st.spinner("Running evaluation on benchmark questions..."):
        from src.evaluation.evaluate import run_evaluation
        eval_results = run_evaluation(pipeline)

    summary = eval_results["summary"]
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Retrieval Precision", f"{summary['avg_retrieval_precision']}%")
    e2.metric("Answer Faithfulness", f"{summary['avg_answer_faithfulness']}%")
    e3.metric("Answer Relevance", f"{summary['avg_answer_relevance']}%")
    e4.metric("Avg Latency", f"{summary['avg_latency_ms']}ms")

    import pandas as pd
    df = pd.DataFrame(eval_results["results"])
    st.dataframe(df[["question", "retrieval_precision", "answer_faithfulness",
                      "answer_relevance", "latency_ms"]].round(1),
                 use_container_width=True)

    st.session_state.run_eval = False


# ── History ──────────────────────────────────────────────────
if st.session_state.get("history"):
    st.markdown("---")
    st.markdown("### 🕐 Query History")
    for i, h in enumerate(st.session_state.history[:5]):
        with st.expander(f"Q: {h['question'][:80]}...", expanded=False):
            st.markdown(f"**Answer:** {h['answer']}")
            st.caption(f"Sources: {', '.join(h['sources'])} | {h['latency_ms']:.0f}ms")
