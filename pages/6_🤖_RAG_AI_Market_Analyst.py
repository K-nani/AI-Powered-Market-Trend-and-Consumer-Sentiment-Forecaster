"""Dashboard 6: RAG AI Market Analyst (FAISS + LLM)."""

import streamlit as st
import pandas as pd
from src.ui_common import setup_page, render_sidebar, render_banner, format_metric_card
from src.data.market_data import PRODUCT_CATALOG, get_multichannel_review_corpus
from src.engine.rag_engine import ReviewRAGIndex, MarketCopilot


setup_page("AI Market Analyst", "🤖")
config = render_sidebar()

meta = PRODUCT_CATALOG[config["product_id"]]
reviews = get_multichannel_review_corpus(config["product_id"])

render_banner(
    title=f"AI Market Analyst & Semantic RAG Copilot",
    subtitle=f"Conversational Market Intelligence backed by FAISS vector similarity search and LLM contextual synthesis ({meta['name']})",
    badge="Vector RAG Engine"
)

# Initialize FAISS Index
if "rag_index_market" not in st.session_state or st.session_state.get("current_market_prod") != config["product_id"]:
    rag = ReviewRAGIndex(embedding_dim=64)
    rag.build_index(reviews)
    st.session_state["rag_index_market"] = rag
    st.session_state["current_market_prod"] = config["product_id"]

rag = st.session_state["rag_index_market"]
copilot = MarketCopilot(rag, gemini_api_key=config.get("gemini_api_key"))

# Vector DB Stats
v1, v2, v3, v4 = st.columns(4)
with v1:
    st.markdown(format_metric_card("Vector Database Engine", "FAISS In-Memory", "Dense Vector Store", True), unsafe_allow_html=True)
with v2:
    st.markdown(format_metric_card("Indexed Multi-Channel Voices", f"{len(reviews)} Documents", "Twitter, Amazon, Reddit", True), unsafe_allow_html=True)
with v3:
    st.markdown(format_metric_card("Embedding Dimensions", "64-D Normalized", "Cosine Similarity Search", True), unsafe_allow_html=True)
with v4:
    llm_name = "Gemini 2.5 Flash" if config.get("gemini_api_key") else "Local Semantic Synthesizer"
    st.markdown(format_metric_card("Active LLM Backend", llm_name, "Ready for inquiries", True), unsafe_allow_html=True)

st.markdown("---")

# Query Presets
if config["product_id"] == "electronics_flagship":
    presets = [
        "What are the main customer complaints regarding thermal throttling and charging speed?",
        "How is the titanium build quality and camera zoom received across social media?",
        "What is the forecast impact of customer dissatisfaction regarding the missing wall charger?"
    ]
elif config["product_id"] == "ev_automotive":
    presets = [
        "What are owners saying about Supercharger V4 speeds and charging network reliability?",
        "Are panel gaps and tailgate fit issues still affecting brand loyalty?",
        "How do recent price incentive cuts impact monthly order velocity?"
    ]
else:
    presets = [
        "Why are customers warning others about narrow sizing and blister discomfort?",
        "How do limited streetwear collab drops drive secondary market buzz and resale hype?",
        "What are the top durability complaints regarding sole delamination?"
    ]

st.markdown("### 💬 Ask the Market Analyst")

selected_preset = st.selectbox("Quick Market Inquiries", ["Select or enter custom question..."] + presets)
initial_q = presets[0] if selected_preset == "Select or enter custom question..." else selected_preset

q_col1, q_col2 = st.columns([4, 1])
with q_col1:
    user_query = st.text_input("Enter your market intelligence question:", value=initial_q)
with q_col2:
    top_k = st.slider("Retrieved Sources (k)", min_value=2, max_value=8, value=4)

if st.button("🔍 Run Vector Search & Synthesize Market Intelligence", type="primary"):
    with st.spinner("Querying FAISS vector index & running contextual synthesis..."):
        res = copilot.answer_query(user_query, top_k=top_k)

    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 4px solid #6366f1; margin-top: 16px;">
            <div style="font-size: 0.8rem; color: #818cf8; text-transform: uppercase; font-weight: 700;">
                🧠 Intelligence Synthesized via {res['engine_used']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(res["answer"])

    st.markdown("---")
    st.markdown("### 📑 Grounded Consumer Citations (FAISS Dense Vector Matches)")

    for idx, doc in enumerate(res["retrieved_evidence"]):
        sim = doc.get("similarity_score", 0.0)
        channel = doc.get("channel", "Twitter / X")
        rating = doc.get("rating", 4.0)
        votes = doc.get("helpfulness", 0)
        sent = doc.get("sentiment", "Neutral")
        aspect = doc.get("aspect", "General")
        badge_color = "#10b981" if sent == "Positive" else ("#ef4444" if sent == "Negative" else "#94a3b8")

        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div>
                        <span style="background: rgba(99, 102, 241, 0.25); color: #c7d2fe; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">{channel}</span>
                        <span style="background: {badge_color}33; color: {badge_color}; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; margin-left: 6px;">{sent}</span>
                        <span style="background: rgba(245, 158, 11, 0.2); color: #fcd34d; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; margin-left: 6px;">🏷️ {aspect}</span>
                        <span style="font-size: 0.8rem; color: #cbd5e1; margin-left: 8px;">★ {rating}/5.0</span>
                        <span style="font-size: 0.8rem; color: #94a3b8; margin-left: 8px;">👍 {votes} votes</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #818cf8; font-family: monospace;">
                        Cosine Match: {sim:.4f}
                    </div>
                </div>
                <div style="font-size: 0.92rem; color: #f1f5f9; line-height: 1.5;">
                    "{doc.get('text', '')}"
                </div>
                <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                    Author: {doc.get('author', 'Anonymous')} | Date: {doc.get('date', 'Recent')}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
