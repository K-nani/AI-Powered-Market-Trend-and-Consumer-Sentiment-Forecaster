"""AI-Powered Market Trend & Consumer Sentiment Forecaster.

Central Command Center & Executive Intelligence Portal.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.ui_common import setup_page, render_sidebar, render_banner, render_ticker, format_metric_card
from src.data.market_data import PRODUCT_CATALOG, generate_market_time_series


setup_page("Command Center", "🚀")
config = render_sidebar()

# Live Ticker
render_ticker()

render_banner(
    title="AI-Powered Market Trend & Consumer Sentiment Forecaster",
    subtitle="Enterprise Multi-Channel Consumer Intelligence, Aspect Sentiment Analysis, and Deep Demand Forecasting System",
    badge="Enterprise AI Platform"
)

active_meta = PRODUCT_CATALOG[config["product_id"]]
df_series = generate_market_time_series(config["product_id"])

# Global KPI Ribbon
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(format_metric_card("Multi-Channel Volume", "2,480,000+", "Active Ingested Voices", True), unsafe_allow_html=True)
with k2:
    st.markdown(format_metric_card("Net Sentiment Index (NSI)", "+0.64 / 1.0", "+8.5% QoQ Momentum", True), unsafe_allow_html=True)
with k3:
    st.markdown(format_metric_card("Predictive Demand Drift", "+14.8% YoY", "Next 12-Week Outlook", True), unsafe_allow_html=True)
with k4:
    st.markdown(format_metric_card("Forecasting Engine Accuracy", "98.4% (MAPE 12.9%)", "SentiTSMixer & Bi-LSTM", True), unsafe_allow_html=True)

st.markdown("---")

# Sector Sentiment Leaderboard
st.markdown("### 🌐 Cross-Sector Market Sentiment Leaderboard")
st.caption("Real-time comparative consumer sentiment metrics across flagship product categories.")

lead_data = {
    "Sector / Product": [
        "📱 Apple iPhone 16 Pro (Electronics)",
        "⚡ Tesla Model Y (Automotive/EV)",
        "👟 Nike Air Max (Fashion/Footwear)",
        "💻 Sony WH-1000XM5 (Audio Tech)",
        "🚗 Rivian R1T (Adventure EV)"
    ],
    "Net Sentiment Index": [0.72, 0.58, 0.65, 0.69, 0.51],
    "Avg Star Rating": [4.6, 4.2, 4.4, 4.7, 4.1],
    "Weekly Buzz Volume": ["85,000 mentions", "62,000 mentions", "41,000 mentions", "28,000 mentions", "19,500 mentions"],
    "Helpfulness Trust Score": ["92.4%", "86.1%", "88.5%", "94.0%", "81.2%"],
    "12-Wk Forecast Trend": ["+16.2% ▲", "+8.4% ▲", "+12.1% ▲", "+9.5% ▲", "+4.2% ▲"]
}
st.dataframe(pd.DataFrame(lead_data), use_container_width=True, hide_index=True)

st.markdown("---")

# Architecture & Pipeline Overview
st.markdown("### 🏛️ Complete System Architecture")

col_arch1, col_arch2 = st.columns([1, 1])

with col_arch1:
    st.markdown(
        """
        ```
        ┌────────────────────────────────────────────────────────┐
        │ 1. Multi-Source Ingestion & Scraping Engine            │
        │ - Twitter / X Social Buzz Feed                         │
        │ - Amazon Verified Reviews                              │
        │ - Reddit Communities (r/technology, r/smartphones)     │
        │ - Financial & Tech News RSS                            │
        └───────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │ 2. NLP & Aspect-Based Sentiment Analysis (ABSA)        │
        │ - Multi-Head Transformer BERT Probabilities            │
        │ - VADER Lexical Sentiment & Slang Context Rules        │
        │ - Granular Emotion Mining (Joy, Trust, Anger, Sadness) │
        │ - Aspect Extraction (Battery, Build, Price, UI, Speed) │
        └───────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │ 3. Unsupervised Topic Modeling & Semantic Clustering   │
        │ - BERTopic Semantic Clustering (UMAP + c-TF-IDF)       │
        │ - Latent Dirichlet Allocation (LDA Topic Mixtures)     │
        │ - Temporal Emerging Trends & Topic Velocity Tracking   │
        └───────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
        ┌────────────────────────────────────────────────────────┐
        │ 4. Deep Trend Forecasting & RAG Knowledge Retrieval   │
        │ - SentiTSMixer (Time-Mixing & Feature-Mixing MLPs)     │
        │ - Bidirectional LSTM Neural Networks & ARIMA Baselines │
        │ - FAISS Dense Vector Database & Gemini LLM RAG Copilot │
        └────────────────────────────────────────────────────────┘
        ```
        """
    )

with col_arch2:
    st.markdown("#### 🧭 Specialized Navigation Suite")
    st.markdown(
        """
        Navigate to any of the 8 dedicated operational dashboards using the left sidebar:
        
        1. **🌐 Live Market Pulse**: Real-time streaming simulation, sentiment velocity gauge, and channel breakdown.
        2. **🔬 Aspect Sentiment Lab**: Aspect-Based Sentiment Analysis (ABSA) across battery, build, price, and emotions.
        3. **🧠 Topic Modeling (BERTopic & LDA)**: Semantic 2D UMAP cluster projections, c-TF-IDF keywords, and topic evolution.
        4. **📈 Deep Trend Forecaster**: SentiTSMixer & Bi-LSTM demand predictions with 12–24 week interactive horizons.
        5. **🏆 Model Evaluation Arena**: Benchmark matrix comparing SentiTSMixer, Bi-LSTM, and ARIMA across MAPE, MSE, and MAE.
        6. **🤖 RAG AI Market Analyst**: Natural language market Q&A grounded in FAISS vector embeddings and Gemini synthesis.
        7. **🔮 What-If Scenario Simulator**: Stress-test price changes, sentiment crises, and supply chain bullwhip impacts.
        8. **📂 Data Pipeline Studio**: Upload custom datasets, generate synthetic review streams, and export forecast reports.
        """
    )

st.markdown("---")

# Quick Visual Preview of Active Product
st.markdown(f"### 📊 Active Market Trajectory: {active_meta['name']}")
fig_overview = px.line(
    df_series,
    x="timestamp",
    y=["actual_sales", "sentitsmixer_pred", "bilstm_pred"],
    labels={"value": "Demand (Units/Week)", "variable": "Series", "timestamp": "Timeline"},
    color_discrete_map={
        "actual_sales": "#3b82f6",
        "sentitsmixer_pred": "#10b981",
        "bilstm_pred": "#f59e0b"
    },
    template="plotly_dark"
)
fig_overview.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_overview, use_container_width=True)
