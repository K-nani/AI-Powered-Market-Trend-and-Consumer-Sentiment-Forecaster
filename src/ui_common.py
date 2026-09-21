"""Shared UI Components, Modern Enterprise Styling, and Navigation Controls."""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any
from src.data.market_data import PRODUCT_CATALOG


def setup_page(title: str, icon: str = "📈"):
    """Sets standard page config with wide layout and modern styling."""
    st.set_page_config(
        page_title=f"{title} | AI Market Trend & Sentiment Forecaster",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    apply_enterprise_styling()


def apply_enterprise_styling():
    """Injects high-end, modern SaaS dark aesthetic with glowing neon accents."""
    st.markdown(
        """
        <style>
        /* Import Inter Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Modern Glassmorphic Container Cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.85));
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            transition: all 0.25s ease-in-out;
            margin-bottom: 12px;
        }
        .metric-card:hover {
            border-color: rgba(99, 102, 241, 0.5);
            box-shadow: 0 12px 30px -5px rgba(99, 102, 241, 0.25);
            transform: translateY(-2px);
        }
        .metric-label {
            font-size: 0.80rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.08em;
            color: #94a3b8;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #f8fafc;
            letter-spacing: -0.02em;
        }
        .metric-sub {
            font-size: 0.82rem;
            color: #10b981;
            font-weight: 600;
            margin-top: 4px;
        }
        .metric-sub-negative {
            font-size: 0.82rem;
            color: #ef4444;
            font-weight: 600;
            margin-top: 4px;
        }

        /* Enterprise Banner Hero */
        .hero-banner {
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 70%, #020617 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.6);
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0;
            letter-spacing: -0.03em;
        }
        .hero-sub {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-top: 6px;
            margin-bottom: 14px;
        }
        .badge-pill {
            display: inline-block;
            background: rgba(99, 102, 241, 0.25);
            border: 1px solid rgba(129, 140, 248, 0.5);
            color: #e0e7ff;
            padding: 4px 14px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-right: 8px;
        }
        .badge-live {
            background: rgba(16, 185, 129, 0.25);
            border: 1px solid rgba(52, 211, 153, 0.5);
            color: #a7f3d0;
        }

        /* Live Ticker Bar */
        .ticker-wrap {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(51, 65, 85, 0.6);
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 20px;
            font-size: 0.85rem;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            overflow-x: hidden;
            white-space: nowrap;
        }
        .ticker-badge {
            background: #ef4444;
            color: #ffffff;
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.72rem;
            margin-right: 12px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_sidebar() -> Dict[str, Any]:
    """Renders persistent global configuration controls in the sidebar."""
    st.sidebar.markdown("## ⚡ Forecaster Control")

    # Global Brand/Product Selector
    catalog_keys = list(PRODUCT_CATALOG.keys())
    selected_prod = st.sidebar.selectbox(
        "Active Market / Brand Sector",
        options=catalog_keys,
        format_func=lambda x: f"{PRODUCT_CATALOG[x]['icon']} {PRODUCT_CATALOG[x]['name']}",
        index=0
    )

    # Sentiment Classification Engine
    selected_nlp = st.sidebar.radio(
        "Primary NLP Sentiment Engine",
        options=["Transformer BERT (Multi-Head)", "VADER Lexical Rule-Base", "Hybrid Ensemble (Recommended)"],
        index=2,
        help="Selects neural transformer, lexical rule-based, or hybrid ensemble classification."
    )

    # Multi-Channel Filter
    channel_filter = st.sidebar.multiselect(
        "Ingestion Channels",
        options=["Twitter / X", "Amazon Verified Reviews", "Reddit Communities", "Financial & Tech News"],
        default=["Twitter / X", "Amazon Verified Reviews", "Reddit Communities", "Financial & Tech News"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Generative AI Copilot")
    gemini_key = st.sidebar.text_input(
        "Gemini API Key (Optional)",
        type="password",
        placeholder="AIzaSy...",
        help="Optional: Enter a Gemini API Key to enable conversational RAG market synthesis."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
            <b>System:</b> AI-Powered Market Trend Forecaster<br/>
            <b>Architecture:</b> SentiTSMixer + Bi-LSTM + FAISS RAG<br/>
            <b>Status:</b> Production Active 🟢
        </div>
        """,
        unsafe_allow_html=True
    )

    return {
        "product_id": selected_prod,
        "nlp_engine": selected_nlp,
        "channels": channel_filter,
        "gemini_api_key": gemini_key
    }


def render_banner(title: str, subtitle: str, badge: str = "Enterprise AI Intelligence"):
    """Displays a standardized hero header banner."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <span class="badge-pill">{badge}</span>
            <span class="badge-pill badge-live">● Live Stream Online</span>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_ticker(items: list = None):
    """Renders real-time simulated market ticker."""
    if not items:
        items = [
            "APPLE: NSI +12.4% (Camera sensor praise surging on Reddit)",
            "TESLA: FSD v13 praise up +24% | Tailgate alignment complaints down 8%",
            "NIKE: Air Max retro drop generates 45k Twitter mentions in 2 hours",
            "SAMSUNG: Pre-orders +18% following battery longevity benchmark reveals"
        ]
    ticker_text = "  &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp;  ".join(items)
    st.markdown(
        f"""
        <div class="ticker-wrap">
            <span class="ticker-badge">LIVE PULSE</span>
            <span>{ticker_text}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_metric_card(label: str, value: str, delta: str = "", is_positive: bool = True) -> str:
    """HTML helper for high-visibility metric cards."""
    sub_class = "metric-sub" if is_positive else "metric-sub-negative"
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="{sub_class}">{delta}</div>' if delta else ''}
    </div>
    """
