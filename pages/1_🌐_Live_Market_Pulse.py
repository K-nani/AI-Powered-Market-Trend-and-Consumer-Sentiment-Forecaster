"""Dashboard 1: Live Market Pulse & Real-Time Stream."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.ui_common import setup_page, render_sidebar, render_banner, render_ticker, format_metric_card
from src.data.market_data import PRODUCT_CATALOG, generate_market_time_series, get_multichannel_review_corpus


setup_page("Live Market Pulse", "🌐")
config = render_sidebar()

meta = PRODUCT_CATALOG[config["product_id"]]
df = generate_market_time_series(config["product_id"])
reviews = get_multichannel_review_corpus(config["product_id"])

render_ticker()

render_banner(
    title=f"Live Market Pulse: {meta['name']}",
    subtitle="Continuous streaming monitoring across Twitter/X, Amazon Verified Reviews, Reddit, and Financial News",
    badge=f"Sector: {meta['category']}"
)

# High-Frequency Metric Strip
k1, k2, k3, k4 = st.columns(4)
cur_nsi = df["net_sentiment_index"].iloc[-1]
avg_buzz = df["buzz_volume"].iloc[-1]

with k1:
    st.markdown(format_metric_card("Live Sentiment Index (NSI)", f"{cur_nsi:+.2f} / 1.0", "Positive Bullish Velocity", cur_nsi >= 0), unsafe_allow_html=True)
with k2:
    st.markdown(format_metric_card("Weekly Social Buzz", f"{avg_buzz:,} Mentions", "+14.2% vs 4-Wk Moving Avg", True), unsafe_allow_html=True)
with k3:
    st.markdown(format_metric_card("Community Helpfulness Ratio", "87.4%", "Verified engagement upvotes", True), unsafe_allow_html=True)
with k4:
    st.markdown(format_metric_card("Sentiment-Driven Demand Drift", "+11.8%", "Predicted upward order lift", True), unsafe_allow_html=True)

st.markdown("---")

# Visual Pulse: Sentiment Gauge & Channel Breakdown
g_col1, g_col2 = st.columns([1, 1])

with g_col1:
    st.markdown("### 🧭 Net Sentiment Velocity Gauge")
    
    # Plotly Gauge Indicator
    gauge_val = (cur_nsi + 1.0) * 50.0 # Map -1..1 to 0..100
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gauge_val,
        delta={"reference": 50, "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": "#6366f1"},
            "bgcolor": "#0f172a",
            "borderwidth": 2,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 35], "color": "rgba(239, 68, 68, 0.4)"},
                {"range": [35, 65], "color": "rgba(234, 179, 8, 0.4)"},
                {"range": [65, 100], "color": "rgba(16, 185, 129, 0.4)"}
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.75,
                "value": gauge_val
            }
        }
    ))
    fig_gauge.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption("0–35: Bearish Friction | 35–65: Neutral Equilibrium | 65–100: Strong Bullish Advocacy")

with g_col2:
    st.markdown("### 📡 Ingestion Channel Share")
    channel_counts = {"Twitter / X": 45, "Amazon Verified": 28, "Reddit": 18, "Financial & Tech News": 9}
    fig_channel = go.Figure(data=[go.Pie(
        labels=list(channel_counts.keys()),
        values=list(channel_counts.values()),
        hole=0.6,
        marker=dict(colors=["#38bdf8", "#f59e0b", "#f97316", "#a855f7"]),
        textinfo="label+percent"
    )])
    fig_channel.update_layout(
        template="plotly_dark",
        height=280,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_channel, use_container_width=True)
    st.caption("Real-time distributed multi-channel consumer footprint.")

st.markdown("---")

# Live Feed Stream Simulation
st.markdown("### ⚡ Live Incoming Consumer Feed Stream")
st.caption("Simulated real-time ingest with automatic sentiment classification and aspect tagging.")

# Filter by channel if selected in sidebar
selected_channels = config.get("channels", ["Twitter / X", "Amazon Verified Reviews", "Reddit Communities", "Financial & Tech News"])

for item in reviews:
    channel_name = item.get("channel", "Twitter / X")
    sent = item.get("sentiment", "Neutral")
    rating = item.get("rating", 4.0)
    aspect = item.get("aspect", "General")
    votes = item.get("helpfulness", 0)
    
    badge_bg = "rgba(16, 185, 129, 0.2)" if sent == "Positive" else ("rgba(239, 68, 68, 0.2)" if sent == "Negative" else "rgba(148, 163, 184, 0.2)")
    badge_col = "#34d399" if sent == "Positive" else ("#f87171" if sent == "Negative" else "#cbd5e1")

    st.markdown(
        f"""
        <div class="metric-card" style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div>
                    <span style="background: rgba(99, 102, 241, 0.25); color: #c7d2fe; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem;">{channel_name}</span>
                    <span style="background: {badge_bg}; color: {badge_col}; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; margin-left: 6px;">{sent}</span>
                    <span style="background: rgba(245, 158, 11, 0.2); color: #fcd34d; padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; margin-left: 6px;">🏷️ {aspect}</span>
                    <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 8px;">★ {rating}/5.0</span>
                </div>
                <div style="color: #64748b; font-size: 0.8rem;">
                    👍 {votes} votes • {item.get('date', 'Just now')}
                </div>
            </div>
            <div style="font-size: 0.94rem; color: #f8fafc; line-height: 1.5; margin-top: 6px;">
                "{item.get('text', '')}"
            </div>
            <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">
                Author: {item.get('author', 'Anonymous')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
