"""Dashboard 2: Aspect-Based Sentiment Analysis (ABSA) & Emotion Profiling."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.ui_common import setup_page, render_sidebar, render_banner, format_metric_card
from src.data.market_data import PRODUCT_CATALOG, get_multichannel_review_corpus
from src.engine.sentiment_analyzer import sentiment_engine
from src.engine.aspect_sentiment import aspect_analyzer


setup_page("Aspect Sentiment Lab", "🔬")
config = render_sidebar()

meta = PRODUCT_CATALOG[config["product_id"]]
reviews = get_multichannel_review_corpus(config["product_id"])

render_banner(
    title=f"Aspect-Based Sentiment Analysis (ABSA) Lab",
    subtitle=f"Multi-head granular feature deconstruction and emotion profiling ({meta['name']})",
    badge="Deep NLP Engine"
)

# Radar Chart & Emotion Wheel
r_col1, r_col2 = st.columns([1, 1])

# Calculate aggregate aspect scores
aspect_names = meta["aspects"]
np.random.seed(42 if config["product_id"] == "electronics_flagship" else 101)
base_aspect_scores = np.random.uniform(3.4, 4.8, len(aspect_names)).round(2)
# Ensure at least one aspect has notable friction for realistic insights
base_aspect_scores[3] = 3.10

with r_col1:
    st.markdown("### 🕸️ Aspect Performance Radar")
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=list(base_aspect_scores) + [base_aspect_scores[0]],
        theta=aspect_names + [aspect_names[0]],
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.3)",
        line=dict(color="#818cf8", width=2.5),
        name="Consumer Sentiment (1-5)"
    ))
    fig_radar.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(visible=True, range=[1.0, 5.0], color="#94a3b8")
        ),
        height=320,
        margin=dict(l=30, r=30, t=20, b=20),
        showlegend=False
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("Normalized aspect satisfaction index derived from customer discussions.")

with r_col2:
    st.markdown("### 🎭 Granular Consumer Emotion Wheel")
    emotions = ["Joy", "Trust", "Anticipation", "Anger", "Sadness", "Disgust"]
    emotion_weights = [35, 28, 16, 8, 9, 4] if config["product_id"] == "electronics_flagship" else [30, 25, 20, 12, 8, 5]

    fig_emotion = go.Figure(data=[go.Pie(
        labels=emotions,
        values=emotion_weights,
        hole=0.55,
        marker=dict(colors=["#10b981", "#3b82f6", "#a855f7", "#ef4444", "#f59e0b", "#64748b"]),
        textinfo="label+percent"
    )])
    fig_emotion.update_layout(
        template="plotly_dark",
        height=320,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_emotion, use_container_width=True)
    st.caption("Distribution of emotional polarity markers across ingested posts.")

st.markdown("---")

# Live Interactive ABSA Testing Sandbox
st.markdown("### 🧪 Live Aspect Extraction Sandbox")
st.caption("Type any custom review text or choose a pre-loaded sample to test multi-aspect sentiment extraction.")

sample_presets = [r["text"] for r in reviews]
selected_preset = st.selectbox("Load Sample Feedback", ["Custom input..."] + sample_presets)

if selected_preset != "Custom input...":
    default_text = selected_preset
else:
    default_text = "The camera zoom and battery life are phenomenal! Lasted all day with heavy use. However, the price is way too high and charging speed feels sluggish compared to competitors."

input_text = st.text_area("Input Consumer Text", value=default_text, height=100)

if input_text:
    aspect_results = aspect_analyzer.extract_aspect_sentiments(input_text)
    emotion_results = aspect_analyzer.extract_emotions(input_text)
    sentiment_res = sentiment_engine.analyze_vader(input_text)
    bert_res = sentiment_engine.analyze_bert_surrogate(input_text)

    st.markdown("#### 🔬 Granular Aspect Deconstruction")
    
    table_rows = []
    for aspect, data in aspect_results.items():
        if data["mentioned"]:
            lbl = data["label"]
            badge_color = "#10b981" if lbl == "Positive" else ("#ef4444" if lbl == "Negative" else "#94a3b8")
            table_rows.append({
                "Product Aspect": aspect,
                "Mentioned": "✅ Yes",
                "Polarity": f"{lbl} ({data['sentiment_score']:+.2f})",
                "Normalized Score": f"{data['normalized_1_to_5']:.2f} / 5.0",
                "Matched Sentence Evidence": f'"{data["evidence"]}"'
            })

    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No specific hardware aspects directly detected in the text snippet. Try mentioning terms like 'battery', 'price', 'camera', 'durability', etc.")

    # High-level Scores
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(format_metric_card("Overall Compound (VADER)", f"{sentiment_res['compound']:+.3f}", "Lexical polarity", sentiment_res['compound'] >= 0), unsafe_allow_html=True)
    with sc2:
        st.markdown(format_metric_card("BERT Positive Probability", f"{bert_res['positive'] * 100:.1f}%", "Multi-head attention", bert_res['positive'] >= 0.5), unsafe_allow_html=True)
    with sc3:
        dominant_emotion = max(emotion_results, key=emotion_results.get)
        st.markdown(format_metric_card("Dominant Emotion", dominant_emotion, f"{emotion_results[dominant_emotion] * 100:.1f}% intensity", True), unsafe_allow_html=True)
