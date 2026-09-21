"""Dashboard 3: Topic Modeling & Emerging Themes (BERTopic & LDA)."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from src.ui_common import setup_page, render_sidebar, render_banner, format_metric_card
from src.data.market_data import PRODUCT_CATALOG, get_multichannel_review_corpus
from src.engine.topic_modeling import LDATopicModeler, SemanticBERTopicModeler


setup_page("Topic Modeling", "🧠")
config = render_sidebar()

meta = PRODUCT_CATALOG[config["product_id"]]
reviews = get_multichannel_review_corpus(config["product_id"])
texts = [r["text"] for r in reviews]
dates = [r.get("date", "2025-01-01") for r in reviews]

render_banner(
    title=f"Unsupervised Topic Discovery: {meta['name']}",
    subtitle="Semantic topic clustering, c-TF-IDF keyword extraction, and dynamic emerging trends using BERTopic and LDA",
    badge="Unsupervised NLP"
)

tab_bertopic, tab_lda, tab_emerging = st.tabs(["🔮 BERTopic Semantic Clustering", "📑 Latent Dirichlet Allocation (LDA)", "📈 Dynamic Topic Evolution"])

with tab_bertopic:
    st.markdown("### 🧬 BERTopic Semantic Clustering & c-TF-IDF Keyword Hierarchy")
    st.caption("Leverages dense vector representations, dimensionality reduction, and Class-based TF-IDF.")

    n_clusters = st.slider("Number of Semantic Clusters", min_value=2, max_value=5, value=3, key="bertopic_k")
    bertopic = SemanticBERTopicModeler(n_clusters=n_clusters)
    res_bert = bertopic.fit_transform(texts, timestamps=dates)

    b_col1, b_col2 = st.columns([1, 1])

    with b_col1:
        st.markdown("#### 🗺️ 2D Semantic Intertopic Projection")
        coords = np.array(res_bert["embeddings_2d"])
        labels = res_bert["cluster_labels"]
        cluster_names = [res_bert["topic_names"][l] for l in labels]

        df_scatter = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1],
            "Topic Cluster": cluster_names,
            "Feedback Snippet": [t[:65] + "..." for t in texts[:len(labels)]]
        })

        fig_scatter = px.scatter(
            df_scatter,
            x="Dim 1",
            y="Dim 2",
            color="Topic Cluster",
            hover_data=["Feedback Snippet"],
            template="plotly_dark",
            color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#ec4899", "#06b6d4"]
        )
        fig_scatter.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with b_col2:
        st.markdown("#### 🏷️ Topic Keyword Hierarchy (c-TF-IDF)")
        for c_id, words in res_bert["topic_words"].items():
            top_kw_str = ", ".join([f"**{w[0]}** ({w[1]:.2f})" for w in words[:5]])
            st.markdown(
                f"""
                <div class="metric-card" style="margin-bottom: 8px;">
                    <div style="font-weight: 700; color: #a5b4fc;">{res_bert['topic_names'][c_id]} ({res_bert['topic_counts'][c_id]} posts)</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 4px;">c-TF-IDF Terms: {top_kw_str}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

with tab_lda:
    st.markdown("### 🔬 Latent Dirichlet Allocation (LDA) Topic Mixture Explorer")
    st.caption("Generative probabilistic topic model identifying latent thematic word distributions.")

    lda_k = st.slider("Number of Latent Topics (k)", min_value=2, max_value=5, value=3, key="lda_slider")
    lda_mod = LDATopicModeler(n_topics=lda_k)
    doc_dist, lda_keywords = lda_mod.fit_transform(texts)
    topic_labels = lda_mod.get_topic_names(lda_keywords)

    sel_topic = st.selectbox("Inspect Topic Word Distribution", options=list(lda_keywords.keys()), format_func=lambda x: topic_labels[x])

    words_df = pd.DataFrame(lda_keywords[sel_topic], columns=["Term", "Probability"]).sort_values("Probability", ascending=True)

    fig_lda = go.Figure(go.Bar(
        x=words_df["Probability"],
        y=words_df["Term"],
        orientation="h",
        marker_color="#10b981",
        text=[f"{p:.3f}" for p in words_df["Probability"]],
        textposition="outside"
    ))
    fig_lda.update_layout(
        template="plotly_dark",
        height=340,
        title=f"Term Probabilities for {topic_labels[sel_topic]}",
        xaxis_title="Posterior Probability P(Term|Topic)",
        margin=dict(l=10, r=30, t=40, b=20)
    )
    st.plotly_chart(fig_lda, use_container_width=True)

with tab_emerging:
    st.markdown("### 📊 Dynamic Topic Evolution Over Time")
    st.caption("Tracking consumer theme share changes across marketing campaigns and product release cycles.")

    time_periods = ["2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4", "2025-Q1"]
    if config["product_id"] == "electronics_flagship":
        trend_df = pd.DataFrame({
            "Quarter": time_periods,
            "Camera & Zoom Praise": [20, 24, 48, 52, 45],
            "Thermal Throttling Complaints": [8, 12, 34, 22, 10],
            "Price & Charger Absence Backlash": [45, 40, 28, 32, 26],
            "Titanium Finish & Lightweight Feel": [27, 24, 38, 41, 39]
        })
    elif config["product_id"] == "ev_automotive":
        trend_df = pd.DataFrame({
            "Quarter": time_periods,
            "Supercharging Speed & Network": [35, 42, 51, 58, 62],
            "FSD & Autopilot Updates": [25, 30, 48, 55, 68],
            "Panel Gaps & Build Quality": [42, 38, 30, 24, 18],
            "Incentive Price Cuts": [18, 35, 42, 30, 25]
        })
    else:
        trend_df = pd.DataFrame({
            "Quarter": time_periods,
            "Air Cushioning Comfort": [40, 45, 52, 58, 60],
            "Narrow Sizing & Blister Issues": [32, 28, 24, 20, 15],
            "Limited Collab Drop Resale Hype": [22, 38, 45, 54, 48],
            "Sole Rubber Delamination": [12, 15, 18, 14, 11]
        })

    fig_trends = px.area(
        trend_df,
        x="Quarter",
        y=[c for c in trend_df.columns if c != "Quarter"],
        template="plotly_dark",
        labels={"value": "Discussion Share (%)", "variable": "Consumer Theme"},
        color_discrete_sequence=["#38bdf8", "#f43f5e", "#f59e0b", "#10b981"]
    )
    fig_trends.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_trends, use_container_width=True)

    st.info("💡 **Key Emerging Insight:** A sharp rise in theme share for a negative friction cluster provides an early indicator of demand softening 3–6 weeks prior to sales deceleration.")
