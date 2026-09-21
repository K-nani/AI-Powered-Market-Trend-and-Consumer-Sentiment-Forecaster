"""Dashboard 8: Data Pipeline Studio & ETL Ingestion."""

import streamlit as st
import pandas as pd
import numpy as np
import io

from src.ui_common import setup_page, render_sidebar, render_banner, format_metric_card
from src.data.market_data import PRODUCT_CATALOG, generate_market_time_series, get_multichannel_review_corpus
from src.engine.sentiment_analyzer import sentiment_engine


setup_page("Data Pipeline Studio", "📂")
config = render_sidebar()

meta = PRODUCT_CATALOG[config["product_id"]]
active_series = generate_market_time_series(config["product_id"])

render_banner(
    title=f"Data Pipeline Studio & Ingestion Engine",
    subtitle="Upload multi-channel consumer feedback, generate synthetic evaluation feeds, run automated NLP ETL pipelines, and export forecasts",
    badge="ETL & Data Architecture"
)

tab_upload, tab_synthetic, tab_pipeline, tab_export = st.tabs([
    "📤 Custom Data Ingestion",
    "⚡ Synthetic Stream Generator",
    "⚙️ Interactive NLP Pipeline Visualizer",
    "💾 Export Intelligence Reports"
])

with tab_upload:
    st.markdown("### 📥 Ingest Custom Customer Review Data")
    st.markdown(
        """
        Upload your raw consumer reviews as a **CSV** or **JSON** file. Required columns:
        - `text` or `review_text` (str): Text content of customer review.
        - `rating` (float 1-5): Star rating.
        - `helpfulness` (int): Number of community upvotes/likes.
        - `timestamp` (date/str): Date or datetime of review.
        - `channel` (optional): Twitter, Amazon, Reddit, News.
        """
    )

    uploaded_file = st.file_uploader("Choose a CSV or JSON file", type=["csv", "json"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                user_df = pd.read_csv(uploaded_file)
            else:
                user_df = pd.read_json(uploaded_file)

            st.success(f"Successfully loaded {len(user_df)} records from {uploaded_file.name}!")
            st.dataframe(user_df.head(5), use_container_width=True)

            if st.button("⚡ Run End-to-End Processing Pipeline"):
                with st.spinner("Executing NLP cleaning, sentiment scoring, helpfulness normalization, and aggregation..."):
                    results = []
                    for idx, row in user_df.head(200).iterrows():
                        txt = str(row.get("review_text", row.get("text", "Good product")))
                        rat = float(row.get("rating", 4.0))
                        vot = float(row.get("helpfulness", row.get("votes", 5)))
                        res = sentiment_engine.process_review_record(txt, rat, vot, engine_type="bert")
                        results.append(res)

                    proc_df = pd.DataFrame(results)
                    st.success("Pipeline executed successfully! Output feature representations:")
                    st.dataframe(proc_df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("💡 **Try with template data:** Download a sample template CSV below.")
        sample_reviews = get_multichannel_review_corpus(config["product_id"])
        sample_df = pd.DataFrame(sample_reviews)
        st.download_button(
            label="📥 Download Sample Ingestion Template (CSV)",
            data=sample_df.to_csv(index=False),
            file_name=f"template_{config['product_id']}.csv",
            mime="text/csv"
        )

with tab_synthetic:
    st.markdown("### 🎲 Synthetic Multi-Channel Data Generator")
    st.caption("Generate realistic, high-volume consumer feedback streams for any custom product or benchmark evaluation.")

    syn_col1, syn_col2, syn_col3 = st.columns(3)
    with syn_col1:
        syn_product_name = st.text_input("Product Name", value="Quantum X1 Wireless Earbuds")
    with syn_col2:
        syn_count = st.slider("Record Volume", min_value=50, max_value=500, value=150, step=50)
    with syn_col3:
        syn_bias = st.select_slider("Target Sentiment Bias", options=["Bearish Friction (Negative)", "Neutral Mixed", "Bullish Advocacy (Positive)"], value="Bullish Advocacy (Positive)")

    if st.button("✨ Generate Synthetic Ingestion Stream"):
        with st.spinner("Synthesizing multi-channel review dataset..."):
            np.random.seed(42)
            channels = ["Twitter / X", "Amazon Verified", "Reddit (r/gadgets)", "Tech Review Blog"]
            adjectives_pos = ["revolutionary", "incredible", "seamless", "exceptional", "top-tier", "flawless", "durable"]
            adjectives_neg = ["overpriced", "disappointing", "buggy", "sluggish", "poorly-built", "uncomfortable", "faulty"]

            syn_rows = []
            for i in range(syn_count):
                ch = np.random.choice(channels, p=[0.4, 0.35, 0.15, 0.1])
                is_pos = (syn_bias == "Bullish Advocacy (Positive)") or (syn_bias == "Neutral Mixed" and np.random.rand() > 0.4)
                
                if is_pos:
                    adj = np.random.choice(adjectives_pos)
                    txt = f"The {syn_product_name} is truly {adj}. Battery endurance is top-notch and setup was effortless."
                    rating = np.random.choice([4.0, 4.5, 5.0])
                    sentiment_tag = "Positive"
                else:
                    adj = np.random.choice(adjectives_neg)
                    txt = f"Experiencing {adj} build quality on the {syn_product_name}. Bluetooth drops frequently and customer support was unhelpful."
                    rating = np.random.choice([1.0, 1.5, 2.0, 2.5])
                    sentiment_tag = "Negative"

                votes = int(np.random.exponential(18))
                syn_rows.append({
                    "id": f"SYN_{i+1:04d}",
                    "product": syn_product_name,
                    "channel": ch,
                    "rating": rating,
                    "helpfulness_votes": votes,
                    "sentiment": sentiment_tag,
                    "text": txt
                })

            df_syn = pd.DataFrame(syn_rows)
            st.success(f"Generated {len(df_syn)} synthetic consumer reviews!")
            st.dataframe(df_syn.head(10), use_container_width=True)

            st.download_button(
                label="⬇️ Download Generated Synthetic Corpus (CSV)",
                data=df_syn.to_csv(index=False),
                file_name=f"synthetic_{syn_product_name.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )

with tab_pipeline:
    st.markdown("### ⚙️ End-to-End NLP ETL Pipeline Visualizer")
    st.caption("Inspect each discrete transformation applied to consumer feedback before time-series injection.")

    st.markdown(
        """
        ```
        [Raw Multi-Channel Ingest]
                   │
                   ▼  (1) Text Normalization: Regex noise removal, lowercasing, punctuation handling
        [Cleaned Canonical String]
                   │
                   ▼  (2) Tokenization & Lemmatization: WordPiece + NLTK token stream
        [Filtered Term Tokens]
                   │
                   ▼  (3) Dual Sentiment Inference: BERT Softmax + VADER Context Rules
        [Continuous Valence Score]
                   │
                   ▼  (4) Standardized Feature Scaling: Eq. 8 (0..1), Eq. 9 (1..5), Eq. 10 (Helpfulness)
        [Normalized Vectors: R, S, H, SC]
                   │
                   ▼  (5) Temporal Aggregation: Weekly Resampling & SentiTSMixer Input Prep
        [Forecasting Model Ready Tensor (T, C)]
        ```
        """
    )

with tab_export:
    st.markdown("### 💾 Export Processed Series & Forecast Outputs")
    st.caption("Download the complete historical time-series, multi-model predictions, and market signals.")

    fmt = st.radio("Choose Export Format", ["CSV", "JSON"], horizontal=True)

    if fmt == "CSV":
        csv_bytes = active_series.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Forecasting Dataset (CSV)",
            data=csv_bytes,
            file_name=f"market_forecast_{config['product_id']}.csv",
            mime="text/csv"
        )
    else:
        json_bytes = active_series.to_json(orient="records", date_format="iso").encode("utf-8")
        st.download_button(
            label="⬇️ Download Forecasting Dataset (JSON)",
            data=json_bytes,
            file_name=f"market_forecast_{config['product_id']}.json",
            mime="application/json"
        )
