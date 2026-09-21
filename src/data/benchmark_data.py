"""Benchmark Datasets and Historical Time Series Series.

Generates realistic weekly aggregated series and customer review corpora
matching the experimental setups and Figures 2, 3, 4, 5, 6, 7 of Ghosh et al. (IEEE Access 2025).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


CASE_STUDY_METADATA = {
    "case_1_magazine": {
        "id": "case_1_magazine",
        "name": "Magazine Subscriptions",
        "scale": "Small Dataset",
        "total_reviews": 89689,
        "date_range": "2017-09 to 2018-09",
        "description": "Subscription publications and periodicals. Exhibits notable seasonality and sentiment shift in late summer 2018.",
        "icon": "📰",
        "best_model": "Proposed SentiTSMixer (BERT)",
        "paper_table_error": "Table 1",
        "paper_feature_table": "Table 4 (BERT) / Table 5 (VADER)"
    },
    "case_2_fashion": {
        "id": "case_2_fashion",
        "name": "Amazon Fashion",
        "scale": "Medium Dataset",
        "total_reviews": 882403,
        "date_range": "2017-09 to 2018-09",
        "description": "Apparel, footwear, and accessories. Characterized by high review volume, seasonal trend shifts, and size/fit sentiment variance.",
        "icon": "👗",
        "best_model": "Proposed SentiTSMixer (BERT)",
        "paper_table_error": "Table 2",
        "paper_feature_table": "Table 6 (BERT) / Table 7 (VADER)"
    },
    "case_3_cellphones": {
        "id": "case_3_cellphones",
        "name": "Cell Phones and Accessories",
        "scale": "Large Dataset",
        "total_reviews": 10053882,
        "date_range": "2017-09 to 2018-09",
        "description": "High-volume consumer electronics, chargers, cases, and devices. High helpfulness impact and stable generalization.",
        "icon": "📱",
        "best_model": "Proposed SentiTSMixer (VADER: MAPE 12.99%)",
        "paper_table_error": "Table 3",
        "paper_feature_table": "Table 8 (BERT) / Table 9 (VADER)"
    }
}


def generate_weekly_series(case_id: str, n_weeks: int = 54) -> pd.DataFrame:
    """Generates weekly aggregated time series matching the IEEE Access 2025 paper.
    Columns:
    - timestamp
    - average_sentiment
    - average_rating
    - helpfulness
    - mixture
    - actual_sales
    - ts_mixer_bert_pred
    - ts_mixer_vader_pred
    - arima_pred
    - lstm_pred
    """
    np.random.seed(42 if case_id == "case_1_magazine" else (101 if case_id == "case_2_fashion" else 777))
    start_date = pd.to_datetime("2017-09-03")
    dates = [start_date + pd.Timedelta(weeks=i) for i in range(n_weeks)]

    t = np.linspace(0, 4 * np.pi, n_weeks)

    if case_id == "case_1_magazine":
        # Magazine Subscriptions (Figures 2 & 3 in paper)
        # Normalized range ~0.15 - 0.45, with noticeable deviation in 2018-07 to 2018-09
        base_sent = 0.28 + 0.08 * np.sin(t) + np.random.normal(0, 0.03, n_weeks)
        base_rating = 0.38 + 0.05 * np.cos(t * 0.8) + np.random.normal(0, 0.02, n_weeks)
        base_help = 0.16 + 0.04 * np.sin(t * 1.5) + np.random.normal(0, 0.015, n_weeks)
        base_mix = (base_sent * 0.4 + base_rating * 0.4 + base_help * 0.2) + np.random.normal(0, 0.01, n_weeks)

        # Sales volume
        actual_sales = 1200 + 400 * base_mix * 5 + 150 * np.sin(t) + np.random.normal(0, 30, n_weeks)

        # Predictions
        # SentiTSMixer tracks closely, with small divergence near end (weeks 44-52: July to Sept 2018)
        divergence = np.zeros(n_weeks)
        divergence[44:] = 0.05 * np.linspace(0.2, 1.0, len(divergence[44:]))

        bert_pred_mix = base_mix + np.random.normal(0, 0.012, n_weeks) + divergence * 0.5
        vader_pred_mix = base_mix + np.random.normal(0, 0.018, n_weeks) + divergence * 0.8

        bert_sales_pred = actual_sales * (1 + np.random.normal(0, 0.025, n_weeks) + divergence * 0.3)
        vader_sales_pred = actual_sales * (1 + np.random.normal(0, 0.045, n_weeks) + divergence * 0.5)

        # ARIMA & LSTM have much higher deviations
        arima_sales_pred = actual_sales * (1 + 0.45 * np.sin(t * 2) + np.random.normal(0, 0.15, n_weeks))
        lstm_sales_pred = actual_sales * (1 + 0.30 * np.cos(t * 1.2) + np.random.normal(0, 0.10, n_weeks))

    elif case_id == "case_2_fashion":
        # Amazon Fashion (Figures 4 & 5 in paper)
        # Much closer fit across all periods due to 882k reviews
        base_sent = 0.24 + 0.06 * np.sin(t * 1.2) + np.random.normal(0, 0.015, n_weeks)
        base_rating = 0.34 + 0.04 * np.cos(t) + np.random.normal(0, 0.015, n_weeks)
        base_help = 0.12 + 0.03 * np.sin(t * 0.9) + np.random.normal(0, 0.01, n_weeks)
        base_mix = (base_sent * 0.3 + base_rating * 0.3 + base_help * 0.4) + np.random.normal(0, 0.008, n_weeks)

        actual_sales = 8500 + 2200 * base_mix * 5 + 600 * np.sin(t * 1.4) + np.random.normal(0, 90, n_weeks)

        bert_pred_mix = base_mix + np.random.normal(0, 0.008, n_weeks)
        vader_pred_mix = base_mix + np.random.normal(0, 0.012, n_weeks)

        bert_sales_pred = actual_sales * (1 + np.random.normal(0, 0.022, n_weeks))
        vader_sales_pred = actual_sales * (1 + np.random.normal(0, 0.035, n_weeks))

        arima_sales_pred = actual_sales * (1 + 0.55 * np.sin(t) + np.random.normal(0, 0.20, n_weeks))
        lstm_sales_pred = actual_sales * (1 + 0.28 * np.sin(t * 1.1) + np.random.normal(0, 0.12, n_weeks))

    else:
        # Cell Phones & Accessories (Figures 6 & 7 in paper, 10M+ reviews)
        # High stability, lowest MSE
        base_sent = 0.31 + 0.03 * np.sin(t) + np.random.normal(0, 0.01, n_weeks)
        base_rating = 0.40 + 0.02 * np.cos(t) + np.random.normal(0, 0.008, n_weeks)
        base_help = 0.14 + 0.025 * np.sin(t * 1.1) + np.random.normal(0, 0.008, n_weeks)
        base_mix = (base_sent * 0.35 + base_rating * 0.35 + base_help * 0.3) + np.random.normal(0, 0.005, n_weeks)

        actual_sales = 45000 + 8000 * base_mix * 5 + 1800 * np.sin(t * 1.2) + np.random.normal(0, 250, n_weeks)

        bert_pred_mix = base_mix + np.random.normal(0, 0.005, n_weeks)
        vader_pred_mix = base_mix + np.random.normal(0, 0.006, n_weeks)

        bert_sales_pred = actual_sales * (1 + np.random.normal(0, 0.016, n_weeks))
        vader_sales_pred = actual_sales * (1 + np.random.normal(0, 0.012, n_weeks))

        arima_sales_pred = actual_sales * (1 + 0.85 * np.sin(t) + np.random.normal(0, 0.25, n_weeks))
        lstm_sales_pred = actual_sales * (1 + 0.42 * np.sin(t * 0.8) + np.random.normal(0, 0.15, n_weeks))

    df = pd.DataFrame({
        "timestamp": dates,
        "week_label": [d.strftime("%Y-%m-%d") for d in dates],
        "average_sentiment": np.clip(base_sent, 0.05, 0.60).round(4),
        "average_rating": np.clip(base_rating, 0.10, 0.65).round(4),
        "helpfulness": np.clip(base_help, 0.05, 0.45).round(4),
        "mixture": np.clip(base_mix, 0.05, 0.50).round(4),
        "actual_sales": actual_sales.round(1),
        "ts_mixer_bert_pred": bert_sales_pred.round(1),
        "ts_mixer_vader_pred": vader_sales_pred.round(1),
        "arima_pred": arima_sales_pred.round(1),
        "lstm_pred": lstm_sales_pred.round(1),
        "ts_mixer_bert_mix_pred": np.clip(bert_pred_mix, 0.05, 0.50).round(4),
        "ts_mixer_vader_mix_pred": np.clip(vader_pred_mix, 0.05, 0.50).round(4),
    })

    # Add 80/20 train/test split indicator (initial 80% train, remaining 20% test as stated in paper Section V)
    split_idx = int(len(df) * 0.8)
    df["split"] = ["Train" if i < split_idx else "Test" for i in range(len(df))]

    return df


def get_domain_sample_reviews(case_id: str) -> List[Dict[str, Any]]:
    """Generates authentic customer reviews for the domain, used in Topic Modeling and RAG vector store."""
    if case_id == "case_1_magazine":
        return [
            {
                "id": "rev_m1",
                "text": "The articles are insightful and cover in-depth scientific breakthroughs every single month. Delivery has been prompt.",
                "rating": 5.0,
                "helpfulness": 34,
                "sentiment": "Positive",
                "date": "2018-03-12"
            },
            {
                "id": "rev_m2",
                "text": "Great photography and journalism. Love reading it over morning coffee. A must-have subscription.",
                "rating": 5.0,
                "helpfulness": 21,
                "sentiment": "Positive",
                "date": "2018-04-05"
            },
            {
                "id": "rev_m3",
                "text": "Missed two issues in July and customer support was completely unresponsive. Very frustrated with delivery.",
                "rating": 1.0,
                "helpfulness": 68,
                "sentiment": "Negative",
                "date": "2018-07-19"
            },
            {
                "id": "rev_m4",
                "text": "The renewal price doubled without any notice. Content is decent, but billing practice feels predatory.",
                "rating": 2.0,
                "helpfulness": 45,
                "sentiment": "Negative",
                "date": "2018-08-02"
            },
            {
                "id": "rev_m5",
                "text": "Good technical magazine. Some digital app sync glitches, but print edition is always top quality.",
                "rating": 4.0,
                "helpfulness": 12,
                "sentiment": "Positive",
                "date": "2018-05-22"
            },
            {
                "id": "rev_m6",
                "text": "Too many full-page advertisements now compared to editorial articles. Quality has degraded recently.",
                "rating": 2.0,
                "helpfulness": 54,
                "sentiment": "Negative",
                "date": "2018-08-15"
            },
            {
                "id": "rev_m7",
                "text": "Essential reading for educators and students. Arrived in pristine condition each month.",
                "rating": 5.0,
                "helpfulness": 19,
                "sentiment": "Positive",
                "date": "2018-02-14"
            },
            {
                "id": "rev_m8",
                "text": "Digital edition tablet layout is superb. Easy bookmarking and interactive charts are fantastic.",
                "rating": 5.0,
                "helpfulness": 15,
                "sentiment": "Positive",
                "date": "2018-06-11"
            }
        ]
    elif case_id == "case_2_fashion":
        return [
            {
                "id": "rev_f1",
                "text": "The fabric is ultra-soft breathable cotton, stitching is sturdy, and sizing runs true to measurement chart.",
                "rating": 5.0,
                "helpfulness": 78,
                "sentiment": "Positive",
                "date": "2018-02-11"
            },
            {
                "id": "rev_f2",
                "text": "Color was completely different from the catalog photo. Looked dull gray instead of deep navy. Returned it.",
                "rating": 2.0,
                "helpfulness": 52,
                "sentiment": "Negative",
                "date": "2018-05-18"
            },
            {
                "id": "rev_f3",
                "text": "Shrank two full sizes after the very first cold wash. Cheap material and terrible durability.",
                "rating": 1.0,
                "helpfulness": 112,
                "sentiment": "Negative",
                "date": "2018-06-25"
            },
            {
                "id": "rev_f4",
                "text": "Got compliments all day at work! Very flattering cut and comfortable waistband. Will purchase more colors.",
                "rating": 5.0,
                "helpfulness": 43,
                "sentiment": "Positive",
                "date": "2018-03-30"
            },
            {
                "id": "rev_f5",
                "text": "Zipper got jammed on day three. Jacket looks stylish but hardware is low grade.",
                "rating": 2.0,
                "helpfulness": 39,
                "sentiment": "Negative",
                "date": "2018-07-04"
            },
            {
                "id": "rev_f6",
                "text": "Affordable fast fashion essentials. Good for a season, comfortable fit for everyday wear.",
                "rating": 4.0,
                "helpfulness": 16,
                "sentiment": "Positive",
                "date": "2018-04-12"
            },
            {
                "id": "rev_f7",
                "text": "Fit is tight around shoulders, order one size up if you have broader posture.",
                "rating": 3.0,
                "helpfulness": 65,
                "sentiment": "Neutral",
                "date": "2018-01-20"
            },
            {
                "id": "rev_f8",
                "text": "Premium feel without the luxury price tag. Hemline is cleanly finished and pockets are deep.",
                "rating": 5.0,
                "helpfulness": 29,
                "sentiment": "Positive",
                "date": "2018-08-19"
            }
        ]
    else:
        # Cell Phones and Accessories
        return [
            {
                "id": "rev_c1",
                "text": "Charging speed is lightning fast! Tested with USB-C power delivery meter and hit 65W consistently without overheating.",
                "rating": 5.0,
                "helpfulness": 145,
                "sentiment": "Positive",
                "date": "2018-02-15"
            },
            {
                "id": "rev_c2",
                "text": "Screen protector shattered inside my pocket within 48 hours without dropping. Completely brittle glass.",
                "rating": 1.0,
                "helpfulness": 89,
                "sentiment": "Negative",
                "date": "2018-04-10"
            },
            {
                "id": "rev_c3",
                "text": "MagSafe magnet grip is rock solid in my car mount. Sleek profile and tactile volume buttons.",
                "rating": 5.0,
                "helpfulness": 98,
                "sentiment": "Positive",
                "date": "2018-05-02"
            },
            {
                "id": "rev_c4",
                "text": "Bluetooth drops connection whenever I put the phone in my back pocket. Terrible antenna range.",
                "rating": 2.0,
                "helpfulness": 74,
                "sentiment": "Negative",
                "date": "2018-06-19"
            },
            {
                "id": "rev_c5",
                "text": "Battery life lasts nearly two full days of heavy usage. Camera low-light performance is jaw-dropping.",
                "rating": 5.0,
                "helpfulness": 220,
                "sentiment": "Positive",
                "date": "2018-03-28"
            },
            {
                "id": "rev_c6",
                "text": "Earbuds audio is clear for calls, active noise cancellation is decent for the price point.",
                "rating": 4.0,
                "helpfulness": 31,
                "sentiment": "Positive",
                "date": "2018-07-14"
            },
            {
                "id": "rev_c7",
                "text": "Phone case started yellowing around edges after three weeks in sunlight. Not anti-yellowing as advertised.",
                "rating": 2.0,
                "helpfulness": 63,
                "sentiment": "Negative",
                "date": "2018-08-08"
            },
            {
                "id": "rev_c8",
                "text": "Heavy duty drop protection saved my phone from a 6-foot fall onto concrete. Raised bezels protected camera glass.",
                "rating": 5.0,
                "helpfulness": 105,
                "sentiment": "Positive",
                "date": "2018-01-19"
            }
        ]
