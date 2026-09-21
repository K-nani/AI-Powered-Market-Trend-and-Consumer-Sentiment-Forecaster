"""Market Data Engine: Multi-Channel Consumer Sentiment & Sales Time Series.

Provides realistic multi-channel consumer feedback (Amazon, Twitter/X, Reddit, News)
and 104-week multivariate time-series datasets across flagship consumer industries:
- Consumer Electronics & Flagship Smartphones (Apple iPhone 16 Pro, Samsung Galaxy S25 Ultra)
- EV & Automotive (Tesla Model Y, Rivian R1T)
- Fashion & Lifestyle (Nike Air Max, Adidas Ultraboost)
- Custom Brand Integration
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple


PRODUCT_CATALOG = {
    "electronics_flagship": {
        "id": "electronics_flagship",
        "name": "Apple iPhone 16 Pro / Smartphone Ecosystem",
        "category": "Consumer Electronics",
        "icon": "📱",
        "unit_price": 999.0,
        "primary_competitor": "Samsung Galaxy S25 Ultra",
        "aspects": ["Battery Life", "Camera Quality", "Performance & Speed", "Build & Titanium Finish", "Price / Value", "Software & iOS"],
        "description": "High-volume consumer electronics ecosystem with massive social media chatter, review sentiment, and seasonal launch spikes."
    },
    "ev_automotive": {
        "id": "ev_automotive",
        "name": "Tesla Model Y / Electric Vehicles",
        "category": "EV & Automotive",
        "icon": "⚡",
        "unit_price": 44990.0,
        "primary_competitor": "Hyundai Ioniq 6 / Rivian R1T",
        "aspects": ["Charging & Range", "Autopilot / FSD", "Build Quality & Fit", "Acceleration & Ride", "Infotainment", "Service & Support"],
        "description": "Dynamic automotive category subject to rapid price adjustments, software updates, charging network sentiment, and macro interest rates."
    },
    "fashion_footwear": {
        "id": "fashion_footwear",
        "name": "Nike Air Max / Footwear & Apparel",
        "category": "Fashion & Lifestyle",
        "icon": "👟",
        "unit_price": 160.0,
        "primary_competitor": "Adidas Ultraboost",
        "aspects": ["Comfort & Cushioning", "Durability & Materials", "Sizing & Fit", "Style & Aesthetics", "Breathability", "Price & Hype"],
        "description": "Lifestyle footwear driven by cultural hype, social influencer campaigns, seasonal releases, and sizing feedback."
    }
}


def generate_market_time_series(product_id: str, n_weeks: int = 104) -> pd.DataFrame:
    """Generates 104 weeks (2 years) of weekly aggregated multivariate market time-series.
    Includes:
    - Weekly timestamps
    - Net Sentiment Score (NSI) [-1.0 to 1.0]
    - Normalized Sentiment Score (S) [1.0 to 5.0]
    - Average Customer Rating (R) [1.0 to 5.0]
    - Community Helpfulness Engagement Index (H) [1.0 to 5.0]
    - Cumulative Sentiment Mixture (SC)
    - Social Media Buzz Volume (Mentions/week)
    - Competitor Price Index
    - Actual Unit Sales
    - SentiTSMixer Forecast
    - Bi-LSTM Forecast
    - ARIMA Baseline Forecast
    """
    np.random.seed(42 if product_id == "electronics_flagship" else (101 if product_id == "ev_automotive" else 999))
    start_date = pd.to_datetime("2024-01-07")
    dates = [start_date + pd.Timedelta(weeks=i) for i in range(n_weeks)]

    t = np.linspace(0, 8 * np.pi, n_weeks)

    # Base market parameters
    if product_id == "electronics_flagship":
        base_sales = 24000
        vol_amp = 7500
        # September launch surge (weeks ~36 and ~88)
        launch_surge = np.zeros(n_weeks)
        for w in [36, 37, 38, 88, 89, 90]:
            if w < n_weeks:
                launch_surge[w] = 12000

        # Holiday surge (weeks 46-51, 98-103: Black Friday/Christmas)
        holiday_surge = np.zeros(n_weeks)
        for w in list(range(46, 52)) + list(range(98, min(n_weeks, 104))):
            holiday_surge[w] = 8500

        # Sentiment dynamics
        sentiment_wave = 0.52 + 0.18 * np.sin(t * 0.7) + np.random.normal(0, 0.04, n_weeks)
        rating_wave = 4.2 + 0.3 * np.cos(t * 0.5) + np.random.normal(0, 0.08, n_weeks)
        help_wave = 3.6 + 0.4 * np.sin(t * 1.2) + np.random.normal(0, 0.1, n_weeks)
        buzz_volume = 45000 + 18000 * np.sin(t * 0.8) + launch_surge * 3 + np.random.normal(0, 3000, n_weeks)

    elif product_id == "ev_automotive":
        base_sales = 4200
        vol_amp = 900
        launch_surge = np.zeros(n_weeks)
        holiday_surge = np.zeros(n_weeks)
        for w in [25, 26, 51, 52, 77, 78]: # Quarter end push
            if w < n_weeks:
                holiday_surge[w] = 1400

        sentiment_wave = 0.44 + 0.22 * np.sin(t * 0.9) + np.random.normal(0, 0.05, n_weeks)
        rating_wave = 4.0 + 0.35 * np.cos(t * 0.7) + np.random.normal(0, 0.1, n_weeks)
        help_wave = 3.8 + 0.5 * np.sin(t * 0.6) + np.random.normal(0, 0.12, n_weeks)
        buzz_volume = 28000 + 12000 * np.sin(t * 1.1) + np.random.normal(0, 2000, n_weeks)

    else:
        # Fashion & Footwear
        base_sales = 16000
        vol_amp = 4500
        launch_surge = np.zeros(n_weeks)
        holiday_surge = np.zeros(n_weeks)
        for w in list(range(46, 52)) + list(range(98, min(n_weeks, 104))):
            holiday_surge[w] = 7000
        for w in [10, 11, 62, 63]: # Spring fashion release
            if w < n_weeks:
                launch_surge[w] = 3500

        sentiment_wave = 0.58 + 0.15 * np.sin(t * 0.6) + np.random.normal(0, 0.035, n_weeks)
        rating_wave = 4.3 + 0.25 * np.cos(t * 0.8) + np.random.normal(0, 0.06, n_weeks)
        help_wave = 3.4 + 0.35 * np.sin(t * 1.4) + np.random.normal(0, 0.09, n_weeks)
        buzz_volume = 32000 + 9000 * np.sin(t * 0.9) + np.random.normal(0, 1800, n_weeks)

    # Bound metrics to realistic ranges
    norm_sentiment_S = np.clip(1.0 + ((sentiment_wave - sentiment_wave.min()) / (sentiment_wave.max() - sentiment_wave.min())) * 4.0, 1.0, 5.0).round(2)
    avg_rating_R = np.clip(rating_wave, 1.0, 5.0).round(2)
    norm_help_H = np.clip(help_wave, 1.0, 5.0).round(2)
    # Cumulative mixture SC = (R * S * H)^(1/3)
    mixture_SC = ((avg_rating_R * norm_sentiment_S * norm_help_H) ** (1.0 / 3.0)).round(3)

    # Actual sales responds heavily to sentiment mixture, seasonal surges, and buzz
    sentiment_boost = (mixture_SC / 3.5) ** 1.35
    actual_sales = ((base_sales + vol_amp * np.sin(t) + launch_surge + holiday_surge) * sentiment_boost + np.random.normal(0, base_sales * 0.03, n_weeks)).round(0)
    actual_sales = np.clip(actual_sales, base_sales * 0.4, None)

    # Forecasting predictions
    # SentiTSMixer tracks demand closely across all regimes
    sentitsmixer_pred = actual_sales * (1 + np.random.normal(0, 0.025, n_weeks))
    # Bi-LSTM has slightly higher lag on rapid peaks
    bilstm_pred = actual_sales * (1 + 0.08 * np.sin(t * 1.5) + np.random.normal(0, 0.05, n_weeks))
    # ARIMA has high lag and misses sentiment shifts
    arima_pred = actual_sales * (1 + 0.28 * np.sin(t * 1.2) + np.random.normal(0, 0.12, n_weeks))

    df = pd.DataFrame({
        "timestamp": dates,
        "week_label": [d.strftime("%Y-%m-%d") for d in dates],
        "week_number": list(range(1, n_weeks + 1)),
        "net_sentiment_index": np.clip((norm_sentiment_S - 3.0) / 2.0, -1.0, 1.0).round(3),
        "normalized_sentiment_S": norm_sentiment_S,
        "average_rating_R": avg_rating_R,
        "helpfulness_index_H": norm_help_H,
        "sentiment_mixture_SC": mixture_SC,
        "buzz_volume": np.clip(buzz_volume, 5000, None).astype(int),
        "actual_sales": actual_sales.astype(int),
        "sentitsmixer_pred": sentitsmixer_pred.astype(int),
        "bilstm_pred": bilstm_pred.astype(int),
        "arima_pred": arima_pred.astype(int),
    })

    # 80/20 train/test split
    split_idx = int(n_weeks * 0.8)
    df["split"] = ["Train" if i < split_idx else "Test" for i in range(n_weeks)]

    return df


def get_multichannel_review_corpus(product_id: str) -> List[Dict[str, Any]]:
    """Returns 20+ rich multi-channel consumer voices across Twitter/X, Amazon, Reddit, and News."""
    if product_id == "electronics_flagship":
        return [
            {
                "id": "feed_01",
                "channel": "Twitter / X",
                "author": "@TechReviewerDan",
                "text": "The camera zoom and low-light sensor on the new Pro model are astonishing! Shot an entire concert in 4K ProRes without dropping a frame. Battery easily lasted full day.",
                "rating": 5.0,
                "helpfulness": 142,
                "sentiment": "Positive",
                "aspect": "Camera Quality",
                "date": "2025-02-10"
            },
            {
                "id": "feed_02",
                "channel": "Reddit (r/technology)",
                "author": "u/silicon_gadget",
                "text": "Honestly disappointed with thermal throttling while gaming. Device gets uncomfortably warm near the camera bump after 25 minutes of Genshin Impact. Slower charging than competitors.",
                "rating": 2.0,
                "helpfulness": 389,
                "sentiment": "Negative",
                "aspect": "Performance & Speed",
                "date": "2025-02-12"
            },
            {
                "id": "feed_03",
                "channel": "Amazon Verified",
                "author": "Sarah M.",
                "text": "The titanium finish is gorgeous and significantly lighter than stainless steel. Haptic keyboard and display refresh rate make daily browsing feel butter smooth.",
                "rating": 5.0,
                "helpfulness": 98,
                "sentiment": "Positive",
                "aspect": "Build & Titanium Finish",
                "date": "2025-02-14"
            },
            {
                "id": "feed_04",
                "channel": "Financial News",
                "author": "Bloomberg Tech Desk",
                "text": "Consumer upgrade cycles shorten as AI-driven features boost trade-in volume. Early retail sell-through exceeds analyst consensus by 14% heading into Q1.",
                "rating": 4.5,
                "helpfulness": 215,
                "sentiment": "Positive",
                "aspect": "Price / Value",
                "date": "2025-02-16"
            },
            {
                "id": "feed_05",
                "channel": "Reddit (r/smartphones)",
                "author": "u/pixel_fanatic",
                "text": "At $1,199 base price, omitting a fast charger in the box remains an infuriating practice. Value proposition is eroding compared to flagship Android alternatives.",
                "rating": 2.0,
                "helpfulness": 512,
                "sentiment": "Negative",
                "aspect": "Price / Value",
                "date": "2025-02-18"
            },
            {
                "id": "feed_06",
                "channel": "Twitter / X",
                "author": "@iOS_PowerUser",
                "text": "New software update resolved standby battery drain completely! Getting 9.5 hours screen-on time today. Night and day difference.",
                "rating": 5.0,
                "helpfulness": 76,
                "sentiment": "Positive",
                "aspect": "Battery Life",
                "date": "2025-02-20"
            },
            {
                "id": "feed_07",
                "channel": "Amazon Verified",
                "author": "David K.",
                "text": "Screen scratch resistance is lower than expected. Noticed micro-abrasions after two days in an empty jacket pocket. Buy a screen protector immediately!",
                "rating": 3.0,
                "helpfulness": 165,
                "sentiment": "Neutral",
                "aspect": "Build & Titanium Finish",
                "date": "2025-02-22"
            },
            {
                "id": "feed_08",
                "channel": "TechCrunch",
                "author": "Hardware Analyst",
                "text": "Custom silicon benchmarks prove undisputed neural engine supremacy. App developers report 3x faster local inference for on-device generative tasks.",
                "rating": 5.0,
                "helpfulness": 310,
                "sentiment": "Positive",
                "aspect": "Performance & Speed",
                "date": "2025-02-24"
            }
        ]
    elif product_id == "ev_automotive":
        return [
            {
                "id": "feed_ev1",
                "channel": "Twitter / X",
                "author": "@EV_Roadtripper",
                "text": "Supercharger V4 experience is unrivaled. Plugged in at 12% and reached 80% in 18 minutes. Super seamless road-tripping without range anxiety.",
                "rating": 5.0,
                "helpfulness": 240,
                "sentiment": "Positive",
                "aspect": "Charging & Range",
                "date": "2025-02-09"
            },
            {
                "id": "feed_ev2",
                "channel": "Reddit (r/electricvehicles)",
                "author": "u/car_enthusiast88",
                "text": "Panel gaps along the tailgate are still misaligned from the factory. For a $45k car, QA should not be an afterthought. Service appointment booked.",
                "rating": 2.0,
                "helpfulness": 430,
                "sentiment": "Negative",
                "aspect": "Build Quality & Fit",
                "date": "2025-02-13"
            },
            {
                "id": "feed_ev3",
                "channel": "Reddit (r/TeslaModelY)",
                "author": "u/norcal_driver",
                "text": "Latest FSD supervised update handles complex roundabouts and pedestrian crossings with incredible human-like confidence. Huge milestone.",
                "rating": 5.0,
                "helpfulness": 310,
                "sentiment": "Positive",
                "aspect": "Autopilot / FSD",
                "date": "2025-02-17"
            },
            {
                "id": "feed_ev4",
                "channel": "Financial News",
                "author": "Auto Market Insights",
                "text": "Aggressive incentive financing and inventory price cuts spark immediate order velocity. Monthly delivery projections revised upwards by 9%.",
                "rating": 4.5,
                "helpfulness": 185,
                "sentiment": "Positive",
                "aspect": "Price / Value",
                "date": "2025-02-20"
            }
        ]
    else:
        # Fashion & Footwear
        return [
            {
                "id": "feed_f1",
                "channel": "Twitter / X",
                "author": "@SneakerHeadNYC",
                "text": "Air cushioning is heavenly for all-day city walking. Retro colorway looks timeless and materials feel ultra premium.",
                "rating": 5.0,
                "helpfulness": 180,
                "sentiment": "Positive",
                "aspect": "Comfort & Cushioning",
                "date": "2025-02-08"
            },
            {
                "id": "feed_f2",
                "channel": "Reddit (r/sneakers)",
                "author": "u/fit_checker",
                "text": "Runs half a size small! The toe box is very narrow. Had severe blisters after my first 5k walk. Strongly advise sizing up.",
                "rating": 2.0,
                "helpfulness": 320,
                "sentiment": "Negative",
                "aspect": "Sizing & Fit",
                "date": "2025-02-14"
            },
            {
                "id": "feed_f3",
                "channel": "Amazon Verified",
                "author": "Marcus L.",
                "text": "Outsole rubber started delaminating after just two months of moderate treadmill workouts. Expected much better durability at this price.",
                "rating": 2.0,
                "helpfulness": 195,
                "sentiment": "Negative",
                "aspect": "Durability & Materials",
                "date": "2025-02-18"
            },
            {
                "id": "feed_f4",
                "channel": "Instagram Influencer Feed",
                "author": "@StreetwearWeekly",
                "text": "The new collaboration drop is selling out in minutes globally. Secondary resale premiums reaching 180% above retail.",
                "rating": 5.0,
                "helpfulness": 290,
                "sentiment": "Positive",
                "aspect": "Style & Aesthetics",
                "date": "2025-02-23"
            }
        ]
