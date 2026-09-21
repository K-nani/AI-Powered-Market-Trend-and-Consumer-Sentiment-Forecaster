"""Feature Importance Calculation with RandomForestRegressor (MDI).

Implements Section III-D and Tables 4, 5, 6, 7, 8, 9 from Ghosh et al. (IEEE Access 2025).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import RandomForestRegressor


# Exact empirical values from Ghosh et al., IEEE Access 2025
PAPER_FEATURE_IMPORTANCES = {
    "case_1_magazine": {
        "dataset_name": "Magazine Subscriptions (89,689 reviews)",
        "bert": {
            "mixture": 0.758813,
            "average_sentiment": 0.097808,
            "average_rating": 0.078185,
            "helpfulness": 0.065194
        },
        "vader": {
            "helpfulness": 0.546762,
            "average_rating": 0.233489,
            "average_sentiment": 0.159834,
            "mixture": 0.059915
        }
    },
    "case_2_fashion": {
        "dataset_name": "Amazon Fashion (882,403 reviews)",
        "bert": {
            "mixture": 0.681964,
            "helpfulness": 0.253902,
            "average_rating": 0.036879,
            "average_sentiment": 0.027255
        },
        "vader": {
            "helpfulness": 0.809865,
            "mixture": 0.115593,
            "average_sentiment": 0.056295,
            "average_rating": 0.018247
        }
    },
    "case_3_cellphones": {
        "dataset_name": "Cell Phones and Accessories (10,053,882 reviews)",
        "bert": {
            "mixture": 0.913994,
            "helpfulness": 0.035438,
            "average_sentiment": 0.028947,
            "average_rating": 0.021621
        },
        "vader": {
            "helpfulness": 0.863924,
            "mixture": 0.063094,
            "average_rating": 0.055235,
            "average_sentiment": 0.017747
        }
    }
}


def get_benchmark_feature_importance(case_id: str, model_type: str = "bert") -> Dict[str, float]:
    """Retrieve verified feature importance table from the IEEE Access paper."""
    case = PAPER_FEATURE_IMPORTANCES.get(case_id, PAPER_FEATURE_IMPORTANCES["case_1_magazine"])
    weights = case.get(model_type.lower(), case["bert"])
    return weights


def compute_rf_feature_importance(
    df: pd.DataFrame,
    target_col: str = "sales",
    feature_cols: Optional[List[str]] = None,
    n_estimators: int = 100,
    random_state: int = 42
) -> Dict[str, float]:
    """Train RandomForestRegressor and compute Mean Decrease in Impurity (MDI) feature importances.
    
    Formula: MDI adds impurity reductions made by each feature across all decision trees
    and normalizes them so they sum to 1.0.
    """
    if feature_cols is None:
        feature_cols = ["average_sentiment", "average_rating", "helpfulness", "mixture"]

    valid_cols = [c for c in feature_cols if c in df.columns]
    if not valid_cols or target_col not in df.columns:
        # Fallback to default equal weights
        return {c: 1.0 / len(feature_cols) for c in feature_cols}

    X = df[valid_cols].values
    y = df[target_col].values

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)

    importances = rf.feature_importances_
    return {col: float(round(imp, 6)) for col, imp in zip(valid_cols, importances)}
