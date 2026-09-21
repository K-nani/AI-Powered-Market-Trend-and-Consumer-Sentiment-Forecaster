"""Baseline Models and Standard Forecasting Error Metrics.

Implements Equations (5), (6), (7) and comparative baselines:
ARIMA, Modified ARIMA, SARIMA, Modified SARIMA, LSTM, Modified LSTM,
SentiTSMixer (BERT), and SentiTSMixer (VADER).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Exact empirical benchmark tables from Ghosh et al. (IEEE Access 2025)
BENCHMARK_ERROR_TABLES = {
    "case_1_magazine": {
        "dataset_name": "Magazine Subscriptions",
        "records_count": 89689,
        "models": {
            "ARIMA [24], [26]": {"MAPE": 248.055, "MSE": 478.668, "MSLE": 0.855, "type": "Statistical"},
            "ARIMA modified Ghosh et al. [17]": {"MAPE": 157.437, "MSE": 262.270, "MSLE": 0.568, "type": "Hybrid"},
            "SARIMA [27], [28]": {"MAPE": 249.440, "MSE": 484.929, "MSLE": 5.256, "type": "Statistical"},
            "SARIMA modified Ghosh et al. [17]": {"MAPE": 158.187, "MSE": 262.657, "MSLE": 0.570, "type": "Hybrid"},
            "LSTM [21], [25]": {"MAPE": 166.038, "MSE": 387.686, "MSLE": 0.894, "type": "Deep Learning"},
            "LSTM modified Ghosh et al. [17]": {"MAPE": 119.910, "MSE": 262.657, "MSLE": 0.570, "type": "Hybrid DL"},
            "Proposed SentiTSMixer (BERT)": {"MAPE": 18.10074, "MSE": 0.14682, "MSLE": 0.10302, "type": "Proposed SentiTSMixer"},
            "Proposed SentiTSMixer (VADER)": {"MAPE": 30.57934, "MSE": 0.28486, "MSLE": 0.20166, "type": "Proposed SentiTSMixer"}
        }
    },
    "case_2_fashion": {
        "dataset_name": "Amazon Fashion",
        "records_count": 882403,
        "models": {
            "ARIMA [24], [26]": {"MAPE": 271.582, "MSE": 333639.921, "MSLE": 1.353, "type": "Statistical"},
            "ARIMA modified Ghosh et al. [17]": {"MAPE": 160.546, "MSE": 129397.147, "MSLE": 0.802, "type": "Hybrid"},
            "SARIMA [27], [28]": {"MAPE": 271.460, "MSE": 333329.740, "MSLE": 11.895, "type": "Statistical"},
            "SARIMA modified Ghosh et al. [17]": {"MAPE": 160.478, "MSE": 129338.851, "MSLE": 0.802, "type": "Hybrid"},
            "LSTM [21], [25]": {"MAPE": 135.812, "MSE": 180939.624, "MSLE": 1.073, "type": "Deep Learning"},
            "LSTM modified Ghosh et al. [17]": {"MAPE": 95.754, "MSE": 129338.851, "MSLE": 0.802, "type": "Hybrid DL"},
            "Proposed SentiTSMixer (BERT)": {"MAPE": 19.53353, "MSE": 0.05999, "MSLE": 0.04969, "type": "Proposed SentiTSMixer"},
            "Proposed SentiTSMixer (VADER)": {"MAPE": 28.18420, "MSE": 0.12588, "MSLE": 0.09872, "type": "Proposed SentiTSMixer"}
        }
    },
    "case_3_cellphones": {
        "dataset_name": "Cell Phones and Accessories",
        "records_count": 10053882,
        "models": {
            "ARIMA [24], [26]": {"MAPE": 932.518, "MSE": 11748637.320, "MSLE": 0.847, "type": "Statistical"},
            "ARIMA modified Ghosh et al. [17]": {"MAPE": 614.897, "MSE": 6105044.375, "MSLE": 0.553, "type": "Hybrid"},
            "SARIMA [27], [28]": {"MAPE": 934.209, "MSE": 11809204.845, "MSLE": 15.063, "type": "Statistical"},
            "SARIMA modified Ghosh et al. [17]": {"MAPE": 615.944, "MSE": 6105110.817, "MSLE": 0.553, "type": "Hybrid"},
            "LSTM [21], [25]": {"MAPE": 647.431, "MSE": 11429262.803, "MSLE": 0.897, "type": "Deep Learning"},
            "LSTM modified Ghosh et al. [17]": {"MAPE": 439.436, "MSE": 6105110.817, "MSLE": 0.553, "type": "Hybrid DL"},
            "Proposed SentiTSMixer (BERT)": {"MAPE": 18.21725, "MSE": 0.04864, "MSLE": 0.03906, "type": "Proposed SentiTSMixer"},
            "Proposed SentiTSMixer (VADER)": {"MAPE": 12.99038, "MSE": 0.02822, "MSLE": 0.02271, "type": "Proposed SentiTSMixer"}
        }
    }
}


def calculate_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Equation (5): MSE = (1/n) * sum((Y_t - Y_hat_t)^2)"""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    return float(np.mean((actual - predicted) ** 2))


def calculate_msle(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Equation (6): MSLE = (1/n) * sum((log(Y_t + 1) - log(Y_hat_t + 1))^2)"""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    # Clip negative predictions to avoid invalid logs
    pred_safe = np.clip(predicted, 0.0, None)
    act_safe = np.clip(actual, 0.0, None)
    return float(np.mean((np.log1p(act_safe) - np.log1p(pred_safe)) ** 2))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Equation (7): MAPE = (1/n) * sum(|Actual_t - Forecast_t| / |Actual_t|) * 100"""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    # Protect against division by zero
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100.0)


def compute_all_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate MSE, MSLE, and MAPE together."""
    return {
        "MSE": round(calculate_mse(actual, predicted), 6),
        "MSLE": round(calculate_msle(actual, predicted), 6),
        "MAPE": round(calculate_mape(actual, predicted), 4)
    }


def fit_arima_forecast(series: np.ndarray, steps: int = 4, order: Tuple[int, int, int] = (1, 1, 1)) -> np.ndarray:
    """Fit traditional ARIMA model and generate out-of-sample forecast."""
    try:
        model = ARIMA(series, order=order)
        res = model.fit()
        return res.forecast(steps=steps)
    except Exception:
        # Fallback linear trend
        diff = np.diff(series)
        slope = np.mean(diff) if len(diff) > 0 else 0.0
        return series[-1] + np.arange(1, steps + 1) * slope


def fit_sarima_forecast(
    series: np.ndarray,
    steps: int = 4,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 0, 4)
) -> np.ndarray:
    """Fit SARIMA model with seasonal components."""
    try:
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
        res = model.fit(disp=False)
        return res.forecast(steps=steps)
    except Exception:
        return fit_arima_forecast(series, steps=steps, order=order)
