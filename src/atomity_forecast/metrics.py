"""
Forecast evaluation metrics for Atomity Cloud Workload Forecasting.
"""
import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def weighted_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total_y = np.sum(y_true)
    if total_y == 0:
        return float(np.nan)
    return float((np.sum(np.abs(y_true - y_pred)) / total_y) * 100.0)


def peak_quartile_mae(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.75) -> float:
    threshold = np.quantile(y_true, q)
    mask = y_true >= threshold
    if not np.any(mask):
        return mean_absolute_error(y_true, y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()
    return {
        "MAE": mean_absolute_error(y_t, y_p),
        "RMSE": root_mean_squared_error(y_t, y_p),
        "WAPE": weighted_absolute_percentage_error(y_t, y_p),
        "Peak_MAE": peak_quartile_mae(y_t, y_p),
    }