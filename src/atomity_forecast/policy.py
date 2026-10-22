"""
Uncertainty quantification and capacity provisioning policy module.
Calculates residual-based prediction intervals and recommended VM capacity buffers.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atomity_forecast.config import CONFIG, PATHS
from atomity_forecast.data import build_targets, get_train_holdout_split, load_raw_data


def compute_uncertainty_and_capacity_policy():
    # 1. Load predictions and ground truth
    preds_df = pd.read_csv(PATHS.predictions_csv)
    df_readings, _ = load_raw_data(PATHS.data_dir)
    targets_df = build_targets(df_readings)
    train_df, holdout_df = get_train_holdout_split(targets_df, holdout_len=CONFIG.holdout_steps)

    y_train = train_df["aggregate_cpu_utilization"].to_numpy()
    y_holdout = holdout_df["aggregate_cpu_utilization"].to_numpy()
    y_pred = preds_df["aggregate_cpu_utilization"].to_numpy()
    timestamps_holdout = holdout_df["timestamp"].to_numpy()

    # Empirical residual standard deviation on training development window (AR(1) residuals)
    train_residuals = y_train[1:] - y_train[:-1]
    residual_std = float(np.std(train_residuals, ddof=1))

    # 95% Confidence Band (1.96 * residual_std)
    half_width = 1.96 * residual_std
    lower_bound = np.clip(y_pred - half_width, 0.0, 100.0)
    upper_bound = np.clip(y_pred + half_width, 0.0, 100.0)

    # Capacity buffer policy: Target maximum utilization of 80% with safety margin
    # Required capacity = (Predicted CPU + 1.96*sigma) / 0.80
    target_util_cap = 80.0
    recommended_headroom_multiplier = float(np.round(np.max(upper_bound) / target_util_cap, 4))
    sla_violations_count = int(np.sum(y_holdout > upper_bound))

    policy_stats = {
        "residual_std": residual_std,
        "interval_95_half_width": float(half_width),
        "target_util_threshold": target_util_cap,
        "capacity_headroom_multiplier": recommended_headroom_multiplier,
        "holdout_sla_violations": sla_violations_count,
        "holdout_coverage_percentage": float(np.mean((y_holdout >= lower_bound) & (y_holdout <= upper_bound)) * 100.0),
    }

    # Save to reports/numbers.json
    numbers = {}
    if PATHS.numbers_json.exists():
        with open(PATHS.numbers_json, "r") as f:
            numbers = json.load(f)
    numbers.update(policy_stats)
    with open(PATHS.numbers_json, "w") as f:
        json.dump(numbers, f, indent=2)

    # Plot forecast with uncertainty intervals vs actuals
    PATHS.figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.5))
    plt.plot(timestamps_holdout, y_holdout, label="Actual Ground Truth", color="black", lw=1.5, marker="o", markersize=3)
    plt.plot(timestamps_holdout, y_pred, label="Ensemble Forecast", color="#1f77b4", lw=2)
    plt.fill_between(
        timestamps_holdout,
        lower_bound,
        upper_bound,
        color="#1f77b4",
        alpha=0.25,
        label="95% Empirical Prediction Interval",
    )
    plt.title("Holdout Horizon (Steps 110-133): Forecast vs Actual with 95% Uncertainty Band")
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Aggregate CPU Utilization (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PATHS.figures_dir / "holdout_forecast_with_intervals.png", dpi=300)
    plt.close()

    print("=== UNCERTAINTY & CAPACITY POLICY SUMMARY ===")
    print(json.dumps(policy_stats, indent=2))


if __name__ == "__main__":
    compute_uncertainty_and_capacity_policy()