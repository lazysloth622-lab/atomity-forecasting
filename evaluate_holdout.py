"""
Single-pass holdout evaluation script for Atomity Cloud Forecasting.
Calculates final holdout generalization metrics for the frozen submission,
including per-horizon-step errors and baseline comparison.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from atomity_forecast.config import CONFIG, PATHS
from atomity_forecast.data import (
    build_targets,
    get_train_holdout_split,
    load_raw_data,
)
from atomity_forecast.metrics import evaluate_forecast
from atomity_forecast.models import LastValueBaseline, TrainMeanBaseline
from atomity_forecast.predict import generate_predictions


def evaluate_holdout() -> dict:
    # 1. Generate predictions artifact (frozen ensemble, train window only)
    preds_df = generate_predictions()

    # 2. Extract strictly unseen ground-truth holdout
    df_readings, _ = load_raw_data(PATHS.data_dir)
    targets_df = build_targets(df_readings)
    train_df, holdout_df = get_train_holdout_split(
        targets_df, holdout_len=CONFIG.holdout_steps
    )

    y_true_cpu = holdout_df["aggregate_cpu_utilization"].to_numpy()
    y_pred_cpu = preds_df["aggregate_cpu_utilization"].to_numpy()

    y_true_vms = holdout_df["active_vm_count"].to_numpy()
    y_pred_vms = preds_df["active_vm_count"].to_numpy()

    cpu_metrics = evaluate_forecast(y_true_cpu, y_pred_cpu)
    vm_mae = float(np.mean(np.abs(y_true_vms - y_pred_vms)))

    # 3. Baselines fit on the training window only, scored on the same holdout
    y_train = train_df["aggregate_cpu_utilization"].to_numpy()
    persistence = LastValueBaseline().fit(y_train).predict(len(holdout_df))
    train_mean = TrainMeanBaseline().fit(y_train).predict(len(holdout_df))
    persistence_mae = float(np.mean(np.abs(y_true_cpu - persistence)))
    train_mean_mae = float(np.mean(np.abs(y_true_cpu - train_mean)))

    # 4. Per-horizon-step errors (code-generated, backs report section 5.1)
    step_errors = pd.DataFrame(
        {
            "step": np.arange(1, len(holdout_df) + 1),
            "timestamp": holdout_df["timestamp"].to_numpy(),
            "abs_error": np.round(np.abs(y_true_cpu - y_pred_cpu), 6),
            "signed_error": np.round(y_pred_cpu - y_true_cpu, 6),
        }
    )

    holdout_summary = {
        "holdout_steps": int(len(holdout_df)),
        "cpu_MAE": float(cpu_metrics["MAE"]),
        "cpu_RMSE": float(cpu_metrics["RMSE"]),
        "cpu_WAPE": float(cpu_metrics["WAPE"]),
        "cpu_Peak_MAE": float(cpu_metrics["Peak_MAE"]),
        "vm_count_MAE": vm_mae,
        "holdout_persistence_mae": persistence_mae,
        "holdout_train_mean_mae": train_mean_mae,
        "holdout_relative_improvement_vs_persistence_pct": float(
            (1.0 - cpu_metrics["MAE"] / persistence_mae) * 100.0
        ),
        "holdout_step1_mae": float(step_errors["abs_error"].iloc[0]),
        "holdout_step12_mae": float(step_errors["abs_error"].iloc[11]),
        "holdout_step24_mae": float(step_errors["abs_error"].iloc[23]),
    }

    PATHS.tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([holdout_summary]).round(6).to_csv(
        PATHS.tables_dir / "holdout_metrics.csv", index=False
    )
    step_errors.to_csv(
        PATHS.tables_dir / "holdout_error_by_step.csv", index=False
    )

    existing_numbers = {}
    if PATHS.numbers_json.exists():
        with open(PATHS.numbers_json, "r") as f:
            existing_numbers = json.load(f)
    existing_numbers.update(holdout_summary)
    with open(PATHS.numbers_json, "w") as f:
        json.dump(existing_numbers, f, indent=2)

    print("\n=== FINAL HOLDOUT EVALUATION RESULTS ===")
    print(json.dumps(holdout_summary, indent=2))
    return holdout_summary


if __name__ == "__main__":
    evaluate_holdout()
