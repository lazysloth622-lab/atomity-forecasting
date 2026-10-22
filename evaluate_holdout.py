"""
Single-pass holdout evaluation script for Atomity Cloud Workload Forecasting.
Calculates final holdout generalization metrics for the frozen submission.
"""
import json
import pandas as pd
import numpy as np

from atomity_forecast.config import CONFIG, PATHS
from atomity_forecast.data import load_raw_data, build_targets, get_train_holdout_split
from atomity_forecast.metrics import evaluate_forecast
from atomity_forecast.predict import generate_predictions


def evaluate_holdout():
    # 1. Generate predictions artifact
    preds_df = generate_predictions()

    # 2. Extract strictly unseen ground-truth holdout
    df_readings, _ = load_raw_data(PATHS.data_dir)
    targets_df = build_targets(df_readings)
    _, holdout_df = get_train_holdout_split(targets_df, holdout_len=CONFIG.holdout_steps)

    y_true_cpu = holdout_df["aggregate_cpu_utilization"].to_numpy()
    y_pred_cpu = preds_df["aggregate_cpu_utilization"].to_numpy()

    y_true_vms = holdout_df["active_vm_count"].to_numpy()
    y_pred_vms = preds_df["active_vm_count"].to_numpy()

    cpu_metrics = evaluate_forecast(y_true_cpu, y_pred_cpu)
    vm_mae = float(np.mean(np.abs(y_true_vms - y_pred_vms)))

    holdout_summary = {
        "holdout_steps": int(len(holdout_df)),
        "cpu_MAE": float(cpu_metrics["MAE"]),
        "cpu_RMSE": float(cpu_metrics["RMSE"]),
        "cpu_WAPE": float(cpu_metrics["WAPE"]),
        "cpu_Peak_MAE": float(cpu_metrics["Peak_MAE"]),
        "vm_count_MAE": float(vm_mae),
    }

    # Persist metrics to reports/tables and reports/numbers.json
    PATHS.tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([holdout_summary]).to_csv(
        PATHS.tables_dir / "holdout_metrics.csv", index=False
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


if __name__ == "__main__":
    evaluate_holdout()