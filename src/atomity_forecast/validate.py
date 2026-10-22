"""
Rolling-origin cross-validation benchmark runner for Atomity Cloud Workload Forecasting.
Strictly operates within the development/training window (steps 0 to 109).
"""
import json
import numpy as np
import pandas as pd

from atomity_forecast.config import CONFIG, PATHS
from atomity_forecast.data import build_targets, get_train_holdout_split, load_raw_data
from atomity_forecast.split import generate_rolling_origin_folds
from atomity_forecast.metrics import evaluate_forecast
from atomity_forecast.models import (
    TrainMeanBaseline,
    LastValueBaseline,
    AR1Model,
    HoltDampedModel,
    DirectRidgeModel,
    DirectLightGBMModel,
    EnsembleModel,
)


def run_benchmark():
    df_readings, _ = load_raw_data(PATHS.data_dir)
    targets_df = build_targets(df_readings)
    train_df, _ = get_train_holdout_split(targets_df, holdout_len=CONFIG.holdout_steps)
    y_train = train_df["aggregate_cpu_utilization"].to_numpy()

    folds = generate_rolling_origin_folds(
        n_train_points=len(y_train),
        horizon=CONFIG.horizon_steps,
        min_train_len=CONFIG.min_origin,
        step_size=CONFIG.origin_step,
    )

    models_factory = {
        "TrainMean": lambda: TrainMeanBaseline(),
        "LastValue": lambda: LastValueBaseline(),
        "AR1": lambda: AR1Model(),
        "HoltDamped": lambda: HoltDampedModel(),
        "DirectRidge": lambda: DirectRidgeModel(alpha=1.0, lags=6),
        "DirectLightGBM": lambda: DirectLightGBMModel(lags=6),
        "Ensemble": lambda: EnsembleModel(
            [LastValueBaseline(), AR1Model(), DirectRidgeModel(alpha=1.0, lags=6)]
        ),
    }

    fold_results = []
    for fold in folds:
        y_tr = y_train[fold.train_indices]
        y_val = y_train[fold.val_indices]

        for model_name, factory in models_factory.items():
            model = factory()
            if hasattr(model, "fit") and "horizon" in model.fit.__code__.co_varnames:
                model.fit(y_tr, horizon=len(y_val))
            else:
                model.fit(y_tr)

            preds = model.predict(horizon=len(y_val))
            metrics = evaluate_forecast(y_val, preds)
            metrics["fold"] = fold.fold_idx
            metrics["model"] = model_name
            fold_results.append(metrics)

    results_df = pd.DataFrame(fold_results)
    
    # Save per-fold results
    PATHS.tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(PATHS.tables_dir / "cv_fold_metrics.csv", index=False)

    # Compute summary
    summary = (
        results_df.groupby("model")[["MAE", "RMSE", "WAPE", "Peak_MAE"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns
    ]
    summary = summary.sort_values("MAE_mean").reset_index(drop=True)
    summary.to_csv(PATHS.tables_dir / "cv_summary.csv", index=False)

    best_model_name = summary.iloc[0]["model"]
    best_mae = float(summary.iloc[0]["MAE_mean"])
    best_peak_mae = float(summary.iloc[0]["Peak_MAE_mean"])

    print("=== CROSS-VALIDATION SUMMARY (TRAINING WINDOW) ===")
    print(summary[["model", "MAE_mean", "MAE_std", "Peak_MAE_mean"]])
    print(f"\nWinning Model by Pre-Registered Rule: {best_model_name}")

    # Persist to numbers.json
    numbers = {}
    if PATHS.numbers_json.exists():
        with open(PATHS.numbers_json, "r") as f:
            numbers = json.load(f)

    numbers["cv_winning_model"] = best_model_name
    numbers["cv_best_mae"] = best_mae
    numbers["cv_best_peak_mae"] = best_peak_mae
    with open(PATHS.numbers_json, "w") as f:
        json.dump(numbers, f, indent=2)


if __name__ == "__main__":
    run_benchmark()