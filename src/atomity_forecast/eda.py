"""
Workload Exploratory Data Analysis module for Atomity Cloud Forecasting.
Strictly restricted to the development/training window (first N - holdout steps).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from atomity_forecast.config import CONFIG, PATHS
from atomity_forecast.data import build_targets, get_train_holdout_split, load_raw_data


def compute_training_eda(train_df: pd.DataFrame) -> dict:
    cpu_series = train_df["aggregate_cpu_utilization"].to_numpy()
    vms_series = train_df["active_vm_count"].to_numpy()
    n_obs = len(cpu_series)

    nlags = min(30, n_obs // 2)
    autocorr = acf(cpu_series, nlags=nlags, fft=True)
    white_noise_band = 1.96 / np.sqrt(n_obs)

    eda_stats = {
        "n_training_steps": int(n_obs),
        "active_vm_count_constant": bool(np.all(vms_series == vms_series[0])),
        "active_vm_count_last": int(vms_series[-1]),
        "cpu_mean": float(np.mean(cpu_series)),
        "cpu_std": float(np.std(cpu_series, ddof=1)),
        "cpu_min": float(np.min(cpu_series)),
        "cpu_max": float(np.max(cpu_series)),
        "cpu_median": float(np.median(cpu_series)),
        "lag1_autocorr": float(autocorr[1]),
        "lag24_autocorr": float(autocorr[24]) if len(autocorr) > 24 else None,
        "white_noise_band_ci95": float(white_noise_band),
    }
    return eda_stats


def generate_eda_figures_and_tables(
    train_df: pd.DataFrame, stats: dict, figures_dir: Path, tables_dir: Path
):
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    cpu_series = train_df["aggregate_cpu_utilization"].to_numpy()
    timestamps = train_df["timestamp"].to_numpy()
    n_obs = len(cpu_series)

    # 1. Training series plot
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, cpu_series, color="#1f77b4", lw=1.5, label="Train CPU Util (%)")
    plt.axhline(stats["cpu_mean"], color="red", linestyle="--", label=f"Mean: {stats['cpu_mean']:.2f}%")
    plt.title("Development Window Aggregate CPU Utilization (First 110 Steps)")
    plt.xlabel("Timestamp (s)")
    plt.ylabel("Mean CPU (%)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_training_series.png", dpi=300)
    plt.close()

    # 2. Autocorrelation plot with White-noise band
    nlags = min(30, n_obs // 2)
    autocorr = acf(cpu_series, nlags=nlags, fft=True)
    lags = np.arange(len(autocorr))
    ci = stats["white_noise_band_ci95"]

    plt.figure(figsize=(10, 4))
    plt.stem(lags, autocorr, basefmt="k-")
    plt.axhline(ci, color="gray", linestyle="--", label=f"White-noise band (+1.96/√N = {ci:.3f})")
    plt.axhline(-ci, color="gray", linestyle="--", label=f"White-noise band (-1.96/√N = -{ci:.3f})")
    plt.axhline(0, color="black", lw=1)
    plt.title("Aggregate CPU Autocorrelation (Training Window Only)")
    plt.xlabel("Lag (5-min intervals)")
    plt.ylabel("Autocorrelation")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "eda_autocorrelation.png", dpi=300)
    plt.close()

    # Save summary table
    summary_df = pd.DataFrame([stats])
    summary_df.to_csv(tables_dir / "eda_summary.csv", index=False)


def run_eda():
    df_readings, _ = load_raw_data(PATHS.data_dir)
    targets_df = build_targets(df_readings)
    train_df, _ = get_train_holdout_split(targets_df, holdout_len=CONFIG.holdout_steps)

    stats = compute_training_eda(train_df)

    PATHS.numbers_json.parent.mkdir(parents=True, exist_ok=True)
    existing_numbers = {}
    if PATHS.numbers_json.exists():
        with open(PATHS.numbers_json, "r") as f:
            existing_numbers = json.load(f)

    existing_numbers.update(stats)
    with open(PATHS.numbers_json, "w") as f:
        json.dump(existing_numbers, f, indent=2)

    generate_eda_figures_and_tables(train_df, stats, PATHS.figures_dir, PATHS.tables_dir)
    print("EDA execution completed successfully.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    run_eda()