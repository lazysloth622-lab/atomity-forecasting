"""
Generate publication-quality EDA and benchmark figures for the final report.
Saves all outputs directly to reports/figures/.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.atomity_forecast.data import load_raw_data, build_targets, get_train_holdout_split
from src.atomity_forecast.models import DirectRidgeModel

FIG_DIR = Path("reports") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Clean, professional plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})


def generate_all_figures():
    print("[1/5] Ingesting and aggregating trace...")
    df_readings, _ = load_raw_data("challenge_data")
    targets_df = build_targets(df_readings)
    train_df, holdout_df = get_train_holdout_split(targets_df, holdout_len=24)

    y_train = train_df["aggregate_cpu_utilization"].to_numpy()
    y_full = targets_df["aggregate_cpu_utilization"].to_numpy()
    steps = np.arange(len(targets_df))

    # =========================================================================
    # Figure 1: Full Trace Overview (Train vs Holdout Split)
    # =========================================================================
    print("[2/5] Generating Figure 1: trace_overview.png...")
    fig, ax1 = plt.subplots(figsize=(10, 4.5), dpi=300)

    # CPU Utilization line
    ax1.plot(steps[:110], y_full[:110], color="#1f77b4", lw=2, label="Aggregate CPU Util (Development, N=110)")
    ax1.axvline(x=109.5, color="#d62728", linestyle="--", lw=1.8, label="Train / Holdout Boundary (Step 110)")
    ax1.axvspan(110, 133, color="#fee8c8", alpha=0.5, label="Holdout Period (H=24, strictly unseen)")

    ax1.set_xlabel("Time Step (5-minute interval)")
    ax1.set_ylabel("Aggregate CPU Utilization (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlim(-1, 134)

    # Secondary axis for Active VM Count
    ax2 = ax1.twinx()
    ax2.plot(steps, targets_df["active_vm_count"], color="#2ca02c", linestyle=":", lw=1.5, label="Active VMs (10,000)")
    ax2.set_ylabel("Active VM Count", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(9000, 11000)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", frameon=True)

    plt.title("Azure VM Telemetry Trace: 11.1-Hour CPU Utilization & Active Fleet Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "trace_overview.png")
    plt.close()

    # =========================================================================
    # Figure 2: Workload Distribution & Autocorrelation
    # =========================================================================
    print("[3/5] Generating Figure 2: distribution_and_acf.png...")
    fig, (ax_dist, ax_acf) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

    # Subplot A: Histogram + KDE
    counts, bins, _ = ax_dist.hist(
        y_train, bins=16, density=True, color="#4a90e2", edgecolor="black", alpha=0.6, label="Histogram"
    )
    # Simple KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(y_train)
    x_grid = np.linspace(min(y_train) - 0.5, max(y_train) + 0.5, 200)
    ax_dist.plot(x_grid, kde(x_grid), color="#003366", lw=2, label="Density Fit (KDE)")
    ax_dist.axvline(np.mean(y_train), color="#d62728", linestyle="--", label=f"Mean: {np.mean(y_train):.2f}%")
    ax_dist.axvline(np.median(y_train), color="#2ca02c", linestyle=":", label=f"Median: {np.median(y_train):.2f}%")
    ax_dist.set_title("Training Set Utilization Distribution")
    ax_dist.set_xlabel("Aggregate CPU Utilization (%)")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(loc="upper right", frameon=True)

    # Subplot B: Autocorrelation (ACF) up to lag 24
    max_lag = 24
    acf_vals = [1.0]
    for lag in range(1, max_lag + 1):
        corr = np.corrcoef(y_train[:-lag], y_train[lag:])[0, 1]
        acf_vals.append(corr)

    lags_arr = np.arange(max_lag + 1)
    ax_acf.stem(lags_arr, acf_vals, basefmt=" ", linefmt="#1f77b4", markerfmt="o")
    # 95% Bartlett confidence bands
    conf = 1.96 / np.sqrt(len(y_train))
    ax_acf.axhline(conf, color="gray", linestyle="--", lw=1)
    ax_acf.axhline(-conf, color="gray", linestyle="--", lw=1)
    ax_acf.axhline(0, color="black", lw=0.8)
    ax_acf.set_title("Autocorrelation Function (ACF, Lags 0-24)")
    ax_acf.set_xlabel("Lag (5-minute steps)")
    ax_acf.set_ylabel("Autocorrelation")
    ax_acf.set_ylim(-0.3, 1.05)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "distribution_and_acf.png")
    plt.close()

    # =========================================================================
    # Figure 3: Rolling-Origin Cross-Validation Protocol
    # =========================================================================
    print("[4/5] Generating Figure 3: rolling_cv_scheme.png...")
    origins = [50, 58, 66, 74, 82, 86]
    horizon = 24

    fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
    y_ticks, y_labels = [], []

    for idx, orig in enumerate(origins):
        y_pos = len(origins) - 1 - idx
        y_ticks.append(y_pos)
        y_labels.append(f"Fold {idx + 1} (Origin t={orig})")

        # Expanding training window
        ax.barh(y_pos, orig, left=0, height=0.55, color="#3470a3", edgecolor="black", alpha=0.85)
        # 24-step forecast evaluation horizon
        ax.barh(y_pos, horizon, left=orig, height=0.55, color="#ff9933", edgecolor="black", alpha=0.85)

    # Reference holdout bar at the top
    ax.barh(len(origins), 110, left=0, height=0.55, color="#2ca02c", edgecolor="black", alpha=0.85)
    ax.barh(len(origins), 24, left=110, height=0.55, color="#d62728", edgecolor="black", alpha=0.85)
    y_ticks.append(len(origins))
    y_labels.append("Final Holdout Split")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Timeline Step (5-min intervals)")
    ax.set_xlim(0, 135)
    ax.set_title("Expanding-Window Rolling-Origin Validation Scheme (Zero Leakage)")

    # Legend proxy
    p_train = plt.Rectangle((0, 0), 1, 1, fc="#3470a3", ec="black")
    p_test = plt.Rectangle((0, 0), 1, 1, fc="#ff9933", ec="black")
    p_final_train = plt.Rectangle((0, 0), 1, 1, fc="#2ca02c", ec="black")
    p_final_test = plt.Rectangle((0, 0), 1, 1, fc="#d62728", ec="black")
    ax.legend(
        [p_train, p_test, p_final_train, p_final_test],
        ["CV Train Window", "CV Val Horizon (H=24)", "Final Full Train (N=110)", "Untouched Holdout (H=24)"],
        loc="lower left",
        frameon=True,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "rolling_cv_scheme.png")
    plt.close()

    # =========================================================================
    # Figure 4: DirectRidge Holdout Forecast vs Baselines
    # =========================================================================
    print("[5/5] Generating Figure 4: forecast_comparison.png...")
    model = DirectRidgeModel()
    model.fit(y_train, horizon=24)
    ridge_preds = model.predict(horizon=24)
    last_val_preds = np.full(24, y_train[-1])

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)

    # Context history: show the last 30 development steps
    hist_window = 30
    hist_x = np.arange(110 - hist_window, 110)
    ax.plot(hist_x, y_train[-hist_window:], color="black", lw=2, label="Historical Trajectory (Steps 80-109)")

    # Holdout horizon x-coordinates
    horizon_x = np.arange(110, 134)

    # Forecast curves
    ax.plot(horizon_x, ridge_preds, color="#1f77b4", marker="o", markersize=4, lw=2.2, label="DirectRidge Forecast (Benchmark Winner)")
    ax.plot(horizon_x, last_val_preds, color="#7f7f7f", linestyle="--", lw=1.8, label=f"LastValue Baseline ({y_train[-1]:.2f}%)")

    # Shading and dividers
    ax.axvline(x=109.5, color="#d62728", linestyle=":", lw=1.5)
    ax.axvspan(110, 133, color="#f0f0f0", alpha=0.6, label="24-Step Holdout Horizon")

    ax.set_title("DirectRidge 24-Step Holdout Forecast vs Naive Baseline")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Aggregate CPU Utilization (%)")
    ax.set_xlim(80, 135)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_comparison.png")
    plt.close()

    print("\nAll figures successfully created in 'reports/figures/':")
    for f in FIG_DIR.glob("*.png"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    generate_all_figures()