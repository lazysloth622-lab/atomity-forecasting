"""
Metadata-backed workload analysis: VM churn verification and fleet segmentation.

All statistics here are descriptive and use only metadata columns and the
development-window readings; no holdout target values are consumed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from atomity_forecast.config import CONFIG, PATHS
from atomity_forecast.data import build_targets, get_train_holdout_split, load_raw_data


def verify_vm_churn(
    readings: pd.DataFrame, metadata: pd.DataFrame
) -> dict[str, int | float | bool]:
    """Check VM creation/deletion events against the observation window."""
    window_lo = float(readings["timestamp"].min())
    window_hi = float(readings["timestamp"].max())

    if metadata.empty or "timestamp_vm_created" not in metadata.columns:
        return {
            "churn_window_lo_s": window_lo,
            "churn_window_hi_s": window_hi,
            "vms_created_in_window": -1,
            "vms_deleted_in_window": -1,
            "churn_verifiable": False,
        }

    created = metadata["timestamp_vm_created"]
    deleted = metadata["timestamp_vm_deleted"]
    # Creation exactly at the window origin (t == window_lo) is the trace start,
    # not mid-window churn; strictly interior events count as churn.
    created_inside = int(((created > window_lo) & (created <= window_hi)).sum())
    deleted_inside = int(((deleted >= window_lo) & (deleted <= window_hi)).sum())

    return {
        "churn_window_lo_s": window_lo,
        "churn_window_hi_s": window_hi,
        "vms_created_in_window": created_inside,
        "vms_deleted_in_window": deleted_inside,
        "churn_verifiable": True,
        "zero_churn_in_window": bool(created_inside == 0 and deleted_inside == 0),
    }


def segment_summary(
    readings: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Mean utilization and temporal volatility per meaningful VM segment."""
    if metadata.empty or "vm_id" not in metadata.columns:
        return pd.DataFrame()

    seg_cols = [
        c
        for c in ("vm_category", "virtual_core_bucket", "memory_gb_bucket")
        if c in metadata.columns
    ]
    if not seg_cols:
        return pd.DataFrame()

    merged = readings.merge(
        metadata[["vm_id", *seg_cols]], on="vm_id", how="left"
    )

    rows = []
    for col in seg_cols:
        grouped = merged.groupby(["timestamp", col])["avg_cpu"].mean().unstack()
        for segment_name, series in grouped.items():
            rows.append(
                {
                    "segment_type": col,
                    "segment": str(segment_name),
                    "mean_cpu_pct": float(series.mean()),
                    "std_cpu_pct": float(series.std(ddof=1)),
                    "min_cpu_pct": float(series.min()),
                    "max_cpu_pct": float(series.max()),
                    "n_vms": int(
                        metadata.loc[metadata[col] == segment_name, "vm_id"].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["segment_type", "mean_cpu_pct"])


def run_segment_analysis() -> dict:
    readings, metadata = load_raw_data(PATHS.data_dir)
    # Restrict descriptive analysis to the development window.
    targets = build_targets(readings)
    train_df, _ = get_train_holdout_split(targets, holdout_len=CONFIG.holdout_steps)
    train_ts = set(train_df["timestamp"].to_numpy())
    train_readings = readings[readings["timestamp"].isin(train_ts)]

    churn = verify_vm_churn(train_readings, metadata)
    seg = segment_summary(train_readings, metadata)

    PATHS.tables_dir.mkdir(parents=True, exist_ok=True)
    if not seg.empty:
        seg.round(6).to_csv(PATHS.tables_dir / "segment_summary.csv", index=False)

    numbers = {}
    if PATHS.numbers_json.exists():
        with open(PATHS.numbers_json, "r") as f:
            numbers = json.load(f)
    numbers.update(churn)
    if not seg.empty:
        cat = seg[seg["segment_type"] == "vm_category"]
        numbers["segment_category_mean_cpu"] = dict(
            zip(cat["segment"], cat["mean_cpu_pct"].round(4))
        )
        numbers["segment_category_std_cpu"] = dict(
            zip(cat["segment"], cat["std_cpu_pct"].round(4))
        )
    with open(PATHS.numbers_json, "w") as f:
        json.dump(numbers, f, indent=2)

    print("=== CHURN ===")
    print(json.dumps(churn, indent=2))
    print("\n=== SEGMENTS ===")
    print(seg.to_string(index=False) if not seg.empty else "metadata unavailable")
    return {"churn": churn, "segments": seg}


if __name__ == "__main__":
    run_segment_analysis()
