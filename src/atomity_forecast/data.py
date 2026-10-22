"""
Data ingestion, target aggregation, and split utilities for Atomity Cloud Forecasting.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def load_raw_data(data_dir: str | Path = "challenge_data") -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    readings_path = data_path / "vm_cpu_readings.csv.gz"
    meta_path = data_path / "vm_metadata.csv.gz"

    if not readings_path.exists():
        raise FileNotFoundError(f"Missing readings file: {readings_path}")

    df_readings = pd.read_csv(readings_path)
    df_meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()

    return df_readings, df_meta


def build_targets(df_readings: pd.DataFrame) -> pd.DataFrame:
    df = df_readings.sort_values(["timestamp", "vm_id"]).reset_index(drop=True)
    valid_readings = df.dropna(subset=["avg_cpu"])

    targets = (
        valid_readings.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
    )

    targets = targets.sort_values("timestamp").reset_index(drop=True)
    return targets


def get_train_holdout_split(
    targets_df: pd.DataFrame, holdout_len: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_len = len(targets_df)
    if total_len <= holdout_len:
        raise ValueError(
            f"Dataset length ({total_len}) must be strictly greater than holdout_len ({holdout_len})"
        )

    train_df = targets_df.iloc[:-holdout_len].copy().reset_index(drop=True)
    holdout_df = targets_df.iloc[-holdout_len:].copy().reset_index(drop=True)

    return train_df, holdout_df