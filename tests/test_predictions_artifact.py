"""
Artifact specification tests for predictions.csv.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from atomity_forecast.config import PATHS, CONFIG


def test_predictions_artifact_structure_and_constraints():
    assert PATHS.predictions_csv.exists(), f"Missing artifact: {PATHS.predictions_csv}"
    
    df = pd.read_csv(PATHS.predictions_csv)
    
    # 1. Exact shape checks
    assert len(df) == CONFIG.holdout_steps, f"Expected {CONFIG.holdout_steps} rows, got {len(df)}"
    assert list(df.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ], f"Unexpected columns: {list(df.columns)}"

    # 2. Null and integrity checks
    assert not df.isna().any().any(), "predictions.csv contains NaN or null values"
    
    # 3. Domain constraints
    cpu = df["aggregate_cpu_utilization"].to_numpy()
    assert np.all((cpu >= 0.0) & (cpu <= 100.0)), "CPU utilization out of bounds [0, 100]"
    
    # 4. Monotonic timestamps check
    timestamps = df["timestamp"].to_numpy()
    assert np.all(np.diff(timestamps) == CONFIG.timestep_seconds), "Timestamps are not regular 300s steps"