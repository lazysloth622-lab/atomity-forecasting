"""
Artifact structure and schema verification tests.
"""
import numpy as np
import pandas as pd
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
    assert np.all((cpu >= 0.0) & (cpu <= 100.0)), "CPU predictions out of bounds [0, 100]"

    # 4. Monotonic uniform timestamps
    timestamps = df["timestamp"].to_numpy()
    diffs = np.diff(timestamps)
    assert len(np.unique(diffs)) == 1, "Timestamps are not uniform"
    assert diffs[0] > 0, "Timestamps are not strictly increasing"