import numpy as np
import pandas as pd
import pytest
from atomity_forecast.data import build_targets, get_train_holdout_split


def generate_synthetic_smoke_readings(
    n_vms: int = 200,
    n_timesteps: int = 60,
    drop_fraction: float = 0.05,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    vm_ids = [f"vm_{i:04d}" for i in range(n_vms)]
    timestamps = np.arange(0, n_timesteps * 300, 300)

    rows = []
    for ts in timestamps:
        for vid in vm_ids:
            # Randomly simulate missing telemetry
            avg_cpu = np.nan if rng.random() < drop_fraction else rng.uniform(5.0, 85.0)
            rows.append({"timestamp": ts, "vm_id": vid, "avg_cpu": avg_cpu})

    return pd.DataFrame(rows)


def test_smoke_target_aggregation_and_split():
    # Simulate a tiny reviewer smoke dataset: 200 VMs, 60 timesteps
    smoke_df = generate_synthetic_smoke_readings(n_vms=200, n_timesteps=60)
    targets = build_targets(smoke_df)

    # Verifications
    assert len(targets) == 60
    assert "active_vm_count" in targets.columns
    assert "aggregate_cpu_utilization" in targets.columns
    assert (targets["active_vm_count"] <= 200).all()
    assert (targets["aggregate_cpu_utilization"] >= 0.0).all()
    assert (targets["aggregate_cpu_utilization"] <= 100.0).all()

    # Dynamic split check
    train_df, holdout_df = get_train_holdout_split(targets, holdout_len=24)
    assert len(train_df) == 36
    assert len(holdout_df) == 24