"""
Synthetic smoke test to ensure pipeline runs without hardcoded dataset sizes.
"""
import numpy as np
import pandas as pd
from atomity_forecast.config import CONFIG
from atomity_forecast.data import build_targets, get_train_holdout_split


def generate_synthetic_smoke_readings(n_vms: int = 200, n_timesteps: int = 60) -> pd.DataFrame:
    records = []
    base_time = 1000
    for t_idx in range(n_timesteps):
        ts = base_time + (t_idx * 300)
        for vm_id in range(n_vms):
            records.append({
                "timestamp": ts,
                "vm_id": f"vm_{vm_id}",
                "avg_cpu": float(np.random.uniform(2.0, 25.0)),
            })
    return pd.DataFrame(records)


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
    assert len(holdout_df) == 24
    assert len(train_df) == len(targets) - 24