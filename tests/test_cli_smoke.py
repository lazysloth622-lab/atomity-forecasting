"""
End-to-end smoke test: a reviewer can swap in a tiny synthetic dataset and
produce a valid predictions artifact through the documented CLI.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

from atomity_forecast.cli import main


def _write_synthetic_release(data_dir, n_vms: int = 50, n_steps: int = 80) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for step in range(n_steps):
        ts = 1000 + step * 300
        for vm in range(n_vms):
            avg = float(rng.uniform(2.0, 25.0))
            rows.append(
                {
                    "timestamp": ts,
                    "vm_id": f"vm_{vm}",
                    "min_cpu": max(0.0, avg - 1.0),
                    "max_cpu": min(100.0, avg + 1.0),
                    "avg_cpu": avg,
                }
            )
    readings = pd.DataFrame(rows)
    metadata = pd.DataFrame(
        {
            "vm_id": [f"vm_{vm}" for vm in range(n_vms)],
            "timestamp_vm_created": 1000,
            "timestamp_vm_deleted": 1000 + n_steps * 300,
            "vm_category": "Interactive",
            "virtual_core_bucket": "2",
            "memory_gb_bucket": "4",
        }
    )
    with gzip.open(data_dir / "vm_cpu_readings.csv.gz", "wt") as f:
        readings.to_csv(f, index=False)
    with gzip.open(data_dir / "vm_metadata.csv.gz", "wt") as f:
        metadata.to_csv(f, index=False)


def test_predict_cli_on_smoke_dataset(tmp_path):
    data_dir = tmp_path / "challenge_data"
    out_file = tmp_path / "predictions.csv"
    _write_synthetic_release(data_dir)

    main(["predict", "--data", str(data_dir), "--output", str(out_file)])

    df = pd.read_csv(out_file)
    assert list(df.columns) == [
        "timestamp",
        "active_vm_count",
        "aggregate_cpu_utilization",
    ]
    assert len(df) == 24
    assert not df.isna().any().any()
    assert (df["active_vm_count"] >= 0).all()
    assert (df["active_vm_count"] <= 50).all()
    cpu = df["aggregate_cpu_utilization"].to_numpy()
    assert np.all((cpu >= 0.0) & (cpu <= 100.0))
    assert (np.diff(df["timestamp"].to_numpy()) == 300).all()
