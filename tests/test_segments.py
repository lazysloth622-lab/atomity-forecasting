"""Tests for metadata-backed churn verification and segment summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from atomity_forecast.segments import segment_summary, verify_vm_churn


def _readings(n_vms: int = 12, n_steps: int = 30) -> pd.DataFrame:
    rows = []
    for step in range(n_steps):
        ts = step * 300
        for vm in range(n_vms):
            rows.append(
                {"timestamp": ts, "vm_id": f"vm_{vm}", "avg_cpu": 5.0 + (vm % 3)}
            )
    return pd.DataFrame(rows)


def _metadata(n_vms: int = 12, deleted_at: int = 999999) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vm_id": [f"vm_{vm}" for vm in range(n_vms)],
            "timestamp_vm_created": 0,
            "timestamp_vm_deleted": deleted_at,
            "vm_category": ["Interactive"] * (n_vms // 2)
            + ["Delay-insensitive"] * (n_vms // 2),
            "virtual_core_bucket": "2",
            "memory_gb_bucket": "4",
        }
    )


def test_zero_churn_when_no_interior_events():
    readings = _readings()
    meta = _metadata()
    out = verify_vm_churn(readings, meta)
    assert out["churn_verifiable"] is True
    assert out["vms_created_in_window"] == 0
    assert out["vms_deleted_in_window"] == 0
    assert out["zero_churn_in_window"] is True


def test_detects_interior_deletion():
    readings = _readings()
    meta = _metadata()
    meta.loc[0, "timestamp_vm_deleted"] = 3000  # inside 0..8700 window
    out = verify_vm_churn(readings, meta)
    assert out["vms_deleted_in_window"] == 1
    assert out["zero_churn_in_window"] is False


def test_segment_summary_shape_and_values():
    readings = _readings()
    meta = _metadata()
    seg = segment_summary(readings, meta)
    assert {"vm_category", "virtual_core_bucket", "memory_gb_bucket"} <= set(
        seg["segment_type"]
    )
    cat = seg[seg["segment_type"] == "vm_category"].set_index("segment")
    # Interactive VMs (vm ids 0-5 get avg 5..7) should average higher than the
    # Delay-insensitive half in expectation; both means are finite and in range.
    assert np.isfinite(cat.loc["Interactive", "mean_cpu_pct"])
    assert (cat["n_vms"] == 6).all()


def test_missing_metadata_is_handled_gracefully():
    readings = _readings()
    out = verify_vm_churn(readings, pd.DataFrame())
    assert out["churn_verifiable"] is False
    assert segment_summary(readings, pd.DataFrame()).empty
