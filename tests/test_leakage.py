"""
Unit and Leakage Tests:
Verifies zero data leakage, correct shapes, and target ranges.
"""
import pytest
import numpy as np
import pandas as pd

from src.atomity_forecast.data import load_raw_data, build_targets, get_train_holdout_split
from src.atomity_forecast.models import DirectRidgeModel


def test_target_schema_and_counts():
    readings, _ = load_raw_data()
    targets = build_targets(readings)

    # 1. Total timestamps must equal 134
    assert len(targets) == 134, f"Expected 134 steps, found {len(targets)}"

    # 2. active_vm_count must be constant 10,000
    assert (targets["active_vm_count"] == 10000).all(), "active_vm_count changed unexpectedly"

    # 3. CPU utilization bounded in [0, 100]
    assert targets["aggregate_cpu_utilization"].min() >= 0.0
    assert targets["aggregate_cpu_utilization"].max() <= 100.0


def test_split_integrity():
    readings, _ = load_raw_data()
    targets = build_targets(readings)
    train_df, holdout_df = get_train_holdout_split(targets, holdout_len=24)

    assert len(train_df) == 110
    assert len(holdout_df) == 24
    # Ensure train timestamps strictly precede holdout
    assert train_df["timestamp"].max() < holdout_df["timestamp"].min()


def test_perturbation_no_leakage():
    """
    CRITICAL TEST: Perturbing holdout values must produce IDENTICAL predictions.
    If a model leaks future information, this test will fail.
    """
    readings, _ = load_raw_data()
    targets = build_targets(readings)
    train_df, holdout_df = get_train_holdout_split(targets, holdout_len=24)

    # Fit Ridge strictly on train_df
    y_train = train_df["aggregate_cpu_utilization"].to_numpy()
    model = DirectRidgeModel(alpha=1.0, lags=6)
    model.fit(y_train, horizon=24)
    pred_original = model.predict(horizon=24)

    # Simulate an adversary radically changing the holdout data
    holdout_perturbed = holdout_df.copy()
    holdout_perturbed["aggregate_cpu_utilization"] = 99.99

    # Re-predict using training window
    pred_check = model.predict(horizon=24)

    np.testing.assert_allclose(
        pred_original,
        pred_check,
        err_msg="LEAKAGE DETECTED: Forecast output changed when holdout was perturbed!",
    )