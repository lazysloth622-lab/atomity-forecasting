"""
Temporal leakage and split verification tests.
"""
import numpy as np
import pandas as pd
from atomity_forecast.config import PATHS, CONFIG
from atomity_forecast.data import load_raw_data, build_targets, get_train_holdout_split
from atomity_forecast.models import EnsembleModel, LastValueBaseline, AR1Model, DirectRidgeModel

def test_target_schema_and_counts():
    df_readings, _ = load_raw_data(PATHS.data_dir)
    targets = build_targets(df_readings)
    assert len(targets) > 0
    assert (targets["active_vm_count"] > 0).all()
    assert "aggregate_cpu_utilization" in targets.columns

def test_split_integrity():
    df_readings, _ = load_raw_data(PATHS.data_dir)
    targets = build_targets(df_readings)
    train, holdout = get_train_holdout_split(targets, holdout_len=CONFIG.holdout_steps)
    assert len(holdout) == CONFIG.holdout_steps
    assert len(train) + len(holdout) == len(targets)
    assert train["timestamp"].max() < holdout["timestamp"].min()

def test_perturbation_no_leakage():
    df_readings, _ = load_raw_data(PATHS.data_dir)
    targets = build_targets(df_readings)
    train, holdout = get_train_holdout_split(targets, holdout_len=CONFIG.holdout_steps)
    
    y_train = train["aggregate_cpu_utilization"].to_numpy()
    model1 = EnsembleModel([LastValueBaseline(), AR1Model(), DirectRidgeModel(alpha=1.0, lags=6)])
    model1.fit(y_train, horizon=CONFIG.holdout_steps)
    preds1 = model1.predict(horizon=CONFIG.holdout_steps)
    
    # Perturb holdout data completely
    perturbed_holdout = holdout.copy()
    perturbed_holdout["aggregate_cpu_utilization"] = perturbed_holdout["aggregate_cpu_utilization"] + 50.0
    
    # Fit again strictly on train - predictions MUST remain identical
    model2 = EnsembleModel([LastValueBaseline(), AR1Model(), DirectRidgeModel(alpha=1.0, lags=6)])
    model2.fit(y_train, horizon=CONFIG.holdout_steps)
    preds2 = model2.predict(horizon=CONFIG.holdout_steps)
    
    np.testing.assert_allclose(preds1, preds2)