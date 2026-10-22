"""
Edge case and mathematical unit tests for metrics.
"""
import numpy as np
from atomity_forecast.metrics import weighted_absolute_percentage_error, mean_absolute_error


def test_wape_returns_nan_on_zero_denominator():
    y_true = np.zeros(10)
    y_pred = np.ones(10)
    wape = weighted_absolute_percentage_error(y_true, y_pred)
    assert np.isnan(wape), f"Expected NaN for WAPE with zero sum denominator, got {wape}"


def test_wape_correct_nonzero():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 30.0])
    # abs errors: [1.0, 1.0, 0.0] -> sum = 2.0. total_y = 60.0 -> (2/60)*100 = 3.333%
    wape = weighted_absolute_percentage_error(y_true, y_pred)
    assert np.isclose(wape, (2.0 / 60.0) * 100.0)