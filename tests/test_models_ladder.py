import numpy as np
import pytest
from atomity_forecast.models import (
    TrainMeanBaseline,
    LastValueBaseline,
    AR1Model,
    HoltDampedModel,
    DirectRidgeModel,
    DirectLightGBMModel,
    EnsembleModel,
)


@pytest.fixture
def dummy_series():
    np.random.seed(42)
    return 10.0 + np.sin(np.linspace(0, 10, 80)) + np.random.normal(0, 0.2, 80)


def test_all_models_produce_bounded_24_step_forecast(dummy_series):
    horizon = 24
    models = [
        TrainMeanBaseline(),
        LastValueBaseline(),
        AR1Model(),
        HoltDampedModel(),
        DirectRidgeModel(),
        DirectLightGBMModel(),
    ]

    for model in models:
        if hasattr(model, "fit") and "horizon" in model.fit.__code__.co_varnames:
            model.fit(dummy_series, horizon=horizon)
        else:
            model.fit(dummy_series)

        preds = model.predict(horizon=horizon)
        assert len(preds) == horizon
        assert not np.isnan(preds).any()
        assert np.all((preds >= 0.0) & (preds <= 100.0))

    # Test Ensemble
    ensemble = EnsembleModel([LastValueBaseline(), AR1Model(), DirectRidgeModel()])
    ensemble.fit(dummy_series, horizon=horizon)
    ens_preds = ensemble.predict(horizon=horizon)
    assert len(ens_preds) == horizon
    assert not np.isnan(ens_preds).any()