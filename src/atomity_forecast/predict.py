"""
Holdout prediction generator for Atomity Cloud Workload Forecasting.
Generates predictions.csv using the frozen winning Ensemble model.

Leakage protocol: only the first (N - holdout) target observations are used
for fitting. Holdout values are touched solely to read their timestamps.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from atomity_forecast.config import CONFIG, PATHS
from atomity_forecast.data import (
    build_targets,
    get_train_holdout_split,
    load_raw_data,
)
from atomity_forecast.models import (
    AR1Model,
    DirectRidgeModel,
    EnsembleModel,
    LastValueBaseline,
)


def generate_predictions(
    data_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else PATHS.data_dir
    output_path = Path(output_path) if output_path is not None else PATHS.predictions_csv

    df_readings, _ = load_raw_data(data_dir)
    targets_df = build_targets(df_readings)
    train_df, holdout_df = get_train_holdout_split(
        targets_df, holdout_len=CONFIG.holdout_steps
    )

    y_train = train_df["aggregate_cpu_utilization"].to_numpy()
    horizon = len(holdout_df)

    # The active VM count is a constant/persistence target in this release.
    # Taking the last observed value keeps the pipeline smoke-safe on fleets
    # of any size instead of hard-coding 10,000.
    inferred_active_vms = int(round(train_df["active_vm_count"].iloc[-1]))

    # Frozen winning Ensemble: LastValue + AR(1) + direct multi-step Ridge.
    ensemble = EnsembleModel(
        [
            LastValueBaseline(),
            AR1Model(),
            DirectRidgeModel(alpha=1.0, lags=6),
        ]
    )
    ensemble.fit(y_train, horizon=horizon)
    pred_cpu = ensemble.predict(horizon=horizon)

    predictions_df = pd.DataFrame(
        {
            "timestamp": holdout_df["timestamp"].values,
            "active_vm_count": inferred_active_vms,
            "aggregate_cpu_utilization": np.round(pred_cpu, 6),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved {len(predictions_df)} holdout predictions to {output_path}")
    return predictions_df


if __name__ == "__main__":
    generate_predictions()
