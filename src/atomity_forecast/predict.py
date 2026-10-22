"""
Holdout prediction generator for Atomity Cloud Workload Forecasting.
Generates predictions.csv using the frozen winning Ensemble model.
"""
import pandas as pd
import numpy as np

from atomity_forecast.config import CONFIG, PATHS
from atomity_forecast.data import load_raw_data, build_targets, get_train_holdout_split
from atomity_forecast.models import (
    LastValueBaseline,
    AR1Model,
    DirectRidgeModel,
    EnsembleModel,
)


def generate_predictions() -> pd.DataFrame:
    df_readings, _ = load_raw_data(PATHS.data_dir)
    targets_df = build_targets(df_readings)
    train_df, holdout_df = get_train_holdout_split(targets_df, holdout_len=CONFIG.holdout_steps)

    y_train = train_df["aggregate_cpu_utilization"].to_numpy()
    horizon = len(holdout_df)

    # Dynamic active_vm_count (prevents smoke test failures on non-10k VM sets)
    inferred_active_vms = int(round(train_df["active_vm_count"].iloc[-1]))

    # Instantiate and fit frozen winning Ensemble model
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

    # Save artifact strictly matching task specifications
    predictions_df.to_csv(PATHS.predictions_csv, index=False)
    print(f"Saved {len(predictions_df)} holdout predictions to {PATHS.predictions_csv}")
    return predictions_df


if __name__ == "__main__":
    generate_predictions()