"""
Generate holdout set predictions (horizon H=24) using DirectRidgeModel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.atomity_forecast.data import (
    load_raw_data,
    build_targets,
    get_train_holdout_split,
)
from src.atomity_forecast.models import DirectRidgeModel

HORIZON = 24
ACTIVE_VM_COUNT_DEFAULT = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DirectRidgeModel on development data and predict holdout horizon."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("challenge_data"),
        help="Path to directory containing raw trace data files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Destination path for final predictions CSV",
    )
    return parser.parse_args()


def run_prediction(data_dir: Path, output_file: Path) -> None:
    print(f"[1/4] Loading raw data from '{data_dir}'...")
    # load_raw_data returns two dataframes: readings and metadata
    df_readings, df_meta = load_raw_data(data_dir)

    print("[2/4] Aggregating VM readings into target time series...")
    targets_df = build_targets(df_readings)

    # Split: first 110 steps = training, last 24 steps = holdout
    train_df, holdout_df = get_train_holdout_split(targets_df, holdout_len=HORIZON)

    print(f"[3/4] Fitting DirectRidgeModel on {len(train_df)} training steps...")
    y_train = train_df["aggregate_cpu_utilization"].to_numpy()

    model = DirectRidgeModel()
    model.fit(y_train, horizon=HORIZON)

    print(f"[4/4] Forecasting next {HORIZON} steps...")
    cpu_pred = model.predict(horizon=HORIZON)

    # Build predictions dataframe matching holdout length and index
    pred_df = pd.DataFrame()
    if "timestamp" in holdout_df.columns:
        pred_df["timestamp"] = holdout_df["timestamp"].values
    elif "step" in holdout_df.columns:
        pred_df["step"] = holdout_df["step"].values
    else:
        pred_df["step"] = list(range(len(train_df), len(train_df) + HORIZON))

    pred_df["active_vm_count"] = ACTIVE_VM_COUNT_DEFAULT
    pred_df["aggregate_cpu_utilization"] = cpu_pred

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_file, index=False)
    print(f"Done! Successfully saved {len(pred_df)} rows to '{output_file}'")


def main() -> None:
    args = parse_args()
    run_prediction(data_dir=args.data, output_file=args.output)


if __name__ == "__main__":
    main()