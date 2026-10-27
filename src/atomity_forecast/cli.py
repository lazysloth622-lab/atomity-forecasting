"""
Command-line interface for the forecasting pipeline.

Examples:
    python -m atomity_forecast.cli predict --data challenge_data --output predictions.csv
    python -m atomity_forecast.cli --data challenge_data --output predictions.csv  # defaults to predict
    atomity-predict --data challenge_data --output predictions.csv
"""
from __future__ import annotations

import argparse

from atomity_forecast.config import PATHS
from atomity_forecast.predict import generate_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomity cloud workload forecasting command line interface."
    )
    parser.add_argument(
        "--data",
        default=str(PATHS.data_dir),
        help="Directory containing vm_cpu_readings.csv.gz and vm_metadata.csv.gz.",
    )
    parser.add_argument(
        "--output",
        default=str(PATHS.predictions_csv),
        help="Destination path for predictions.csv.",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser(
        "predict",
        help="Generate the holdout predictions.csv artifact (default command).",
    )
    p.add_argument("--data", default=argparse.SUPPRESS)
    p.add_argument("--output", default=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    generate_predictions(data_dir=args.data, output_path=args.output)


if __name__ == "__main__":
    main()
