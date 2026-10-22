from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

@dataclass(frozen=True)
class PathConfig:
    data_dir: Path = PROJECT_ROOT / "challenge_data"
    cpu_readings: Path = PROJECT_ROOT / "challenge_data" / "vm_cpu_readings.csv.gz"
    vm_metadata: Path = PROJECT_ROOT / "challenge_data" / "vm_metadata.csv.gz"
    reports_dir: Path = PROJECT_ROOT / "reports"
    tables_dir: Path = PROJECT_ROOT / "reports" / "tables"
    figures_dir: Path = PROJECT_ROOT / "reports" / "figures"
    numbers_json: Path = PROJECT_ROOT / "reports" / "numbers.json"
    predictions_csv: Path = PROJECT_ROOT / "predictions.csv"

@dataclass(frozen=True)
class ForecastingConfig:
    timestep_seconds: int = 300
    horizon_steps: int = 24
    holdout_steps: int = 24
    min_origin: int = 50
    origin_step: int = 8
    random_seed: int = 42

PATHS = PathConfig()
CONFIG = ForecastingConfig()