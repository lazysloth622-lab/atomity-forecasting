\# Cloud Workload Forecasting Pipeline (Azure Public Dataset V2)



Production-grade time-series forecasting pipeline developed for the Atomity ML Engineering take-home assessment. Evaluates aggregate compute demand across 10,000 virtual machines over a 2-hour lookahead horizon (24 steps at 5-minute intervals).



\---



\## 1. Project Overview \& Target Formulation

The pipeline predicts two concurrent targets over a 24-step holdout horizon:

1\. `active\_vm\_count(t)`: The count of distinct VMs reporting valid CPU telemetry at timestep t.

2\. `aggregate\_cpu\_utilization(t)`: The arithmetic mean of `avg\_cpu` across all active VMs at timestep t, strictly bounded in \[0, 100].



\### Architecture \& Hygiene Highlights

\- \*\*Strict Temporal Boundary:\*\* Zero data leakage. The first 110 timesteps serve as the development/training window; the final 24 timesteps represent the strictly unseen holdout horizon.

\- \*\*Dynamic Inference (Smoke-Test Compliant):\*\* No hard-coded VM counts, time-series lengths, or dataset indices. Evaluated and verified against synthetic datasets with varying VM counts.

\- \*\*Model Freezing Checkpoint:\*\* Architecture selection was pre-registered in `reports/selection\_rule.md` and committed under git tag `model-frozen` prior to running single-pass holdout scoring.

\- \*\*Zero Number Hallucination:\*\* Every statistic, table, and markdown metric is generated directly via code into `reports/numbers.json` and `reports/tables/`.



\---



\## 2. Directory Layout

```text

atomity-forecasting/

├── challenge\_data/              # Raw telemetry files (compressed)

├── reports/

│   ├── figures/                 # EDA and holdout forecast plots

│   ├── tables/                  # CV metrics and holdout evaluation summaries

│   ├── numbers.json             # Source-of-truth JSON numbers for reporting

│   ├── selection\_rule.md        # Pre-registered model selection protocol

│   └── final\_report.md          # Programmatically rendered technical report

├── src/atomity\_forecast/        # Core package source

│   ├── config.py                # Centralized paths and hyperparameters

│   ├── data.py                  # Ingestion, validation, and temporal splitting

│   ├── eda.py                   # Development-window EDA runner

│   ├── metrics.py               # MAE, RMSE, WAPE, and Peak-Quartile MAE

│   ├── models.py                # Model ladder (Baselines, AR1, ETS, ML, Ensemble)

│   ├── split.py                 # Rolling-origin cross-validation engine

│   ├── validate.py              # 6-fold benchmark runner

│   ├── predict.py               # Holdout predictions generator

│   ├── policy.py                # Uncertainty quantification and headroom policy

│   └── generate\_report.py       # Report template generator

├── tests/                       # Complete pytest suite (leakage, smoke, bounds)

├── evaluate\_holdout.py          # Single-pass frozen holdout scoring script

├── predictions.csv              # Required final submission artifact (24 rows)

├── pyproject.toml               # PEP 621 package build configuration

└── requirements.txt             # Pinned environment dependencies

