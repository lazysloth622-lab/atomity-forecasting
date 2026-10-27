# Cloud Workload Forecasting Pipeline (Azure Public Dataset V2)

End-to-end time-series forecasting pipeline for the Atomity ML take-home:
aggregate compute demand across 10,000 VMs, forecasting a 2-hour horizon
(24 steps at 5-minute cadence) for two targets:

1. `active_vm_count(t)` — distinct VMs with a valid CPU reading at timestamp `t`.
2. `aggregate_cpu_utilization(t)` — mean of VM-level `avg_cpu` at `t`, in `[0, 100]`.

## 1. Environment setup

Python 3.11+ required.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"          # core + pytest (preferred)
# alternative fully pinned install: pip install -r requirements.txt
```

Verify:

```bash
pytest -q
```

## 2. Dataset placement

Download the fixed `data-v1` release (`data-v1.tar.gz`) from the challenge
repository's GitHub Releases and extract it so the layout is:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

Optional integrity check: `cd challenge_data && sha256sum -c SHA256SUMS`.
The data directory is git-ignored and is never committed.

## 3. Reproduce the submission artifact

One command regenerates `predictions.csv` (24 rows, schema per `SUBMISSION.md`):

```bash
python -m atomity_forecast.cli predict --data challenge_data --output predictions.csv
# equivalent: python -m atomity_forecast.predict
```

The CLI flags make the pipeline work with a reviewer-supplied smoke dataset;
nothing is hard-coded to 10,000 VMs or 134 timesteps.

## 4. Full pipeline (figures, tables, report)

Run in order from the repository root:

```bash
python -m atomity_forecast.eda            # EDA stats + autocorrelation figures
python -m atomity_forecast.segments       # churn verification + VM segment analysis
python -m atomity_forecast.validate       # 6-fold rolling-origin model benchmark
python evaluate_holdout.py                # single-pass frozen holdout scoring
python -m atomity_forecast.policy         # uncertainty band + capacity policy
python -m atomity_forecast.generate_figures
python -m atomity_forecast.generate_report
```

Outputs:

- `predictions.csv` — required submission artifact.
- `reports/numbers.json` — single source of truth for every reported statistic.
- `reports/tables/` — CV fold/summary, holdout metrics, per-step errors, segments, EDA.
- `reports/figures/` — EDA, validation and holdout plots.
- `reports/final_report.md` — rendered technical report (all numbers injected from code).
- `reports/selection_rule.md` — pre-registered model selection protocol.

## 5. Modeling decisions (summary)

- **Leakage-safe by construction:** the final 24 timestamps are never used in
  fitting, feature engineering, scaling, tuning or selection; a perturbation
  test proves holdout values cannot change predictions.
- **Validation:** chronological 110/24 outer split plus 6 expanding-window
  rolling-origin folds (horizon 24, origins from t=50) inside the development
  window. No random shuffling.
- **Model ladder:** TrainMean / LastValue baselines, AR(1), damped Holt,
  direct multi-step Ridge and LightGBM; frozen winner is an ensemble of
  LastValue + AR(1) + direct Ridge, selected solely on CV MAE (Peak MAE tie-break).
- **`active_vm_count`:** the curated fleet is static, so last-value persistence
  is exact (holdout MAE 0.0 VMs) and stays valid on non-constant smoke datasets.
- **Uncertainty/policy:** empirical residual sigma gives a 95% band and an
  80%-target-utilization capacity rule; assumptions and limitations are stated
  in `reports/final_report.md` sections 8–9.

## 6. Project layout

```text
├── src/atomity_forecast/
│   ├── cli.py               # argparse entry point (atomity-predict)
│   ├── config.py            # paths and hyperparameters
│   ├── data.py              # ingestion, target aggregation, temporal split
│   ├── eda.py               # development-window EDA
│   ├── segments.py          # metadata churn check and segment analysis
│   ├── metrics.py           # MAE, RMSE, WAPE, peak-quartile MAE
│   ├── models.py            # baseline / statistical / ML / ensemble models
│   ├── split.py             # rolling-origin fold generator
│   ├── validate.py          # cross-validation benchmark
│   ├── predict.py           # predictions.csv generator
│   ├── policy.py            # prediction intervals and capacity policy
│   ├── generate_figures.py  # report figures
│   └── generate_report.py   # canonical final_report.md renderer
├── tests/                   # leakage, schema, edge-case, ladder, smoke tests
├── evaluate_holdout.py      # one-pass holdout scoring script
├── verify_data.py           # minimal data integrity check
├── reports/                 # generated artifacts (see section 4)
├── pyproject.toml
└── requirements.txt
```
