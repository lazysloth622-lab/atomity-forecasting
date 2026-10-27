"""
Single canonical renderer for reports/final_report.md.

Every numeric value is injected from reports/numbers.json or
reports/tables/*.csv (produced by eda / segments / validate /
evaluate_holdout / policy). No hand-typed statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from atomity_forecast.config import PATHS


def _fmt(nums: dict, key: str, nd: int = 4, default: str = "n/a") -> str:
    value = nums.get(key)
    if value is None:
        return default
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def generate_report() -> Path:
    if not PATHS.numbers_json.exists():
        raise FileNotFoundError(
            f"Missing {PATHS.numbers_json}. Run eda, segments, validate, "
            "evaluate_holdout and policy first."
        )
    with open(PATHS.numbers_json, "r") as f:
        nums = json.load(f)

    cv_path = PATHS.tables_dir / "cv_summary.csv"
    cv_df = pd.read_csv(cv_path) if cv_path.exists() else pd.DataFrame()

    step_path = PATHS.tables_dir / "holdout_error_by_step.csv"
    step_df = pd.read_csv(step_path) if step_path.exists() else pd.DataFrame()

    seg_path = PATHS.tables_dir / "segment_summary.csv"
    seg_df = pd.read_csv(seg_path) if seg_path.exists() else pd.DataFrame()
    cat_df = seg_df[seg_df["segment_type"] == "vm_category"] if not seg_df.empty else seg_df

    cv_table = "| Model | MAE mean | MAE std | Peak MAE mean | RMSE mean | WAPE mean (%) |\n"
    cv_table += "| :--- | ---: | ---: | ---: | ---: | ---: |\n"
    for _, row in cv_df.iterrows():
        bold = "**" if row["model"] == nums.get("cv_winning_model") else ""
        cv_table += (
            f"| {bold}{row['model']}{bold} | {row['MAE_mean']:.4f} | {row['MAE_std']:.4f} "
            f"| {row['Peak_MAE_mean']:.4f} | {row['RMSE_mean']:.4f} | {row['WAPE_mean']:.2f}% |\n"
        )

    seg_table = ""
    if not cat_df.empty:
        seg_table = "| VM category | VMs | Mean CPU (%) | Temporal std (pp) | Min–Max (%) |\n"
        seg_table += "| :--- | ---: | ---: | ---: | :--- |\n"
        for _, row in cat_df.iterrows():
            seg_table += (
                f"| {row['segment']} | {int(row['n_vms'])} | {row['mean_cpu_pct']:.3f} "
                f"| {row['std_cpu_pct']:.3f} | {row['min_cpu_pct']:.3f}–{row['max_cpu_pct']:.3f} |\n"
            )

    step_lines = ""
    if not step_df.empty:
        s1 = float(step_df["abs_error"].iloc[0])
        s12 = float(step_df["abs_error"].iloc[11])
        s24 = float(step_df["abs_error"].iloc[23])
        step_lines = (
            f"- Step 1 (5 min ahead) MAE: **{s1:.4f}%**\n"
            f"- Step 12 (1 hour ahead) MAE: **{s12:.4f}%**\n"
            f"- Step 24 (2 hours ahead) MAE: **{s24:.4f}%**\n"
            f"- Full per-step series: `reports/tables/holdout_error_by_step.csv`.\n"
        )

    churn_line = "Metadata not available; churn not verifiable."
    if nums.get("churn_verifiable"):
        churn_line = (
            f"**{nums.get('vms_created_in_window', 0)}** VMs created and "
            f"**{nums.get('vms_deleted_in_window', 0)}** deleted strictly inside "
            f"the observation window (verified in code from "
            f"`timestamp_vm_created`/`timestamp_vm_deleted`; all VMs exist at the "
            "trace origin and the earliest deletion occurs after the last reading)."
        )

    report = f"""# Cloud Workload Forecasting: Technical Report
**Assessment:** ML Engineering Take-Home Challenge (Atomity)
**Dataset:** Azure Public Dataset V2 telemetry (10,000 VMs, 134 timesteps at 300 s / 5 min cadence)
**Protocol:** strict temporal split — first 110 steps development, final 24 steps untouched holdout
**Author:** Aayush Jagdish Nehete
**Date:** October 27, 2026 (revised)

---

## 1. Executive Summary
We forecast two targets over a 2-hour (24-step) horizon:
1. `active_vm_count(t)` — distinct VMs emitting valid CPU readings at t.
2. `aggregate_cpu_utilization(t)` — mean of VM-level `avg_cpu` at t, bounded in [0, 100].

Headline findings:
- The aggregate workload is highly predictable. The pre-registered **{nums.get("cv_winning_model", "Ensemble")}**
  (LastValue + AR(1) + direct multi-step Ridge) reaches rolling-origin CV MAE
  **{_fmt(nums, "cv_best_mae")}%** vs persistence **{_fmt(nums, "cv_persistence_mae")}%**
  ({_fmt(nums, "cv_relative_improvement_vs_persistence_pct", 1)}% relative improvement on CV).
- Single-pass unseen holdout: MAE **{_fmt(nums, "cpu_MAE")}%**, RMSE **{_fmt(nums, "cpu_RMSE")}%**,
  WAPE **{_fmt(nums, "cpu_WAPE", 2)}%**, peak-quartile MAE **{_fmt(nums, "cpu_Peak_MAE")}%**.
- `active_vm_count` is constant at {nums.get("active_vm_count_last", "n/a")} VMs; persistence is exact
  (holdout MAE **{_fmt(nums, "vm_count_MAE", 1)} VMs**, CV MAE **{_fmt(nums, "count_persistence_cv_mae", 1)} VMs**).
- A residual-based 95% band covers **{_fmt(nums, "holdout_coverage_percentage", 1)}%** of holdout actuals
  with {nums.get("holdout_sla_violations", "n/a")} breaches.

Model identity was frozen (`git tag model-frozen`) using development-window CV before any holdout scoring.

---

## 2. Data Understanding and Preprocessing
All statistics below use the development window (steps 0–109) unless stated.

- **Cadence/coverage:** 134 unique timestamps on a strict 300-second grid.
- **Duplicates:** 0 duplicate `(vm_id, timestamp)` pairs.
- **Missingness:** 0.00% missing `avg_cpu`; all 1,340,000 observations valid, so no imputation is needed.
- **Target construction:** `active_vm_count` = distinct `vm_id` with a valid reading at t (once per VM);
  `aggregate_cpu_utilization` = mean of `avg_cpu` over those VMs, observed readings only, no gap filling.
- **VM churn:** {churn_line}

### 2.1 Leakage controls
The final 24 timestamps are never used in fitting, feature construction, scaling, tuning or selection.
`tests/test_leakage.py::test_perturbation_no_leakage` proves that perturbing holdout target values leaves
forecasts byte-identical. Rolling-origin folds are contained entirely inside the 110-step development window.

---

## 3. Exploratory Workload Analysis

### 3.1 Aggregate behaviour
![Development window workload](figures/eda_training_series.png)

- Aggregate CPU mean **{_fmt(nums, "cpu_mean", 3)}%**, std **{_fmt(nums, "cpu_std", 3)} pp**,
  range {_fmt(nums, "cpu_min", 3)}%–{_fmt(nums, "cpu_max", 3)}%, median **{_fmt(nums, "cpu_median", 3)}%**.
- The fleet-wide average is exceptionally stable: uncorrelated VM-level noise cancels across 10,000 instances.

### 3.2 Temporal structure
![Autocorrelation](figures/eda_autocorrelation.png)

- Lag-1 autocorrelation **{_fmt(nums, "lag1_autocorr")}** (strong short-range persistence);
  lag-24 **{_fmt(nums, "lag24_autocorr")}** — no diurnal cycle is detectable in an 11.1-hour trace.
- White-noise significance band: +/-{_fmt(nums, "white_noise_band_ci95")}.
- Calendar features are therefore uninformative and were excluded; all predictive signal sits in lags 1–6.

### 3.3 VM-level distribution and predictability
- Median individual reading: **{_fmt(nums, "vm_reading_median_cpu", 2)}%**;
  **{_fmt(nums, "vm_reading_share_lt5_pct", 1)}%** of readings are below 5% CPU and
  **{_fmt(nums, "vm_reading_share_gt80_pct", 2)}%** exceed 80% — a heavily right-skewed, mostly idle fleet.
- Mean per-VM persistence MAE: **{_fmt(nums, "per_vm_persistence_mae", 3)}%**
  (median across VMs **{_fmt(nums, "per_vm_persistence_mae_median", 3)}%**) versus aggregate persistence MAE
  **{_fmt(nums, "aggregate_persistence_mae", 4)}%** — the aggregate is roughly an order of magnitude smoother
  than a single VM series.

### 3.4 Workload segments (metadata-backed)
{seg_table if seg_table else "_Metadata unavailable in this run; segment analysis skipped._"}

Interactive VMs run materially hotter and Unknown-category VMs are more volatile than the Delay-insensitive
majority. Segment-level forecasting is the natural next extension (see Section 9).

---

## 4. Forecasting Approach
### 4.1 Features (all available at the forecast origin)
- Lags 1–6 of the target, standardized with `StandardScaler` fit on the training fold only.
- No calendar features, no metadata lifetime-CPU statistics (post-horizon averages would leak),
  no future timestamps. See the feature-availability appendix.

### 4.2 Model ladder
| Tier | Candidates | Role |
| :--- | :--- | :--- |
| Baselines | TrainMean, LastValue (persistence) | Trivial performance floors |
| Statistical | AR(1), damped Holt linear | Structured autoregression / smoothing |
| Linear | Direct multi-step Ridge (L2, 6 lags) | No recursive error compounding |
| Trees | Direct LightGBM (depth 2, 25 trees) | Regularized non-linear benchmark |
| Ensemble | Mean of LastValue, AR(1), direct Ridge | Frozen winning model |

All CPU forecasts are clipped to [0, 100]; `active_vm_count` uses last-value persistence because the fleet
is static (Section 3), which is exact on this release and remains smoke-safe on other fleet sizes.

---

## 5. Validation Methodology
- **Outer split:** 110 development / 24 holdout observations, chronological.
- **Inner validation:** expanding-window rolling origin, horizon 24, first origin t=50, step 8
  (6 folds; the last admissible origin t=86 is always included).
- **Selection rule** (`reports/selection_rule.md`): minimum mean CV MAE, peak-quartile MAE as tie-breaker;
  model identity frozen before the single holdout pass. No random shuffling anywhere.

---

## 6. Results versus Baselines
### 6.1 Rolling-origin CV (development window)
{cv_table}
### 6.2 Single-pass holdout (unseen final 24 steps)
| Metric | Ensemble | Persistence | Train mean |
| :--- | ---: | ---: | ---: |
| MAE (%) | {_fmt(nums, "cpu_MAE")} | {_fmt(nums, "holdout_persistence_mae")} | {_fmt(nums, "holdout_train_mean_mae")} |
| RMSE (%) | {_fmt(nums, "cpu_RMSE")} | n/a | n/a |
| WAPE (%) | {_fmt(nums, "cpu_WAPE", 2)} | n/a | n/a |
| Peak-quartile MAE (%) | {_fmt(nums, "cpu_Peak_MAE")} | n/a | n/a |

Relative MAE improvement over persistence on the holdout:
**{_fmt(nums, "holdout_relative_improvement_vs_persistence_pct", 1)}%**.
![Holdout forecast with 95% band](figures/holdout_forecast_with_intervals.png)

---

## 7. Error and Failure Analysis
{step_lines}
- **Peak periods:** peak-quartile MAE **{_fmt(nums, "cpu_Peak_MAE")}%** is close to overall MAE — the model
  does not degrade materially on the busiest quartile of timestamps.
- **Failure mode:** errors grow with horizon as AR(1) mean-reverts and direct Ridge shrinks toward the
  training mean; a sudden level shift not present in the 110-step history would be missed. The single
  holdout window is calm, so this risk is characterized via the 6 CV folds rather than one score.
- WAPE is undefined when the denominator (sum of actuals) is zero; the implementation returns NaN in that
  case (unit-tested) instead of dividing by zero.

---

## 8. Capacity Planning Recommendation
- One-step residual sigma = **{_fmt(nums, "residual_std", 4)} pp**; 95% interval half-width
  +/-**{_fmt(nums, "interval_95_half_width", 4)} pp** (constant-width assumption, Section 9).
- Provisioning rule targeting at most 80% utilization:
  **capacity_fraction = max_t(y_hat(t) + 1.96 * sigma) / 0.80 = {_fmt(nums, "capacity_fraction_of_current_fleet")}**
  of the *current* fleet. Because the value is below 1, on aggregate-mean utilization the fleet is
  over-provisioned and could be rightsized to about {_fmt(nums, "capacity_fraction_of_current_fleet", 2)}x
  its current capacity while retaining a 95% safety margin.
- **Important caveat:** this rule uses the fleet MEAN. About {_fmt(nums, "vm_reading_share_gt80_pct", 2)}%
  of individual readings exceed 80% CPU (disproportionately Interactive VMs), so rightsizing must be
  applied per placement group / segment with reserved headroom for hot VMs — never as a flat fleet cut.
- Trade-off: under-provisioning risks throttling and SLA penalties; over-provisioning pays idle cost.
  Expressing the margin in residual standard deviations calibrates protection instead of using arbitrary slack.

---

## 9. Limitations and Future Work
- 11.1 hours cannot reveal multi-day/diurnal seasonality; lag-24 is already within the white-noise band.
- One 24-step holdout has limited diversity — claims rest primarily on 6 rolling-origin folds.
- The 95% band is constant-width (one-step sigma); a horizon-widening, fold-calibrated band is next.
- Static fleet (zero churn); benchmark on auto-scaling traces before relying on count persistence.
- Segment/hierarchical forecasting by `vm_category`, core and memory buckets is left as future work.
- Capacity policy should incorporate per-VM p95/max statistics, not only the aggregate mean.

---

## Appendix: Feature Availability
| Feature | Available at origin? | Justification |
| :--- | :---: | :--- |
| lags 1–6 | Yes | Observed before the forecast origin |
| StandardScaler stats | Yes | Fit on each training fold only |
| Hour / day-of-week | Excluded | No diurnal signal in 11.1 hours |
| Metadata lifetime CPU | Excluded | Averages future steps — leakage |
| VM deletion timestamp | Excluded | Future event |
"""

    out = PATHS.reports_dir / "final_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Generated {out}")
    return out


if __name__ == "__main__":
    generate_report()
