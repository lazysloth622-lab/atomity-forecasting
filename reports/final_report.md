# Cloud Workload Forecasting: Technical Report
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
- The aggregate workload is highly predictable. The pre-registered **Ensemble**
  (LastValue + AR(1) + direct multi-step Ridge) reaches rolling-origin CV MAE
  **0.1849%** vs persistence **0.2288%**
  (19.2% relative improvement on CV).
- Single-pass unseen holdout: MAE **0.1046%**, RMSE **0.1289%**,
  WAPE **1.13%**, peak-quartile MAE **0.1023%**.
- `active_vm_count` is constant at 10000 VMs; persistence is exact
  (holdout MAE **0.0 VMs**, CV MAE **0.0 VMs**).
- A residual-based 95% band covers **100.0%** of holdout actuals
  with 0 breaches.

Model identity was frozen (`git tag model-frozen`) using development-window CV before any holdout scoring.

---

## 2. Data Understanding and Preprocessing
All statistics below use the development window (steps 0–109) unless stated.

- **Cadence/coverage:** 134 unique timestamps on a strict 300-second grid.
- **Duplicates:** 0 duplicate `(vm_id, timestamp)` pairs.
- **Missingness:** 0.00% missing `avg_cpu`; all 1,340,000 observations valid, so no imputation is needed.
- **Target construction:** `active_vm_count` = distinct `vm_id` with a valid reading at t (once per VM);
  `aggregate_cpu_utilization` = mean of `avg_cpu` over those VMs, observed readings only, no gap filling.
- **VM churn:** **0** VMs created and **0** deleted strictly inside the observation window (verified in code from `timestamp_vm_created`/`timestamp_vm_deleted`; all VMs exist at the trace origin and the earliest deletion occurs after the last reading).

### 2.1 Leakage controls
The final 24 timestamps are never used in fitting, feature construction, scaling, tuning or selection.
`tests/test_leakage.py::test_perturbation_no_leakage` proves that perturbing holdout target values leaves
forecasts byte-identical. Rolling-origin folds are contained entirely inside the 110-step development window.

---

## 3. Exploratory Workload Analysis

### 3.1 Aggregate behaviour
![Development window workload](figures/eda_training_series.png)

- Aggregate CPU mean **9.414%**, std **0.194 pp**,
  range 9.112%–9.912%, median **9.378%**.
- The fleet-wide average is exceptionally stable: uncorrelated VM-level noise cancels across 10,000 instances.

### 3.2 Temporal structure
![Autocorrelation](figures/eda_autocorrelation.png)

- Lag-1 autocorrelation **0.6621** (strong short-range persistence);
  lag-24 **-0.0199** — no diurnal cycle is detectable in an 11.1-hour trace.
- White-noise significance band: +/-0.1869.
- Calendar features are therefore uninformative and were excluded; all predictive signal sits in lags 1–6.

### 3.3 VM-level distribution and predictability
- Median individual reading: **5.12%**;
  **48.9%** of readings are below 5% CPU and
  **1.27%** exceed 80% — a heavily right-skewed, mostly idle fleet.
- Mean per-VM persistence MAE: **1.944%**
  (median across VMs **0.977%**) versus aggregate persistence MAE
  **0.1211%** — the aggregate is roughly an order of magnitude smoother
  than a single VM series.

### 3.4 Workload segments (metadata-backed)
| VM category | VMs | Mean CPU (%) | Temporal std (pp) | Min–Max (%) |
| :--- | ---: | ---: | ---: | :--- |
| Delay-insensitive | 6180 | 8.257 | 0.217 | 7.900–8.878 |
| Unknown | 729 | 10.355 | 0.629 | 9.165–11.561 |
| Interactive | 3091 | 11.505 | 0.243 | 10.951–12.017 |


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
| Model | MAE mean | MAE std | Peak MAE mean | RMSE mean | WAPE mean (%) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Ensemble** | 0.1849 | 0.0635 | 0.1658 | 0.2136 | 1.96% |
| AR1 | 0.2006 | 0.0569 | 0.1957 | 0.2285 | 2.12% |
| TrainMean | 0.2078 | 0.0537 | 0.2238 | 0.2362 | 2.20% |
| DirectRidge | 0.2098 | 0.0764 | 0.2542 | 0.2415 | 2.22% |
| DirectLightGBM | 0.2113 | 0.0700 | 0.2430 | 0.2435 | 2.24% |
| HoltDamped | 0.2251 | 0.0939 | 0.1468 | 0.2581 | 2.38% |
| LastValue | 0.2288 | 0.1305 | 0.1392 | 0.2631 | 2.42% |

### 6.2 Single-pass holdout (unseen final 24 steps)
| Metric | Ensemble | Persistence | Train mean |
| :--- | ---: | ---: | ---: |
| MAE (%) | 0.1046 | 0.1468 | 0.1746 |
| RMSE (%) | 0.1289 | n/a | n/a |
| WAPE (%) | 1.13 | n/a | n/a |
| Peak-quartile MAE (%) | 0.1023 | n/a | n/a |

Relative MAE improvement over persistence on the holdout:
**28.7%**.
![Holdout forecast with 95% band](figures/holdout_forecast_with_intervals.png)

---

## 7. Error and Failure Analysis
- Step 1 (5 min ahead) MAE: **0.0343%**
- Step 12 (1 hour ahead) MAE: **0.0018%**
- Step 24 (2 hours ahead) MAE: **0.0631%**
- Full per-step series: `reports/tables/holdout_error_by_step.csv`.

- **Peak periods:** peak-quartile MAE **0.1023%** is close to overall MAE — the model
  does not degrade materially on the busiest quartile of timestamps.
- **Failure mode:** errors grow with horizon as AR(1) mean-reverts and direct Ridge shrinks toward the
  training mean; a sudden level shift not present in the 110-step history would be missed. The single
  holdout window is calm, so this risk is characterized via the 6 CV folds rather than one score.
- WAPE is undefined when the denominator (sum of actuals) is zero; the implementation returns NaN in that
  case (unit-tested) instead of dividing by zero.

---

## 8. Capacity Planning Recommendation
- One-step residual sigma = **0.1580 pp**; 95% interval half-width
  +/-**0.3097 pp** (constant-width assumption, Section 9).
- Provisioning rule targeting at most 80% utilization:
  **capacity_fraction = max_t(y_hat(t) + 1.96 * sigma) / 0.80 = 0.1211**
  of the *current* fleet. Because the value is below 1, on aggregate-mean utilization the fleet is
  over-provisioned and could be rightsized to about 0.12x
  its current capacity while retaining a 95% safety margin.
- **Important caveat:** this rule uses the fleet MEAN. About 1.27%
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
