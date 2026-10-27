# Cloud Workload Forecasting: Technical Report
**Assessment:** ML Engineering Take-Home Challenge (Atomity)  
**Dataset:** Azure Public Dataset V2 Telemetry (10,000 VMs sampled over 134 timesteps at 300 s / 5 min intervals)  
**Evaluation Protocol:** Strict Temporal Split (First 110 steps for development, final 24 steps as strictly unseen holdout horizon)  
**Author:** Aayush Jagdish Nehete  
**Date:** October 26, 2026

---

## 1. Executive Summary & Problem Formulation
The objective is multi-step lookahead forecasting of aggregate cloud compute demand over a 2-hour horizon (24 steps at 5-minute cadence) for two concurrent targets:
1. `active_vm_count(t)`: The count of distinct virtual machines emitting valid CPU readings.
2. `aggregate_cpu_utilization(t)`: The arithmetic mean of `avg_cpu` across active VMs, bounded in [0, 100].

### Headline Findings:
* **Predictability:** The aggregate workload is highly predictable. The pre-registered winning **Ensemble** model (LastValue + AR(1) + DirectRidge) achieved a rolling-origin CV MAE of **0.1849%** (with the individual Direct Ridge candidate at **0.2098%**) and an unseen holdout MAE of **0.1046%**, significantly outperforming naive baselines.
* **Trivial Target:** `active_vm_count` is a constant 10,000 across all 134 timestamps. Its forecast is dynamically inferred from the final observed step and is exact (MAE = 0.0 VMs).
* **Operational Recommendation:** We recommend an empirical capacity provisioning policy based on step-to-step volatility margin: $C(t) = \frac{\hat{y}(t) + 1.96 \cdot \sigma}{0.80}$. This maintains 100.0% empirical coverage across the holdout horizon with zero observed SLA breaches.

All architectural decisions and model selections were finalized and frozen (`git tag model-frozen`) using rolling-origin cross-validation on the development window before evaluating the holdout horizon.

---

## 2. Data Understanding and Preprocessing
The dataset is a curated subset of the Azure Public Dataset V2. All telemetry statistics below are strictly computed on the training period (steps 0 to 109):

* **Cadence & Coverage:** Timestamps are strictly 300 seconds (5 minutes) apart. There are 134 unique timestamps.
* **Duplicates:** 0 duplicate `(vm_id, timestamp)` pairs were found.
* **Missingness:** The curated dataset contains 0.00% missing `avg_cpu` readings across all 1.34 million observations.
* **Active VM Fleet Count:** Exactly 10,000 VMs. Fleet count remained constant across all development steps (`active_vm_count_constant = True`).
* **VM Churn:** 0 VMs were created or deleted within the observation window.
* **Target Construction:**
  * `active_vm_count(t)` = distinct active VMs emitting readings at step $t$.
  * `aggregate_cpu_utilization(t)` = arithmetic mean of `avg_cpu` across active VMs at step $t$.

### 2.1 Leakage Controls
The holdout window (final 24 timestamps) was strictly untouched during feature construction, scaling, and hyperparameter tuning. Perturbation unit tests (`test_perturbation_no_leakage`) confirm that modifying holdout values produces zero change in pipeline predictions.

---

## 3. Exploratory Workload Analysis

### 3.1 Aggregate Behavior
![Development Window Workload](figures/eda_training_series.png)  
*Figure 1: Aggregate CPU utilization and active VM count over the 11.1-hour trace.*

* **Aggregate CPU Mean:** 9.414% (Standard Deviation: 0.194%).
* **Observed CPU Range:** Min 9.112%, Median 9.378%, Max 9.912%.
* **Operational Meaning:** Aggregate fleet CPU exhibits strong central limit smoothing across 10,000 instances, behaving as a low-volatility, stationary series.

### 3.2 Temporal Patterns & Autocorrelation
![Autocorrelation](figures/eda_autocorrelation.png)  
*Figure 2: Training set aggregate CPU autocorrelation function.*

* **Autocorrelation Structure:** Lag-1 autocorrelation $r_1 = 0.6621$, confirming strong local temporal persistence. Lag-24 autocorrelation drops to $r_{24} = -0.0199$, indicating no diurnal periodicity within this 11-hour sample.
* **White-Noise Significance Threshold:** $\pm 1.96 / \sqrt{N} = \pm 0.1869$.

### 3.3 Predictability: Individual VMs vs. Fleet Aggregate
* **Aggregate Persistence Error (LastValue MAE on aggregate):** 0.2288%.
* **Mean Per-VM Persistence Error:** ~2.35% (computed over individual VM step-to-step absolute differences).
* **Operational Meaning:** The macro-level aggregate is roughly **10x smoother** than individual VM series due to uncorrelated variance cancellation across the fleet.

---

## 4. Forecasting Approach & Validation

### 4.1 Features
All features are computed strictly at the forecast origin $t$:
* Lags 1 through 6 (`y[t-1]` through `y[t-6]`).
* Standardized feature matrix via `StandardScaler` to prevent collinear matrix ill-conditioning.
* **Explicit Exclusions:** Calendar features (no diurnal cycle in 11 hours), metadata lifetime CPU statistics (target leakage), and future timestamps.

### 4.2 Model Ladder & Cross-Validation Benchmarking
Models were evaluated across 6 rolling-origin cross-validation folds (expanding window scheme, step size 8, horizon 24, starting origin $t=50$):

| Model | MAE Mean | MAE Std | Peak MAE Mean | RMSE Mean | WAPE Mean (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ensemble** | **0.1849** | 0.0635 | 0.1658 | 0.2136 | 1.96% |
| **AR1** | 0.2006 | 0.0569 | 0.1957 | 0.2285 | 2.12% |
| **TrainMean** | 0.2078 | 0.0537 | 0.2238 | 0.2362 | 2.20% |
| **DirectRidge** | 0.2098 | 0.0764 | 0.2542 | 0.2415 | 2.22% |
| **DirectLightGBM** | 0.2113 | 0.0700 | 0.2430 | 0.2435 | 2.24% |
| **HoltDamped** | 0.2251 | 0.0939 | 0.1468 | 0.2581 | 2.38% |
| **LastValue** | 0.2288 | 0.1305 | 0.1392 | 0.2631 | 2.42% |

### Benchmarking Insights:
1. **The Mean Reversion Trap:** The constant `TrainMean` baseline (MAE 0.2078) was competitive with individual ML models (`DirectRidge`, MAE 0.2098), demonstrating strong mean-reverting dynamics.
2. **Autoregressive Power:** The first-order autoregressive model `AR1` (MAE 0.2006) captured local inertia without compounding drift errors.
3. **Winning Architecture:** The composite **Ensemble** model achieved the lowest CV MAE (0.1849) and superior stability. It was frozen prior to holdout evaluation.

---

## 5. Single-Pass Holdout Generalization
The winning model was frozen and scored once against the strictly unseen final 24 timesteps:

| Target Metric | Holdout Score |
| :--- | :---: |
| **CPU Mean Absolute Error (MAE)** | **0.1046%** |
| **CPU Root Mean Squared Error (RMSE)** | **0.1289%** |
| **CPU Weighted Absolute Percentage Error (WAPE)** | **1.13%** |
| **CPU Peak-Quartile MAE** | **0.1023%** |
| **Active VM Count MAE** | **0.0 VMs** |

![Holdout Horizon Forecast](figures/holdout_forecast_with_intervals.png)  
*Figure 3: 24-step holdout forecast vs. ground truth with 95% empirical prediction interval.*

### 5.1 Error by Horizon Step
* **Step 1 (5 minutes ahead):** MAE ≈ **0.048%**
* **Step 12 (1 hour ahead):** MAE ≈ **0.098%**
* **Step 24 (2 hours ahead):** MAE ≈ **0.142%**
* **Failure Modes:** Errors scale monotonically as the autoregressive signal reverts toward the training mean.

---

## 6. Uncertainty Quantification & Capacity Planning Policy
* **Step-to-Step Volatility (First-Difference Noise):** Standard deviation $\sigma = 0.1580\%$.
* **95% Prediction Interval Half-Width:** $\pm 1.96 \cdot \sigma = \pm 0.3097\%$.
* **Holdout Empirical Coverage:** **100.0%** of holdout actuals fell strictly within the 95% uncertainty envelope (0 SLA breaches observed, indicating $\le 5\%$ expected under-provisioning risk at the 95% confidence level).
* **Recommended Capacity Buffer:** To maintain utilization below 80.0%, the provisioning floor should follow $C(t) = \frac{\hat{y}(t) + 1.96 \cdot \sigma}{0.80}$, yielding a conservative safety buffer multiplier of **0.1211**.

---

## 7. Limitations and Future Work
* **Trace Duration:** An 11.1-hour trace is too brief to extract multi-day periodicity or diurnal seasonality.
* **Single Holdout Window:** A single 24-step holdout offers limited test diversity; benchmark claims are appropriately rooted in the 6-fold rolling CV benchmark.
* **Static Fleet:** The active VM count remained constant at 10,000; future work should benchmark auto-scaling workloads with high VM churn.

---

## Appendix A: Feature Availability Table

| Feature | Available at Origin? | Justification |
| :--- | :---: | :--- |
| `lag_1` ... `lag_6` | ✅ Yes | Observed prior to forecast origin $t$ |
| Hour / Day of Week | ❌ No | Non-informative in an 11-hour sample |
| Metadata Lifetime CPU | ❌ No | Aggregates future steps; violates temporal boundary |
| VM Deletion Timestamp | ❌ No | Future event leakage |

---

## Appendix B: AI Assistance Note
AI tools were used for code scaffolding, linting assistance, and baseline template structuring. All data transformations, temporal boundary checks, model benchmarking, and empirical calculations were executed, audited, and verified locally.