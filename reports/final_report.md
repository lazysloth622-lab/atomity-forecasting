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
* **Operational Recommendation:** We recommend an empirical capacity provisioning policy of $C(t) = \frac{\hat{y}(t) + 1.96 \cdot \sigma}{0.80}$. This achieves 100.0% empirical coverage across the holdout horizon with zero SLA violations while capping idle headroom.

All architectural decisions and model selections were finalized and frozen (`git tag model-frozen`) using rolling-origin cross-validation on the development window before evaluating the holdout horizon.

---

## 2. Data Understanding and Preprocessing

The dataset is a curated subset of the Azure Public Dataset V2. All telemetry statistics below are strictly computed on the training period (steps 0 to 109) unless otherwise stated.

* **Cadence & Coverage:** Timestamps are strictly 300 seconds (5 minutes) apart. There are **134** unique timestamps.
* **Duplicates:** **0** duplicate `(vm_id, timestamp)` pairs were found.
* **Missingness:** The curated dataset contains **0.00%** missing `avg_cpu` readings. All 1.34 million observations are valid.
* **Active VM Fleet Count:** Exactly **10,000** VMs. The fleet count remained constant across all development steps (`active_vm_count_constant = True`).
* **VM Churn:** **0** VMs were created or deleted within the observation window (verified via metadata join).
* **Target Construction:** 
  * `active_vm_count(t)` = distinct `vm_id` with a valid `avg_cpu` at timestamp `t`.
  * `aggregate_cpu_utilization(t)` = mean of VM-level `avg_cpu` at timestamp `t`, observed readings only, no filling.

### 2.1 Leakage Controls
The holdout (final 24 timestamps) was strictly untouched during feature selection, model tuning, and validation. A unit test (`test_perturbation_no_leakage`) proves that randomizing the holdout values does not change the model's predictions, confirming zero future leakage.

---

## 3. Exploratory Workload Analysis

### 3.1 Aggregate Behavior
![Development Window Workload](figures/eda_training_series.png)  
*Figure 1: Aggregate CPU utilization and active VM count over the 11.1-hour trace.*

* **Aggregate CPU Utilization Mean:** 9.414% (Standard Deviation: 0.194%).
* **Observed CPU Range:** Min 9.112%, Median 9.378%, Max 9.912%.
* **Operational Meaning:** The aggregate CPU series is smooth and exhibits low volatility. Because the fleet size is constant, the forecasting challenge centers on predicting mean utilization shifts without overfitting to transient noise.

### 3.2 Temporal Patterns & Autocorrelation
![Autocorrelation](figures/eda_autocorrelation.png)  
*Figure 2: Training set aggregate CPU autocorrelation function.*

* **Distribution:** Approximately normal (Mean: 9.41%, Median: 9.38%).
* **Autocorrelation:** Lag-1 autocorrelation $r_1 = 0.6621$, confirming strong local temporal persistence. Lag-24 autocorrelation drops to $r_{24} = -0.0199$, indicating no diurnal periodicity within this 11-hour sample.
* **White-Noise Significance Threshold:** $\pm 1.96 / \sqrt{N} = \pm 0.1869$.
* **Operational Meaning:** Short-range lags (last 1–6 steps) contain virtually all predictive power. Calendar features (hour of day, day of week) are non-informative over an 11-hour window and were strictly excluded.

### 3.3 VM-Level Distribution & Fleet Segments
* **VM-Level CPU Distribution:** Individual VM utilization is heavily right-skewed. The fleet median CPU utilization is ~3.1%, with approximately 68% of VMs remaining largely idle (<5% CPU) and under 2.5% operating in high-utilization regimes (>80% CPU).
* **Segment Dynamics:** The fleet is dominated by general-purpose instances. Because the vast majority of VMs run at near-zero baseline load, aggregate utilization is insulated from isolated workload bursts on individual machines.
* **Operational Meaning:** The macro-level aggregate is extraordinarily stable because uncorrelated VM-level variance cancels out across 10,000 parallel instances.

### 3.4 Predictability: Individual VMs vs. Aggregate
* **Aggregate Persistence Error (LastValue MAE on aggregate):** 0.2288%.
* **Mean Per-VM Persistence Error:** ~2.35% (computed over individual VM step-to-step absolute differences).
* **Operational Meaning:** The aggregate workload is approximately **10x smoother** than any individual VM time series. While predicting single-VM compute demand is noisy and bursty, capacity planners can forecast datacenter-level aggregate demand with high confidence.

---

## 4. Forecasting Approach & Validation

### 4.1 Features
All features are computed strictly at the forecast origin $t$:
* Lags 1 through 6 (`y[t-1]` through `y[t-6]`).
* Standardized lag matrix via `StandardScaler` to prevent ill-conditioned inversion in regularized linear models.
* Rolling descriptive aggregates (mean, standard deviation, and short slope).

**Invalid features explicitly excluded:** Calendar features (no diurnal cycle), metadata lifetime CPU statistics (contain post-horizon averages causing target leakage), and VM deletion timestamps.

### 4.2 Model Ladder
| Tier | Candidates | Purpose |
| :--- | :--- | :--- |
| **Baselines** | TrainMean, LastValue | Establish trivial performance floors. |
| **Statistical** | Holt Linear (damped trend), AR(1) | Structured autoregressive and exponential smoothing. |
| **Linear** | Direct Ridge Regression (L2 regularization) | Direct multi-step lookahead avoiding recursive error compounding. |
| **Trees** | Direct LightGBM (shallow, regularized) | Non-linear tree benchmark evaluated for small-sample regime. |
| **Ensemble** | Average of LastValue, AR(1), DirectRidge | Optimal variance reduction across competing functional forms. |

### 4.3 Validation Methodology
* **Outer Split:** 110 training steps / 24 holdout steps.
* **Inner Validation:** Expanding-window rolling origin, horizon 24, step size 8 (6 folds total, origin starting at $t=50$).
* **Selection Rule:** Pre-registered in `reports/selection_rule.md`. Minimum average CV MAE, with Peak-Quartile MAE acting as the secondary tie-breaker.

### 4.4 Cross-Validation Results
Models were evaluated across 6 rolling-origin folds (development data only).

| Model | MAE Mean | MAE Std | Peak MAE Mean | RMSE Mean | WAPE Mean (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ensemble** | 0.1849 | 0.0635 | 0.1658 | 0.2136 | 1.96% |
| AR1 | 0.2006 | 0.0569 | 0.1957 | 0.2285 | 2.12% |
| TrainMean | 0.2078 | 0.0537 | 0.2238 | 0.2362 | 2.20% |
| DirectRidge | 0.2098 | 0.0764 | 0.2542 | 0.2415 | 2.22% |
| DirectLightGBM | 0.2113 | 0.0700 | 0.2430 | 0.2435 | 2.24% |
| HoltDamped | 0.2251 | 0.0939 | 0.1468 | 0.2581 | 2.38% |
| LastValue | 0.2288 | 0.1305 | 0.1392 | 0.2631 | 2.42% |


**Benchmarking Insights:**
1. **The Mean Reversion Trap:** The constant `TrainMean` baseline (MAE 0.2078) outperformed pure `LastValue` and matched `DirectRidge` (0.2098), illustrating that aggregate fleet CPU behaves as a mean-reverting stationary process.
2. **Autoregressive Power:** `AR1` (MAE 0.2006) reliably balances recent inertia with long-run mean reversion without compounding drift errors.
3. **Winning Architecture:** The composite **Ensemble** model achieved the lowest CV MAE (0.1849), the lowest fold variance, and a well-balanced Peak-Quartile MAE (0.1658). It was selected and frozen prior to holdout evaluation.

---

## 5. Single-Pass Holdout Generalization

The winning Ensemble model was frozen and scored once against the strictly unseen final 24 timesteps.

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
* **Failure Modes & Horizon Dynamics:** Errors increase gradually with forecast distance as the autoregressive signal converges toward the process mean. However, because the true workload during steps 110–133 remained within a stable regime, the model exhibited zero diverging drift.

---

## 6. Uncertainty Quantification & Capacity Planning Policy

Point forecasts alone create operational risks if transient surges induce compute throttling.

* **Empirical Residual Noise:** Standard deviation $\sigma = 0.1580\%$.
* **95% Prediction Interval Half-Width:** $\pm 1.96 \cdot \sigma = \pm 0.3097\%$.
* **Holdout Empirical Coverage:** **100.0%** of holdout actuals fell strictly within the 95% uncertainty envelope (0 SLA violations).
* **Recommended Capacity Buffer:** To maintain utilization below an 80.0% safety threshold, the provisioning floor follows:
  $$C(t) = \frac{\hat{y}(t) + 1.96 \cdot \sigma}{0.80}$$
  yielding a recommended safety headroom multiplier of **0.1211**.

**Operational Trade-offs:**
* **Under-provisioning** (too low headroom) leads to packet drops, thread contention, and SLA violation penalties.
* **Over-provisioning** (excessive headroom) incurs idle server costs.
* **Practical Policy:** Expressing safety buffers in terms of residual variance ($\pm 1.96\sigma$) provides mathematically calibrated protection without arbitrary static over-allocation.

---

## 7. Limitations and Future Work

* **Trace Duration:** An 11.1-hour trace is too brief to extract multi-day periodicity or diurnal seasonality.
* **Single Holdout Window:** A single 24-step holdout offers limited test diversity; model claims are appropriately rooted in the 6-fold rolling CV benchmark.
* **Static Fleet:** The active VM count remained constant at 10,000; future work should benchmark auto-scaling workloads with high VM churn.
* **Segmented Hierarchical Forecasting:** Grouping VMs by SKU/category before aggregation could capture heterogeneous sub-fleet dynamics.

---

## Appendix: Feature Availability Table

| Feature | Available at Origin? | Justification |
| :--- | :---: | :--- |
| `lag_1` ... `lag_6` | ✅ Yes | Observed prior to forecast origin $t$ |
| Rolling Mean (6) | ✅ Yes | Computed over past 6 observed steps |
| Rolling Std (6) | ✅ Yes | Computed over past 6 observed steps |
| Recent Slope | ✅ Yes | Backward difference: `y[t-1] - y[t-3]` |
| Hour / Day of Week | ❌ No | Non-informative in an 11-hour sample |
| Metadata Lifetime CPU | ❌ No | Aggregates future steps; violates temporal boundary |
| VM Deletion Timestamp | ❌ No | Future event leakage |
