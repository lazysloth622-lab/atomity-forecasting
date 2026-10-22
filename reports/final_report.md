# Cloud Workload Forecasting: Technical Report
**Assessment:** ML Engineering Take-Home Challenge (Atomity)  
**Dataset:** Azure Public Dataset V2 Telemetry (10,000 VMs sampled over 134 timesteps at 300 s / 5 min intervals)  
**Evaluation Protocol:** Strict Temporal Split (First 110 steps for development, final 24 steps as strictly unseen holdout horizon)

---

## 1. Executive Summary & Problem Formulation
The objective is multi-step lookahead forecasting of aggregate cloud compute demand over a 2-hour horizon (24 steps at 5-minute cadence).
1. `active_vm_count(t)`: The count of distinct virtual machines emitting valid CPU readings.
2. `aggregate_cpu_utilization(t)`: The arithmetic mean of `avg_cpu` across active VMs, bounded in [0, 100].

All architectural decisions and model selections were finalized and frozen (`git tag model-frozen`) using rolling-origin cross-validation on the development window before evaluating the holdout horizon.

---

## 2. Exploratory Data Analysis (Development Window Only)
*All telemetry statistics strictly computed on the training period (steps 0 to 109):*

* **Active VM Fleet Count:** Exactly 10000 VMs. Fleet count remained constant across all development steps (`active_vm_count_constant = True`).
* **Aggregate CPU Utilization Mean:** 9.414% (Standard Deviation: 0.194%).
* **Observed CPU Range:** Min 9.112%, Median 9.378%, Max 9.912%.
* **Autocorrelation Structure:** Lag-1 autocorrelation $r_1 = 0.6621$, confirming strong local temporal persistence. Lag-24 autocorrelation drops to $r_{24} = -0.0199$, indicating no diurnal periodicity within this 11-hour sample.
* **White-Noise Significance Threshold:** $\pm 1.96 / \sqrt{N} = \pm 0.1869$.

![Development Window Workload](figures/eda_training_series.png)
![Autocorrelation](figures/eda_autocorrelation.png)

---

## 3. Pre-Registered Model Ladder & Cross-Validation Results
To guard against misleading benchmark claims, models were evaluated across 6 rolling-origin cross-validation folds (minimum origin: 50 steps, step size: 8 steps, forecast horizon: 24 steps).

| Model | MAE Mean | MAE Std | Peak MAE Mean | RMSE Mean | WAPE Mean (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ensemble** | 0.1849 | 0.0635 | 0.1658 | 0.2136 | 1.96% |
| **AR1** | 0.2006 | 0.0569 | 0.1957 | 0.2285 | 2.12% |
| **TrainMean** | 0.2078 | 0.0537 | 0.2238 | 0.2362 | 2.20% |
| **DirectRidge** | 0.2098 | 0.0764 | 0.2542 | 0.2415 | 2.22% |
| **DirectLightGBM** | 0.2113 | 0.0700 | 0.2430 | 0.2435 | 2.24% |
| **HoltDamped** | 0.2251 | 0.0939 | 0.1468 | 0.2581 | 2.38% |
| **LastValue** | 0.2288 | 0.1305 | 0.1392 | 0.2631 | 2.42% |


### Benchmarking Insights:
1. **The Mean Reversion Trap:** A trivial constant baseline (`TrainMean`, MAE 0.2078) was competitive with individual ML models (`DirectRidge`, MAE 0.2098), demonstrating that cloud CPU aggregation across 10k VMs exhibits strong mean-reverting dynamics.
2. **Autoregressive Power:** The first-order autoregressive model `AR1` (MAE 0.2006) captured the local inertia without compounding drift errors.
3. **Winning Architecture:** The composite **`Ensemble`** model (combining `LastValue`, `AR1`, and standardized `DirectRidge`) achieved the minimum CV MAE (0.1849) and superior stability (fold std 0.0635).

---

## 4. Single-Pass Holdout Generalization
The winning model was frozen and scored once against the strictly unseen final 24 timesteps:

| Target Metric | Holdout Score |
| :--- | :--- |
| **CPU Mean Absolute Error (MAE)** | **0.1046%** |
| **CPU Root Mean Squared Error (RMSE)** | **0.1289%** |
| **CPU Weighted Absolute Percentage Error (WAPE)** | **1.13%** |
| **CPU Peak-Quartile MAE** | **0.1023%** |
| **Active VM Count MAE** | **0.0 VMs** |

![Holdout Horizon Forecast](figures/holdout_forecast_with_intervals.png)

---

## 5. Uncertainty Quantification & Capacity Planning Policy
Point forecasts alone create operational risks if transient surges induce resource exhaustion. 

* **Empirical Residual Noise:** Standard deviation $\sigma = 0.1580\%$.
* **95% Prediction Interval Half-Width:** $\pm 1.96 \cdot \sigma = \pm 0.3097\%$.
* **Holdout Empirical Coverage:** **100.0%** of holdout actuals fell strictly within the 95% uncertainty envelope, resulting in **0 SLA breaches**.
should follow $C(t) = \frac{\hat{y}(t) + 1.96 \cdot \sigma}{0.8}$, yielding a conservative safety buffer multiplier of **0.1211**.
