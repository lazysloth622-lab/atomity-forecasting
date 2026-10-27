# Model Selection Protocol

## Primary Metric

- **Mean Absolute Error (MAE)** averaged across all 6 rolling-origin
  cross-validation folds on the development window (first 110 timesteps).

## Tie-Breaking / Secondary Metric

- **Peak-Quartile MAE (top 25% CPU values).** In capacity planning,
  underestimating peak load is costlier than average error. If two models
  differ by less than 0.01 MAE, the one with lower Peak MAE is selected.

## Decision Rule

1. Compute mean CV MAE across the 6 folds.
2. Select the candidate with the minimum CV MAE, provided its Peak MAE does
   not severely degrade relative to the LastValue baseline.
3. Freeze the model identity and architecture (recorded in
   `reports/numbers.json` and git tag `model-frozen`) **before** any holdout
   evaluation. The holdout is scored exactly once.
