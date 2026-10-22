\# Model Selection Protocol



\## Primary Metric

\- \*\*Mean Absolute Error (MAE)\*\* averaged across all 6 rolling-origin cross-validation folds on the training window (first 110 timesteps).



\## Tie-Breaking / Secondary Metric

\- \*\*Peak-Quartile MAE (top 25% CPU values)\*\*: In capacity planning, underestimating or misforecasting peak load is costlier than average error. If two models are within 0.01 MAE, the one with lower Peak MAE is selected.



\## Decision Rule

1\. Compute mean CV MAE across the 6 folds.

2\. Select the candidate model with the minimum CV MAE, provided its Peak MAE does not severely degrade relative to the baseline LastValue.

3\. Once selected, freeze the model identity and architecture in `reports/numbers.json` under `selected\_model` prior to any holdout evaluation.

