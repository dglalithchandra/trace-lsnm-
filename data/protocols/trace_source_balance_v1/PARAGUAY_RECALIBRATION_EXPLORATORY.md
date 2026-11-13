# Paraguay baseline recalibration: exploratory result

The earlier frozen DINOv2-S RFMiD DR head was evaluated on a deterministic split of the previously studied Paraguay cohort. These are **not A1 scores** and the cohort is not fresh confirmation. A SHA-256 assignment of known exact-duplicate group, or record ID when no group is known, assigned 588 images to the calibration pool and 849 to evaluation. Patient identifiers are unavailable; patient separation is not established.

Each budget draws distinct calibration units without replacement, repeating 100 times. Budget zero uses the original RFMiD threshold 0.989772. Other budgets fit a balanced-accuracy threshold and Platt calibrator on the sampled calibration records. Evaluation labels are not used to fit. Values below are medians over feasible draws, on the same fixed evaluation images; they are not confidence intervals.

| Local labels | Feasible draws | Sensitivity | Specificity | Brier | ECE |
|---:|---:|---:|---:|---:|---:|
| 0 | 100/100 | 0.3318 | 0.9928 | 0.2199 | 0.2360 |
| 5 | 60/100 | 0.7807 | 0.9222 | 0.1432 | 0.0998 |
| 10 | 95/100 | 0.7912 | 0.9043 | 0.1467 | 0.1022 |
| 20 | 100/100 | 0.7796 | 0.9199 | 0.1457 | 0.1092 |
| 50 | 100/100 | 0.7923 | 0.9043 | 0.1393 | 0.0730 |

Small label samples often fail the requirement for at least two examples per class or a positive Platt slope. Excluding these failed samples biases a successful-fit summary; 5 labels should not be presented as reliably sufficient. Sensitivity gains trade off against specificity. Repeated samples reuse the same calibration pool and evaluation cohort and do not represent independent sites or patients. No causal, clinical, universal-camera, fresh-test, or A1 claim follows.

The ignored output `outputs/paraguay_recalibration_exploratory_v1/result.json` includes all 500 draws, infeasibility reasons, ranking and risk-coverage metrics, counts and input hashes. Reproduce after moving aside any existing output with `scripts/run_paraguay_recalibration_exploratory.py`. No protected test records were accessed.
