# EDID exploratory transfer results

Status: **exploratory external folder-label evaluation, not confirmatory clinical or camera evaluation**. Completed 11 November 2026 without opening protected RFMiD test images.

## Fixed setup

- Official, unaugmented EDID archive was checksum verified. Exact and reviewed near copies were removed before scoring. The frozen query manifest is `data/manifests/edid_exploratory_manifest_v2.csv` and the review ledger is `data/manifests/edid_near_duplicate_visual_review.csv`.
- The preprocessing audit accepted all 1,333 target/healthy images: 326 macular scar (MS), 124 retinitis pigmentosa (RP), 106 disc edema (ODE), and 777 healthy controls. Two MS images carried warnings and remained included.
- DINOv2-S, DINOv2-L and RETFound-DINOv2 each produced finite features for the identical 1,333 record IDs in identical order. The encoder weights were frozen. Geometry preprocessing was active; photometric harmonisation was disabled.
- Support came only from RFMiD v1 training. MS reused 50 fixed subsets per K=1/3/5; RP enumerated all six K=1 and 20 K=3 positive subsets; ODE reused 50 fixed base-label subsets per K=1/3/5. Each encoder used its own classifier head and hyperparameter choices previously selected on RFMiD v1 **base labels**, with all available training negatives for the target label. No EDID query label selected a head or support subset.
- The task for each label was its EDID folder versus the EDID Healthy folder. Other disease folders were not treated as negatives.

## Paired-image ranking results

Mean scores across the fixed support subsets were ranked against the same EDID queries. AUPRC depends on prevalence; its random ranking baselines are MS 0.296, RP 0.138 and ODE 0.120. The intervals in the saved CSVs are stratified image bootstraps conditional on the fixed support ensemble and do **not** account for unknown patient clustering or source uncertainty.

| Label | K | DINOv2-S AUPRC / AUROC | DINOv2-L AUPRC / AUROC | RETFound AUPRC / AUROC |
|---|---:|---:|---:|---:|
| MS | 1 | 0.337 / 0.537 | 0.413 / 0.586 | **0.668 / 0.795** |
| MS | 3 | 0.421 / 0.613 | **0.515 / 0.664** | 0.353 / 0.566 |
| MS | 5 | 0.479 / 0.643 | **0.547 / 0.685** | 0.375 / 0.565 |
| RP | 1 | 0.885 / 0.946 | 0.916 / 0.973 | **0.941 / 0.990** |
| RP | 3 | 0.888 / 0.943 | 0.916 / 0.971 | **0.936 / 0.984** |
| ODE | 1 | **0.632 / 0.876** | 0.594 / 0.879 | 0.592 / 0.856 |
| ODE | 3 | **0.642 / 0.884** | 0.579 / 0.869 | 0.548 / 0.822 |
| ODE | 5 | **0.661 / 0.894** | 0.617 / 0.887 | 0.614 / 0.854 |

Bold marks the highest AUPRC in each row, not a statistically established winner. Full per-episode values, per-label intervals, and file hashes are in `outputs/edid_exploratory_transfer_v1/<encoder>/` on this machine. Feature NPZ files are in `data/features/` and are intentionally excluded from git.

## Interpretation and limits

RP versus healthy ranks well for all three models, but this is an easier comparison than RP versus plausible mimics. Six RFMiD RP support positives underlie every RP subset, so the 26 subsets are correlated and do not represent 26 independent support cohorts. MS transfer is weaker and varies substantially by encoder and K. RETFound's MS K=1 result uses a previously selected **KNN** head; its K=3/5 results use **logistic** heads. The apparent decrease with more shots cannot be attributed to encoder quality or extra examples alone. ODE is a base-label sanity check, not a novel-label result.

No single encoder wins across all labels and K values. This experiment does not select a new TRACE default encoder. EDID folder labels were not clinically adjudicated. Patient IDs and per-image camera IDs are unavailable. Near-duplicate screening proposed priority candidates and was not exhaustive. The healthy folder could contain unlabelled disease. Consequently, these numbers do not establish patient-independent transfer, camera generalisation, rare-disease diagnosis, or clinical screening performance.

Next useful work: perform a same-head sensitivity comparison for the RETFound MS anomaly, then assess whether that finding survives a harder dataset with trustworthy disease/mimic labels. Do not tune the encoder or the head on EDID and then reuse EDID as a confirmatory test.
