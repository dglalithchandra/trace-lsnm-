# TRACE source-balance v1 development result

Status: **A1 three-seed replication complete; A2/A3 blocked pending an audited multi-source pool**. This is development evidence only. No protected RFMiD test image was opened.

## Experiment

A1 continued DINOv2-S on the 1,917-image RFMiD v1 training pool for the frozen 9,580-update budget. It used the protocol in `protocol.json`: LoRA rank 8 in the final four attention blocks, an EMA teacher, a 4,096-entry contrastive queue, batch size 4, and the fixed `plain_crop_matched` view.

Each student backbone was frozen and evaluated with the already locked logistic head (`C=1`, abundant negatives), the same support episodes, and all 639 RFMiD validation queries. A0 is the unadapted frozen DINOv2-S initialization.

## Per-seed results

| Seed | K | A0 macro AUPRC | A1 macro AUPRC | A1 macro AUROC | A1 minus A0 AUPRC | Within-seed 95% interval |
|---:|---:|---:|---:|---:|---:|---:|
| 20261111 | 3 | 0.1988 | 0.2313 | 0.8106 | +0.0325 | [+0.0088, +0.0594] |
| 20261112 | 3 | 0.1988 | 0.2397 | 0.8141 | +0.0409 | [+0.0128, +0.0767] |
| 20261113 | 3 | 0.1988 | 0.2276 | 0.8118 | +0.0288 | [+0.0052, +0.0589] |
| 20261111 | 5 | 0.2266 | 0.2725 | 0.8479 | +0.0458 | [+0.0141, +0.0810] |
| 20261112 | 5 | 0.2266 | 0.2812 | 0.8513 | +0.0545 | [+0.0184, +0.0995] |
| 20261113 | 5 | 0.2266 | 0.2685 | 0.8484 | +0.0419 | [+0.0113, +0.0791] |

All three seeds exceeded the predeclared +0.02 AUPRC screen at K=3 and K=5. Macro AUROC did not fall, no label lost more than 0.10 AUPRC, and every run opened zero protected-test images.

## Three-seed summary

| K | Mean A1 macro AUPRC | Seed SD | Mean A1 macro AUROC | Mean AUPRC delta | Delta seed SD | Positive seeds | Seeds at least +0.02 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.2329 | 0.0062 | 0.8122 | **+0.0341** | 0.0062 | 3/3 | 3/3 |
| 5 | 0.2741 | 0.0065 | 0.8492 | **+0.0474** | 0.0065 | 3/3 | 3/3 |

The predeclared replication rule—positive AUPRC delta in at least two of three seeds at both K values—passed. This is a reproducible development result under this protocol.

## Heterogeneity that must not be hidden

The aggregate gain was not universal. EDN fell in all three seeds: mean -0.0274 AUPRC at K=3 and -0.0432 at K=5. ERM was approximately flat to slightly negative. In contrast, CSR and CRVO improved strongly. Any paper or presentation must show per-label results rather than only the macro average.

## Reproducibility and compute

Each seed completed in about 1,454 seconds (24.2 minutes), drew 38,320 training examples, and reserved at most 505,413,632 CUDA bytes. Completed checkpoint SHA-256 values:

- 20261111: `fead3b6163e1ee29a3f5091a899270e265c24c42c52dc679fc11a941318f7aec`
- 20261112: `2504939a66ebbdaaf907f0cf559af1bc36b3528ef68ed38b60d65b23f8a4f82b`
- 20261113: `3d6a69c34b20c30553d5f083e6a8db233e1d9887ddf7dd6322f586e4580592e7`

Generated checkpoints, feature files, and episode-level outputs are deliberately Git-ignored. The tracked scripts regenerate them.

## Interpretation boundary

The result supports one narrow claim: on RFMiD development data, continued label-hidden retinal SSL of DINOv2-S improved fixed few-shot ranking over the same frozen initialization across three seeds.

It does **not** yet establish improvement on fresh data, cross-camera or cross-site generalisation, calibration, clinical safety, or patient-independent performance. The within-seed intervals resample labels and support episodes while holding the same validation queries fixed; they are not independent clinical confidence intervals. Patient identifiers are unavailable. Three seeds are too few for a reliable seed-level confidence interval.

A2/A3 are still blocked. IDRiD and FIVES already serve development/preservation roles and cannot be silently moved into SSL training, while EyePACS and APTOS are absent. A legal, record-audited, query-excluded multi-source pool is required before testing source-balanced sampling.
