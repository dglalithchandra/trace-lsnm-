# TRACE source-balance v1 development result

Status: **replication in progress**. This file records development evidence only. No protected RFMiD test image was opened.

## A1 seed 20261111

The A1 arm continued DINOv2-S on the 1,917-image RFMiD v1 training pool for the frozen 9,580-update budget. It used the protocol in `protocol.json`: LoRA rank 8 in the final four attention blocks, an EMA teacher, a 4,096-entry contrastive queue, batch size 4, and the fixed `plain_crop_matched` view.

Training completed in 1,453.25 seconds (24.2 minutes), drew 38,320 training examples, reserved at most 505,413,632 CUDA bytes, and opened zero protected-test images. Completed checkpoint SHA-256:

`fead3b6163e1ee29a3f5091a899270e265c24c42c52dc679fc11a941318f7aec`

The student backbone was frozen and evaluated with the already locked logistic head (`C=1`, abundant negatives), the fixed support episodes, and all 639 RFMiD validation queries.

| Model | K | Macro AUPRC | Macro AUROC | AP / prevalence |
|---|---:|---:|---:|---:|
| Frozen DINOv2-S (A0) | 3 | 0.1988 | 0.7946 | 16.70 |
| RFMiD SSL (A1) | 3 | 0.2313 | 0.8106 | 19.01 |
| Frozen DINOv2-S (A0) | 5 | 0.2266 | 0.8279 | 18.91 |
| RFMiD SSL (A1) | 5 | 0.2725 | 0.8479 | 22.19 |

Paired A1 minus A0 macro-AUPRC differences:

- K=3: **+0.0325**, 95% hierarchical interval **[+0.0088, +0.0594]**.
- K=5: **+0.0458**, 95% hierarchical interval **[+0.0141, +0.0810]**.

The interval resamples labels and support episodes, but holds the reused validation queries fixed. It therefore does not solve the limited-independent-query problem.

All predeclared screen gates passed: both AUPRC gains exceeded +0.02, macro AUROC did not fall, and no label's mean AUPRC fell by more than 0.10. The worst label change was EDN at K=5 (-0.0431 AUPRC). That heterogeneity must remain visible; the aggregate improvement does not mean every condition improved.

## Interpretation boundary

This is encouraging one-seed development evidence, not a final result. It does not establish cross-camera generalisation, clinical safety, patient-independent performance, or performance on unseen data. The next allowed action is the already planned seed replication. A2/A3 remain blocked until a legal, leakage-audited multi-source SSL pool exists.
