# Fixed frozen-feature fusion: development screen

12 November 2026. **Not part of the locked final test.** The protected RFMiD test was not accessed.

Equal-weight concatenation of unit-normalised frozen DINOv2-L and RETFound-DINOv2 features was evaluated on the 550 existing RFMiD v1 K=3 validation episodes, using the fixed C=1 logistic head. Record IDs, labels, episodes and train/query splits were paired exactly. The combination was fixed before scoring; no fusion weight was tuned.

| Encoder | Macro AUPRC | Macro AUROC |
|---|---:|---:|
| DINOv2-L | 0.2044 | 0.7826 |
| RETFound-DINOv2 | 0.1793 | 0.7867 |
| Equal frozen fusion | **0.2103** | **0.8028** |
| Matched crop SSL, epoch 5 | 0.2164 | 0.8039 |
| Full-field SSL, epoch 5 | 0.2189 | 0.8023 |

Fusion minus DINOv2-L macro AUPRC was **+0.0059**, with a 95% label-and-support bootstrap interval of **-0.0552 to +0.0500**. Fusion improved 9 of 11 labels, but RT dropped from 0.7590 to 0.5027 AUPRC. It was below both five-epoch adapted encoders on the macro endpoint.

**Decision:** do not promote fusion as a stronger TRACE model or amend the locked final test. The effect is too small and uncertain, one label is badly harmed, and the validation queries have already been used in development. Further tuning of fusion weights on this split would add selection pressure.

Reproduce with `scripts/evaluate_trace_frozen_fusion_dev.py` and `scripts/analyze_trace_frozen_fusion_dev.py`. Generated per-episode rows, per-label rows, input hashes and paired intervals are in ignored `outputs/trace_frozen_fusion_dev/`. The bootstrap holds query images fixed and cannot model patient dependence.
