# TRACE SSL five-epoch cross-release development check

Status: **exploratory; no adapted encoder selected**. Run 12 November 2026. The RFMiD 2.0 test archive and RFMiD v1 test archive remained sealed.

## Fixed comparison

The same 1,917 RFMiD v1 training images supplied support and negatives for five encoders: frozen DINOv2-S, frozen DINOv2-L, frozen RETFound-DINOv2, and the corrected five-epoch plain-crop and full-field TRACE SSL students. The same 50 frozen support draws per label at K=3 and the same base-label-selected logistic head with C=1 were applied to all models. The queries were the same 177 RFMiD 2.0 **validation** images. No v2 image entered SSL training. Both adapted feature passes accepted all 177 queries, with 52 preprocessing warnings each.

The three mapped codes are `CRS` (chorioretinitis), `MS` (macular scar) and `RT` (retinal traction). They have only 7, 6 and 8 positive validation queries respectively. These are low-support dataset labels, not three independently verified rare diagnoses. The code mapping remains provisional; per-image camera metadata and patient IDs are unavailable.

## AUPRC of the 50-draw mean prediction

| Encoder | CRS | MS | RT |
|---|---:|---:|---:|
| Frozen DINOv2-S | 0.0427 | 0.1177 | 0.6464 |
| Frozen DINOv2-L | 0.0601 | 0.0602 | 0.5265 |
| Frozen RETFound-DINOv2 | 0.0725 | 0.2140 | 0.2738 |
| Matched plain-crop SSL, epoch 5 | 0.0469 | 0.1210 | 0.6350 |
| Full-field SSL, epoch 5 | 0.0452 | 0.1237 | 0.6463 |

The positive-prevalence baselines are 0.0395, 0.0339 and 0.0452 for CRS, MS and RT. The full-field minus matched-crop AUPRC differences were −0.0017 (CRS), +0.0027 (MS), and +0.0113 (RT). Paired image-bootstrap 95% intervals for all three differences crossed zero: [−0.0131, +0.0017], [−0.0043, +0.0258], and [−0.0397, +0.0809]. Full-field minus frozen DINOv2-S was +0.0025, +0.0060 and −0.0001; all intervals also crossed zero. The bootstrap fixes the support draws and resamples query images within label class. It cannot account for patient clustering or source-level uncertainty.

## Interpretation

The small RFMiD v1 validation advantage of SSL adaptation **did not become a clear cross-release advantage** here. The strongest model changes by finding. In particular, a general encoder winner or a camera-invariance claim would be unsupported. Do not extend SSL training or add a new loss because of these three numbers alone. Keep the frozen DINOv2-S operational baseline and retain all comparisons as research results.

The feature files and full per-episode, per-label and paired outputs are local under `data/features/` and `outputs/trace_ssl_rfmid2_k3_dev/`; the generated arrays are ignored by git. Reproduce with `scripts/extract_trace_ssl_rfmid2_validation.py --mode plain_crop_matched`, the same script with `--mode full_field`, then `scripts/evaluate_trace_ssl_rfmid2_k3.py`. The query-archive SHA-256 was `f979241706e334e1db1a56635c2d0c0e08ecdcc44a2b19cd7b614a133ea00f74`; both adapted extractions recorded the same ordered image-hash digest `19050ed3797d056452079b426154d32f4e36b0781ccf3f101fbb6d8922a5b878`.

Next decision: freeze a *single* final comparison and audit the untouched test archive for label consistency and cross-release duplication before any one-time test use. The present validation data cannot justify more model selection.
