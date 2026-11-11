# RFMiD 2.0 exploratory transfer result

11 November 2026. **Development analysis only.** The protected RFMiD v1 and RFMiD 2.0 test archives were not opened for this run.

## Setup

- RFMiD v1 training images supplied the 50 fixed support draws per label and K. K was 1, 3, or 5. The abundant-negative protocol and each encoder's previously base-label-selected classifier head were used without retuning on RFMiD 2.0.
- All 177 RFMiD 2.0 validation images were fixed queries. Frozen DINOv2-S, DINOv2-L, and RETFound-DINOv2 features were extracted with the same 448 px TRACE geometry. Photometric harmonisation was disabled. All 177 images passed preprocessing for all three encoders.
- The same query IDs, labels, and order were verified across the three feature files. The query labels had 7 CRS, 6 MS, and 8 RT positives. Fifty support-draw scores were averaged per query before ranking metrics were computed.
- The 95% intervals below use a stratified query bootstrap conditional on those averaged support scores. They omit patient, site, source-label, and support-sampling uncertainty. The small positive counts make them unstable.

## Five-shot results

| Encoder | Label | Positives | AUPRC | 95% query-bootstrap interval | AUROC |
|---|---|---:|---:|---:|---:|
| DINOv2-S | CRS | 7 | 0.064 | 0.046–0.161 | 0.624 |
| DINOv2-L | CRS | 7 | 0.110 | 0.062–0.265 | 0.731 |
| RETFound-DINOv2 | CRS | 7 | 0.088 | 0.050–0.259 | 0.671 |
| DINOv2-S | MS | 6 | 0.123 | 0.030–0.416 | 0.536 |
| DINOv2-L | MS | 6 | 0.064 | 0.037–0.160 | 0.615 |
| RETFound-DINOv2 | MS | 6 | 0.213 | 0.031–0.547 | 0.552 |
| DINOv2-S | RT | 8 | 0.680 | 0.423–0.943 | 0.949 |
| DINOv2-L | RT | 8 | 0.544 | 0.383–0.975 | 0.981 |
| RETFound-DINOv2 | RT | 8 | 0.273 | 0.097–0.578 | 0.811 |

The prevalence baselines for AUPRC are 0.040 for CRS, 0.034 for MS, and 0.045 for RT. The three-label macro AUPRC at K=5 is 0.289 for DINOv2-S, 0.239 for DINOv2-L, and 0.191 for RETFound-DINOv2. The macro value is dominated by RT. Full K=1/3/5 scores and all intervals are in `outputs/rfmid2_development_transfer_v1/<encoder>/per_label_results.csv`.

## Interpretation

The three models transfer unevenly across labels. RT ranking is promising on this split, especially for the generic DINOv2 encoders. CRS and MS remain weak or unstable. DINOv2-S has the highest RT AUPRC, while DINOv2-L has the highest RT AUROC. No model can be declared generally superior from 6–8 positives per label. This result also does not establish a retinal-SSL advantage or disadvantage.

This is a **cross-release** comparison. Camera identity is not available per image, and there are no patient IDs. A development duplicate screen found zero visually confirmed v1/v2 duplicate images among reviewed candidates, but did not prove full patient independence. The two releases may share acquisition context. `MS` and `RT` are findings rather than separate rare-disease diagnoses. `EDN` remains excluded because the v2 paper uses `EX` while the CSV uses `EDN`.

## Next decision

Keep all three encoder candidates. Do not tune an adapter or choose a final model using this small v2 validation set. The next material improvement is a larger independent labelled query cohort with patient IDs and verified device/site information. Until then, a confirmatory camera-generalisation claim is unsupported. A later locked test run requires a separate duplicate and label audit first.

Reproduction: `scripts/extract_rfmid2_validation_features.py` followed by `scripts/evaluate_rfmid2_development_transfer.py` for each encoder. Feature and score files are local generated artifacts; their source hashes are recorded in the per-encoder summary JSON files under `outputs/rfmid2_development_transfer_v1/`.
