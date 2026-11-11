# RFMiD 2.0 development audit

11 November 2026. This audit reads the RFMiD v1 assigned train and validation images, and the RFMiD 2.0 training and validation archives. Neither protected test archive was decoded.

## Label schema

- RFMiD 2.0 has 509 training and 177 validation JPGs, with matching label rows.
- The CSV header contains 51 named condition columns plus `WNL`, `ID`, and one unnamed final column. The `ON` header has a trailing nonbreaking space. The unnamed column contains 10 training and 4 validation values of `1`. Its meaning is unresolved.
- Thirty-eight named condition abbreviations occur in both v1 and v2. This is a **code-level candidate mapping**. It is not a verified clinical-equivalence mapping.
- V2-only codes after trimming whitespace are `CNV, CSC, GRT, HTN, IIH, ME, ON, OPDM, RD, RHL, RTR, SOFE`. V1-only codes are `BRAO, MNF, ODPM, PLQ, PT, PTCR, VH`. `OPDM` and `ODPM` look similar, but must remain separate until the source authors clarify the definition. The v2 paper inconsistently uses `SOFE` and `SOFT` in prose and captions.
- The v2 paper lists macular edema (`ME`) among its conditions. The CSV includes both `ME` and `CME`, and the unnamed column is positive. Do not assign the unnamed values to either disease without evidence.

The eleven v1 few-shot novel labels have these v2 development counts:

| Label | V2 train positives | V2 validation positives |
|---|---:|---:|
| ERM | 3 | 1 |
| LS | 12 | 2 |
| MS | 18 | 6 |
| CSR | 10 | 3 |
| CRVO | 7 | 3 |
| AION | 2 | 1 |
| RT | 24 | 8 |
| RS | 5 | 2 |
| CRS | 25 | 7 |
| EDN | 36 | 16 |
| RPEC | 2 | 1 |

The v2 validation split alone supports only a narrow subset of useful novel-label measurements. Total test positives remain unknown because the test labels are sealed.

## Cross-release image overlap

The first screen compared 2,556 v1 assigned development images with 686 v2 development images using 256-bit difference hashes and grayscale thumbnails. It yielded 3,167 coarse candidate pairs and 642 thresholded `strong_candidate` rows. Fundus circles produce many false positives under those coarse rules, so the second stage used SIFT local keypoints and RANSAC homography.

The detail stage examined 341 higher-priority pairs. Four pairs involving three distinct v2 validation JPGs had at least 20 geometrically consistent matches and at least 35% inliers. The candidates are `810.jpg`, `828.jpg`, and `847.jpg`, all labelled `WNL`. They match v1 assigned development images. `810.jpg` matched two v1 records. These are **probable shared retina images**, pending manual visual confirmation. Quarantine them from any v1-to-v2 evaluation. The search was a candidate screen, not an exhaustive proof that all remaining images are independent. The v1 and v2 protected test sets have not been screened.

## Decision for the benchmark

RFMiD 2.0 remains useful as an external cohort after source definition mapping, visual confirmation and duplicate quarantine. Its small novel-label validation counts prevent treating this split as the sole confirmatory rare-disease benchmark. Per-image camera assignment is still unverified, so device-specific claims need additional metadata.

Sources: [RFMiD v1 paper](https://www.mdpi.com/2306-5729/6/2/14), [RFMiD 2.0 paper](https://www.mdpi.com/2306-5729/8/2/29), [official v2 archive](https://zenodo.org/records/7505822).

Artifacts: `scripts/audit_rfmid2_development_labels.py`, `scripts/screen_rfmid2_development_overlap.py`, `scripts/confirm_rfmid2_development_overlap.py`, `data/manifests/rfmid2_development_overlap_screen.csv`, `data/manifests/rfmid2_development_detail_matches.csv`.
