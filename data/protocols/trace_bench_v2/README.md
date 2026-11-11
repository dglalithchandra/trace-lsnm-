# TRACE-Bench v2: RFMiD 2.0 development transfer draft

Status: **exploratory development protocol**. No protected test member was opened to make this plan. The RFMiD v1 test and RFMiD 2.0 test remain outside this analysis.

## Question

When support images come from RFMiD v1 training, how well do frozen encoder features rank the same labelled findings in RFMiD 2.0 validation images? This is a cross-release check. It is not a verified cross-camera experiment because per-image camera metadata is unavailable. RFMiD 2.0 validation contains only 177 images, so findings for individual labels will be imprecise.

## Source labels and eligibility

The generated `rfmid2_v1_label_mapping_draft.csv` records every RFMiD 2.0 training/validation CSV field, its development positive counts, and its provisional v1 mapping. The two dataset papers describe these three shared codes consistently enough to attempt a development comparison:

| Code | Finding | V2 validation positives | V2 validation negatives |
|---|---|---:|---:|
| CRS | Chorioretinitis | 7 | 170 |
| MS | Macular scar | 6 | 171 |
| RT | Retinal traction | 8 | 169 |

These are **three analysis labels**, not three independently confirmed rare diseases. MS and RT are imaging findings, and dataset frequency does not establish population rarity. The v2 paper calls exudation `EX` while its supplied CSV calls a field `EDN`. V1 defines `EDN` broadly, including several exudative appearances. Exclude `EDN` from the scored cross-release set until the code and annotation scope are clarified. Also exclude the unnamed positive column, and do not equate `ME` with `CME` or `OPDM` with `ODPM` by spelling alone. All other identical abbreviations remain candidate mappings pending definition review.

## Fixed development procedure

1. Use the existing RFMiD v1 training support selections in `../trace_fewshot_v1/episode_plan.csv` for CRS, MS, and RT at K = 1, 3, and 5. Keep the existing replicate seeds and support records unchanged. Use the same v1 training negatives and identical classifier fitting for every encoder.
2. Use all 177 RFMiD 2.0 validation images as a fixed query set for each label. Read their labels from `Validation_set.zip` only. Use images from the validation archive as queries and never as support or encoder adaptation data.
3. Screen each query against the v1 support pool using the existing detail-match and visual-review workflow. The present development screen has zero visually confirmed cross-release duplicates, but is not an exhaustive independence guarantee. Exclude any confirmed duplicate before measuring performance and record the exclusion.
4. Evaluate the *same* query images and support episodes for each frozen encoder. Report per-label AUROC and AUPRC, and prevalence-normalised AUPRC (`AUPRC / query prevalence`). Show exact positive and negative counts and uncertainty intervals. Keep calibration and threshold performance separate from ranking.
5. Average scores across repeated support draws **within each label** before any three-label summary. Repeated episodes reuse the same 177 queries and must not be treated as independent patients. Report labels separately, with a macro summary only as descriptive context.
6. Select any model, resolution, or head using development data alone. Write the frozen choice and analysis code hash before a protected-test evaluation. The test archives must be audited for cross-release duplication and label/schema consistency under a separate, documented procedure before a single final comparison.

## Limits and decision gate

This split is useful for finding obvious transfer failures. With only 6–8 positives per label, it cannot establish small performance differences, clinical utility, or device invariance. A stronger confirmatory study needs additional independent, labelled fundus images, patient identifiers where available, and verified camera/site metadata for a true device-shift analysis. If those data are unavailable, present the RFMiD 2.0 results as exploratory cross-release evidence.

Sources: [RFMiD paper](https://www.mdpi.com/2306-5729/6/2/14), [RFMiD 2.0 paper](https://www.mdpi.com/2306-5729/8/2/29), [official archive](https://zenodo.org/records/7505822).
