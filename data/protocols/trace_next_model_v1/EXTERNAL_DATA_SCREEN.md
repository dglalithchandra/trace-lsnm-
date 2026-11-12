# External-data screen for the next TRACE model

12 November 2026. Source/metadata audit only. No new encoder scores were computed. The RFMiD v1 protected test remains gated.

| Source | Verified availability and terms | Fit for TRACE | Decision |
|---|---|---|---|
| FDR26 | Official Zenodo record 23101200 is open; its API reports CC BY 4.0. Metadata XLSX MD5 matched the publisher's `128b725a17177d38e419993c423ad4b9`; it has exactly 3,000 distinct nonblank `image_id` values and columns `image_id`, `classe`, `grade_DR`, `sexe`, `age`, `type_diabete`. | Useful candidate for an independent **DR-only** shift check or lesion/anatomy audit. It has no explicit patient or camera column in the downloaded metadata; an ID pattern is not evidence of patient identity. It cannot establish rare-disease or arbitrary-device transfer. | Keep as candidate. The 1.68 GB image download was stopped after ~6 MB because the observed rate implied about 1.5 hours. No image integrity or duplicate audit has been completed. |
| AIROGS | Official Zenodo record 5793241 reports 101,442 fundus training images from 54,274 subjects and about 500 sites; its API lists CC BY-NC-ND 4.0, and the full release is ~55 GB. | Valuable acquisition diversity, but mostly glaucoma-screening context. Terms for adapting images into a distributable trained model need review. | Do not silently admit it to the training pool. |
| OIA-DDR | Authors' repository offers Google Drive/Baidu downloads and states CC BY-NC-SA 4.0 terms for images and labels. Published research reports 13,673 images; that count has not been checked against a local archive. The GitHub MIT license applies to code, not the image data. | Useful for a data-scale and small-lesion experiment under noncommercial terms; DR-heavy and not a rare-disease evaluation cohort. | Candidate training source if download and redistribution conditions fit the challenge. Keep original split and deduplicate against all evaluation sets. |
| Local EDID/JSIEC/FIVES/PALM/Paraguay/RFMiD 2.0 | Already assigned development or external evaluation roles in TRACE. Several have sparse labels, unknown patient IDs, or unresolved schema. | Useful for diagnostics with their documented limits. | Do not casually mix evaluation images into SSL pretraining to inflate data count. |

## Next data decision

The fastest defensible path is to **finish the locked RFMiD test review first**, then establish a new protected external query set before choosing a new model. If a larger training pool is acquired, compare RFMiD-only and pooled SSL at matched updates and use source-balanced sampling. Count patient-independent positives and mimics before calling any external set a rare-disease benchmark. The data-scale hypothesis remains untested locally.

Source records and paper: https://zenodo.org/records/23101200 ; https://zenodo.org/records/5793241 ; https://github.com/nkicsl/DDR-dataset ; https://pmc.ncbi.nlm.nih.gov/articles/PMC13418815/ .
