# TRACE dataset acquisition status

Snapshot date: **2026-11-10**

The scope is colour fundus photography for TRACE-F-lite plus source data for the
evidence graph. Downloading every ophthalmic dataset indiscriminately would add
licence risk, duplicates, label conflicts and non-fundus modalities without
improving the stated experiments.

## Ready locally

| Dataset/source | State | Primary role |
|---|---|---|
| RFMiD | verified archive and manifest | multi-label development |
| IDRiD segmentation, grading and localisation | verified archives and manifest | lesion preservation and DR/DME |
| FIVES | verified and extracted | vessel preservation and QC |
| PALM | verified archive and manifest | two-camera/pathologic-myopia external evidence |
| JSIEC-1000 | verified archive | 39-class rare/multi-condition coverage and graph mapping |
| Paraguay DR V03 | verified archive | external hospital/population DR shift and trust evaluation |
| EyeQ labels | labels ready; images absent | quality gate after EyePACS access |
| Mondo/HPO/Uberon/RO/ECO/Orphadata/HGNC/ClinGen/Biolink | source snapshots ready | evidence knowledge graph |

## Access or licence blocked

| Dataset | Blocker | Needed action | Priority |
|---|---|---|---|
| EyePACS | Kaggle account, competition acceptance and API token | user accepts rules and installs a Kaggle API token | high for large retinal pretraining/QC, but not rare labels |
| APTOS 2019 | Kaggle competition acceptance and API token | same Kaggle setup | medium |
| ODIR-5K | official challenge download/account route | user joins/accepts official challenge terms | high for multi-camera, paired-eye evaluation |
| REFUGE/REFUGE2 | challenge registration and research-use terms | user joins challenge | medium for glaucoma/disc-cup and device shift |
| Messidor-2 | personal download form; redistribution prohibited | user completes official academic form | medium for external DR/site shift |
| BRSET | PhysioNet credentialing/data-use access | complete credentialing | high metadata value, but not required to proceed |
| DDR | Google/Baidu Drive multipart release, non-commercial share-alike licence | acquire only if the DR expansion is approved | low-to-medium; large and not rare-disease focused |

## Selection rule

Do not merge all datasets into one training pool. Preserve a `dataset_id`,
source split, subject/eye identity when available, camera metadata, licence and
label provenance. Deduplicate before assigning TRACE splits. External test sets
must not be used to tune preprocessing, heads, calibration or graph weights.
