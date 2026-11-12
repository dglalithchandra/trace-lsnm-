# TRACE verified project report

Report date: 12 November 2026  
Scope: work completed in this repository as of commit preparation on 12 November 2026  
Research status: development-stage research system; no clinical deployment claim

## 1. Executive summary

TRACE investigates whether retinal representation learning can improve few-shot recognition of low-support retinal diseases and findings under acquisition and source shift. The repository now contains a frozen preprocessing/camera module, audited manifests, frozen foundation-model baselines, a fixed few-shot protocol, label-hidden retinal SSL students, external exploratory evaluations, trust and retrieval analyses, and a gated final-test workflow.

The main conclusions supported by completed work are:

- Geometry standardisation and quality reporting are ready as a research preprocessing module.
- Six-characteristic photometric harmonisation did not pass its deployment gate and is disabled.
- Acquisition/source information remains strongly visible in frozen encoder embeddings, but available metadata do not support a universal camera-generalisation claim.
- Frozen DINOv2-S remains the operational baseline because larger and retinal-specific encoders did not establish a consistent same-protocol advantage on the current development episodes.
- Label-hidden retinal SSL improved with longer training on the reused development split, but superiority over the strongest frozen comparator has not been established.
- Ranking can transfer externally while thresholds, calibration and conformal coverage fail under shift.
- The protected final comparison is prepared but has not run. Human image-identity decisions are still required, and no final-test encoder features, predictions or performance results exist.
- The evidence knowledge graph has authoritative source snapshots and a complete build/evaluation specification, but no curated medical assertion graph or graph-reranking result exists yet.

This is a credible research foundation. It is not yet a finished winning model result.

## 2. Repository contents

The repository contains:

- `trace_preprocessing/`: manifest-driven fundus preprocessing, geometry, photometric measurements, bounded experimental correction, quality measurements and preservation checks.
- `trace_camera/`: the frozen camera-module interface and configuration.
- `trace_models/`: frozen encoder and disease-scoring interfaces.
- `trace_protocols/`: few-shot protocol and evaluation support.
- `trace_trust/`: calibration, OOD and selective-prediction metrics.
- `scripts/`: acquisition, audit, training, extraction, evaluation, bootstrap, retrieval and release scripts.
- `tests/`: automated unit and protocol tests.
- `data/manifests/`: versioned metadata, hashes, split/group fields and audit ledgers without retinal pixel data.
- `data/protocols/`: frozen plans, episode schedules and tracked result summaries.
- `data/knowledge_sources/`: checksummed ontology/source inventory; raw source snapshots remain locally ignored where configured.
- `release/trace_camera_module_v1/`: frozen camera-module source manifest and release documentation.

Datasets, processed images, feature arrays, checkpoints and generated output directories are excluded from Git.

## 3. Data work completed

### Core manifest

The verified master manifest contains 3,797 canonical records:

| Source | Records |
|---|---:|
| RFMiD | 3,200 |
| IDRiD grading/localisation | 516 |
| IDRiD segmentation | 81 |

The source audit recorded 56 confirmed duplicate groups, 60 confirmed duplicate pairs and 116 records participating in confirmed duplicate groups. RFMiD contains 45 label columns. Camera assignment is resolved for 2,427 Topcon 3D OCT-2000 images and 276 Topcon TRC-NW300 images; 497 records remain unresolved because the archive-resolution counts and paper description do not align.

### Additional datasets audited or used

| Dataset | Work completed | Scientific role and limitation |
|---|---|---|
| FIVES | 800 rows audited; 794 unique photographs; vessel masks and quality proxies catalogued | Real-image quality/preservation proxy; no public per-image patient/camera IDs |
| PALM | 1,200 released images audited; 1,184 unique primary-analysis images after duplicate handling | Two documented cameras, pathologic-myopia transfer and trust analysis; imbalanced devices |
| Paraguay DR V03 | 1,437 images audited and evaluated with the frozen RFMiD head | External hospital/source transfer for DR; common-disease task rather than rare-disease validation |
| JSIEC-1000 | 1,000 class-folder images catalogued | Exploratory label coverage; no verified patient or per-image camera IDs |
| RFMiD 2.0 | Train/validation/test structure and overlaps audited; validation transfer study completed | Small cross-release development check; test image/label IDs do not align reliably |
| EDID | Original archive verified, duplicate review performed and 1,333 target/healthy images evaluated | Exploratory folder-label transfer; no patient or per-image camera IDs |
| EyeQ metadata | 28,792 quality rows present | EyePACS images remain access-gated, so labels are metadata-only |

BRSET was not used because credentialed image access was unavailable.

### Leakage and integrity controls

- Known duplicate/related groups are kept from crossing the conservative development split.
- Exact hashes, perceptual/detail matching and human-review queues are used instead of relying on filenames.
- Dataset-specific limitations and possible public-pretraining exposure are recorded.
- Patient-independent splitting cannot be claimed where patient identifiers were not supplied.
- Dataset identity is not represented as camera identity.

## 4. Camera and preprocessing module

### Active frozen release

`trace-camera-module-v1.0-research` performs:

1. Input validation while retaining the original image.
2. Fundus-field detection.
3. Crop, centring, 448 by 448 resize and circular masking.
4. Technical measurements for blur, noise, glare, compression and retinal coverage.
5. Provisional accept/warn/reject reporting.
6. Optional frozen DINOv2-S/RFMiD research scores, kept separate from quality flags.
7. Standardised image, mask and machine-readable audit output.

Photometric correction is off by default.

### Geometry audit

- 3,605 non-test train/validation records processed.
- Zero processing errors.
- Zero incorrect output dimensions.
- 817 lesion, optic-disc or vessel masks evaluated.
- Zero non-empty annotation masks lost after transformation.
- Zero final-test records accessed by that audit.
- 24 low-confidence/low-fill cases visually inspected, with no visible crop or centring failure found.

The visual inspection was an engineering review, not a retinal-specialist safety review.

### Camera/acquisition findings

- Geometry-only DINOv2 embeddings predicted three RFMiD native-resolution groups at 97.64% balanced accuracy versus 33.33% chance.
- Disease labels alone predicted the same proxy at 52.33%, demonstrating confounding.
- Among 134 images with no positive RFMiD disease label, probe accuracy remained 98.13%, showing residual acquisition information.
- Twelve reference/strength photometric combinations and six individual components were assessed.
- No correction both preserved task performance and reduced embedding-domain identifiability.

Decision: retain geometry and QC reporting; disable colour/saturation/brightness/contrast/gamma/illumination harmonisation. The result supports acquisition leakage, not universal camera invariance.

### Shift-consistency adapter

The residual paired-view adapter changed bounded synthetic-shift AUPRC from 0.5019 to 0.5065 and mean score change from 0.0186 to 0.0121, but clean sensitivity dropped from 66.26% to 60.74%.

Decision: `not_selected`.

### Artifact stress and learned QC

- Mild mixed synthetic stress reduced micro sensitivity from 66.20% to 60.90%.
- Established paired macro-AUPRC losses were 0.0365 for glare, 0.0214 for resolution loss and 0.0378 for narrow field of view.
- The combined learned QC module was rejected because its glare threshold flagged 84.5% of clean validation images, leaving 15.5% clean acceptance.
- On 794 unique FIVES photographs, the frozen synthetic resolution detector achieved AUROC 0.7620 against the automatic blur proxy but only 1.53% sensitivity at 100% specificity.

Decision: no learned artifact detector is active. Resolution and field-of-view models remain research candidates; glare remains unresolved.

### PALM geometry/camera transfer

Pathologic-myopia ranking transferred strongly, but a ZEISS-derived threshold produced only 42.86% specificity on Canon despite AUROC 0.9796. Across 30 metric/condition comparisons, bootstrap analysis did not establish an advantage for TRACE geometry over resize-only preprocessing.

Decision: geometry is retained for field standardisation, annotation transformation and QC, not as proof of camera invariance.

## 5. Frozen disease baseline

The operational development baseline is official DINOv2-S/14 with all encoder weights frozen and a learned 45-output RFMiD linear head.

| Item | Result |
|---|---:|
| Training records | 1,917 |
| Validation records | 639 |
| Embedding width | 384 |
| Learnable outputs | 43 of 45 |
| Outputs with provisional thresholds | 21 of 45 |
| Validation macro AUPRC | 0.515 across 21 better-supported classes |
| Validation macro AUROC | 0.846 across 38 evaluable classes |

HR and ODPM had no positive training examples and therefore return only smoothed priors. These are development results, not clinical diagnostic accuracy or final-test results.

## 6. Few-shot protocol and foundation-model comparisons

### Frozen protocol

The protocol uses a count-based base/novel split, 639 fixed validation queries and 50 paired support draws for K=1, K=3 and K=5. Shared adapters may learn from base labels only, and records positive for a novel label are excluded from the adapter pool.

Protocol controls recorded on 11 November 2026:

- 24 code tests passed at that stage.
- No known duplicate/related group crossed train and validation.
- Label-shuffle macro AUPRC stayed within 0.0099 of prevalence, inside the 0.02 tolerance.
- No protected test feature was accessed.

### DINOv2-S few-shot heads

Regularised logistic regression was selected on base labels. Under the abundant-negative protocol, novel-label macro AUPRC was:

| K | Macro AUPRC |
|---:|---:|
| 1 | 0.1233 |
| 3 | 0.1988 |
| 5 | 0.2266 |

Its paired advantage over a prototype was established at K=3 and K=5, not K=1. Under strict K-positive/K-negative evaluation, the heads were not distinguishable.

### FLAIR, DINOv2-L and RETFound

- FLAIR was numerically higher in six image-only comparisons, but none of the paired 95% intervals excluded zero.
- FLAIR text-only zero-shot macro AUPRC was 0.2514, but RFMiD appears in its pretraining assembly, so it is tagged `possibly_seen`.
- DINOv2-L, evaluated at the same 448-pixel input and episodes, did not establish an advantage over DINOv2-S.
- Official RETFound-DINOv2 used the same 304,368,640-parameter DINOv2-L/14 teacher architecture. None of six paired intervals against DINOv2-L excluded zero.

Decision: DINOv2-S remains the operational baseline. This is a selection for the current protocol, not proof that retinal foundation models are generally inferior.

## 7. Adaptation and TRACE encoder experiments

### Rare-aware supervised adapter

The adapter improved internal base-label macro AUPRC from 0.3623 to 0.5213 but did not transfer to the novel labels:

| K | Frozen DINOv2-S | Adapter |
|---:|---:|---:|
| 1 | 0.1233 | 0.1090 |
| 3 | 0.1988 | 0.1774 |
| 5 | 0.2266 | 0.2075 |

All paired intervals included zero and all six point estimates favoured the unadapted encoder. Decision: `not_selected`.

### Five-epoch label-hidden SSL development study

On 11 low-support labels and the fixed K=3 validation episodes:

| Encoder | Macro AUPRC | Macro AUROC |
|---|---:|---:|
| Frozen DINOv2-L | 0.2044 | 0.7826 |
| Matched crop SSL, epoch 5 | 0.2164 | 0.8039 |
| Full-field SSL, epoch 5 | 0.2189 | 0.8023 |

The full-field minus DINOv2-L interval was -0.0484 to +0.0620. This was inconclusive.

### Twenty-epoch matched SSL continuation

Both students trained on the same 1,917 label-hidden RFMiD training images and reached 9,580 updates.

| Encoder | K=3 macro AUPRC | Macro AUROC |
|---|---:|---:|
| Matched crop SSL, epoch 20 | 0.2363 | 0.8128 |
| Full-field SSL, epoch 20 | 0.2380 | 0.8123 |

Key paired contrasts:

| Contrast | Difference | 95% interval |
|---|---:|---:|
| Full-field 20 minus crop 20 | +0.0017 | -0.0102 to +0.0139 |
| Full-field 20 minus DINOv2-L | +0.0335 | -0.0278 to +0.0849 |
| Crop 20 minus DINOv2-L | +0.0319 | -0.0344 to +0.0902 |
| Full-field 20 minus full-field 5 | +0.0191 | +0.0021 to +0.0362 |
| Crop 20 minus crop 5 | +0.0199 | +0.0022 to +0.0391 |

Interpretation: longer training is the useful development signal. Full-field views did not beat matched crops, and neither epoch-20 student established superiority to DINOv2-L. Only one seed was used and the same validation queries have been reused.

### Frozen feature fusion

Equal-weight DINOv2-L plus RETFound fusion reached macro AUPRC 0.2103 versus 0.2044 for DINOv2-L. The difference was +0.0059 with interval -0.0552 to +0.0500, and RT performance was materially worse.

Decision: do not promote fusion.

## 8. External and cross-release evaluations

### Paraguay DR

The frozen RFMiD DINOv2-S head was evaluated without external fitting on 1,437 photographs:

| Metric | Result |
|---|---:|
| Any-DR AUROC | 0.8876 |
| Any-DR AUPRC | 0.9141 |
| Frozen-threshold sensitivity | 0.3044 |
| Frozen-threshold specificity | 0.9958 |
| Expected calibration error | 0.2402 |
| Nominal 90% conformal coverage | 72.7% |
| Conformal coverage for diseased eyes | 46.7% |

Conclusion: ranking transferred; threshold and uncertainty calibration did not.

### RFMiD 2.0 validation

All 177 validation images were processed successfully for DINOv2-S, DINOv2-L and RETFound. With only 6 to 8 positives per provisional label, results are unstable. At K=5, three-label macro AUPRC was 0.289 for DINOv2-S, 0.239 for DINOv2-L and 0.191 for RETFound. RT dominated the macro result.

Conclusion: exploratory cross-release evidence only; no general model or camera winner.

### EDID exploratory transfer

After duplicate review, 1,333 target/healthy images were evaluated for macular scar, retinitis pigmentosa and disc oedema. The full table is tracked in `data/protocols/trace_bench_v2/EDID_EXPLORATORY_RESULTS.md`.

Main observations:

- Retinitis pigmentosa versus healthy ranked strongly for all three encoders, with AUROC from 0.943 to 0.990 across the tested settings.
- Macular-scar performance varied substantially by encoder, K and classifier head.
- DINOv2-S had the strongest reported disc-oedema AUPRC at K=1, 3 and 5.
- No single encoder won across all tasks.

These are folder-label results without patient/camera IDs or clinical adjudication and do not establish rare-disease diagnosis or camera generalisation.

## 9. Trust, calibration and retrieval

### PALM trust layer

OOD distance detected ZEISS-to-Canon shift at about 0.88 AUROC, but a source-derived 95th-percentile threshold flagged only around 20% to 29% of Canon images. Nominal 90% conformal coverage fell to 76.8% in one transfer direction and to 42.9% for Canon negatives. No method solved both directions at at least 50% coverage.

Decision: retain calibration/OOD outputs as diagnostics; do not claim a universal safety gate.

### Retrieval

The development retrieval index contains 1,917 DINOv2-L RFMiD training embeddings and returns five examples for each of 639 validation queries.

- 91.6% of unrestricted top-five neighbours shared the query's reported acquisition domain.
- Random training examples would match that domain about 63.0% of the time.
- Cross-domain retrieval sharply reduced same-label hits, but label availability was heavily confounded.

Decision: retain retrieval as an inspectable demo with source and label provenance. Cosine similarity is not a disease probability.

## 10. Final-test preparation and present gate

The final workflow compares five locked encoders using identical K=1/3/5 support draws and a fixed C=1 logistic head. The primary contrast is full-field SSL versus DINOv2-L at K=3, with a 5,000-resample paired interval.

Completed:

- Five training-feature files paired by the same 1,917 record IDs and label order.
- Expected embedding widths verified.
- Locked plan and code-hash checks implemented.
- Extraction refuses to run without an approved release.
- Evaluation refuses to run without approved extraction outputs.
- Development replay preflight reproduced ten previous metric pairs across five models.

Unfinished:

- Twelve image-identity review decisions.
- `audit_decisions.csv`.
- `scoring_release.json`.
- Final-test feature extraction.
- Final-test predictions, metrics and conclusions.

The RFMiD v1 archive was read for duplicate auditing, so it is no longer accurate to call the pixels unopened. No final-test encoder feature, disease score or performance metric was computed. Four v1 test/training pairs are highly likely duplicate photographs pending human adjudication. RFMiD 2.0 test scoring remains blocked by image/label ID mismatches and three exact overlaps with its development data.

## 11. Automated verification run on 12 November 2026

Commands were executed from the repository virtual environment.

| Check | Outcome |
|---|---|
| `python -m pytest -q` | **PASS: 32 tests passed in 7.91 seconds** |
| `python -m compileall` over packages, scripts and tests | **PASS** |
| `scripts/validate_master_manifest.py` | **PASS: 3,797 unique canonical records** |
| `scripts/preflight_trace_final_v1.py` | **PASS_DEVELOPMENT_REPLAY_ONLY: 10 episodes, 5 models, 0 protected test records read** |
| `scripts/check_trace_mac_readiness.py` on this Windows PC | CUDA available; current SSL trainer is not MPS-ready |
| `scripts/verify_environment.py` | **FAIL on two optional imports**: Windows Application Control blocked native `stringzilla` through Albumentations and `_dynfunc` through UMAP |

The environment failure is recorded rather than hidden. Core tests and current CUDA workflows pass, but Albumentations/UMAP-dependent commands may fail on this Windows policy until those packages are replaced, permitted or run in another environment.

The legacy master-manifest validator also prints `final_split: pending duplicate-aware assignment`; later split manifests exist, so that status field should be updated before it is used as a release summary.

## 12. Knowledge graph status and approved design

Prepared source snapshots include MONDO Rare Disease, HPO, UBERON, Relation Ontology, ECO, Orphadata, HGNC, ClinGen and Biolink, with checksums and licence notes. Restricted OMIM/UMLS/SNOMED bulk content is not included.

Not completed:

- No medical assertions have been manually curated.
- No disease/finding/artifact graph has been built.
- No finding-head-to-graph interface has been validated.
- No graph reranking experiment has been run.

Planned useful role:

- Primary input: calibrated encoder `disease_scores`.
- Optional inputs: scores only from genuinely evaluated finding heads and recorded QC/artifact flags.
- Output: disease reranking, supporting/opposing evidence, provenance and abstention.
- Missing finding evidence remains unknown rather than negative.
- Every edge is a first-class assertion with source, locator, evidence type, curator and review state.
- Initial scope: about 10 rare retinal diseases, 5 mimics and fundus-visible findings relevant to TRACE.
- Implementation: validated TSV/SSSOM tables, NetworkX reasoning, JSON explanation payload and JSON-LD/Turtle export.
- Required comparison: encoder alone; encoder plus priors/co-occurrence; encoder plus plain HPO overlap; encoder plus TRACE evidence graph.

The graph reranker will be enabled only if it improves a predeclared development endpoint without violating a clean-ranking non-inferiority margin. A failed reranker may still supply curated explanations, but it must not alter disease scores.

## 13. Work remaining, in dependency order

1. Complete the 12 human image-identity decisions and freeze the final RFMiD v1 scoring release.
2. Run the existing five-model comparison once without changing the locked protocol.
3. Preserve a fresh labelled query cohort before selecting another encoder.
4. Build TRACE-Bench v2 with repeated K=0/1/3/5/10/20 episodes, normalised AUPRC, calibration, risk-coverage and paired uncertainty.
5. Evaluate the strongest frozen comparator set on identical development episodes, including DINOv3-L and RetiZero only if weights, licences and contamination status are verified.
6. Diagnose whether the dominant model failure is lesion scale, source bias or threshold/calibration transfer.
7. Train a single evidence-driven retinal SSL candidate and matched image-only controls with at least three seeds and a larger legal, deduplicated, source-balanced unlabelled pool.
8. Run source/device-shift evaluation where genuine metadata permit it and a site-recalibration label-budget study.
9. Build and curate the TRACE evidence graph in parallel, then run its four-way controlled ablation.
10. Add leakage-safe patch/image retrieval, calibration/OOD display, abstention and quality warnings to the research demo.
11. Run one frozen confirmatory comparison and report positive, null and negative outcomes.
12. Package the encoder, inference interface, model/data cards, contamination ledger, UMAP, ablations, latency, memory, compute record and reproducible commands.

## 14. Permitted claims today

Supported:

- TRACE has a frozen, camera-ID-agnostic fundus preprocessing and quality-reporting module.
- Acquisition/source signal remains present in encoder representations.
- Tested photometric harmonisation and learned QC variants failed their selection gates and were disabled.
- Longer label-hidden retinal SSL improved the reused development endpoint relative to its five-epoch checkpoints.
- External ranking can transfer while thresholds and calibration fail.
- The final comparison is gated against known leakage and has not been scored.

Not supported:

- Universal camera generalisation.
- Patient-independent clinical accuracy.
- A statistically confirmed superior TRACE encoder.
- Reliable disease probability on arbitrary images.
- A clinically safe abstention mechanism.
- A beneficial knowledge-graph reranker.
- Final-test performance.

## 15. Reproduction boundary

The Git repository intentionally contains code, small manifests, fixed protocols, test files and result summaries. It does not contain retinal images, model weights, feature arrays, generated predictions or credentials. Reproducing numerical experiments therefore requires lawful dataset access, the documented checkpoints or weight-download steps, matching hashes and the pinned environment.
