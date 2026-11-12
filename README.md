# TRACE Preprocessing Module

Current two-laptop assignments: [sprint order](TRACE_CURRENT_SPRINT_ORDER.md) and
[paste-ready friend prompt](TRACE_FRIEND_FEATURE_PROMPT.md). These supersede the
older broad collaboration plan; friend work is on `feature`, with an
explanation-only graph.

The current PC-side model study is frozen under
`data/protocols/trace_source_balance_v1/`. Its core-only manifest and separate
fixed-update trainer are ready; expanded-pool arms remain blocked until
record-level dataset admission is complete.

For the complete verified project status, completed experiments, test results,
limitations and next work, see
[`TRACE_VERIFIED_PROJECT_REPORT_2026-11-12.md`](TRACE_VERIFIED_PROJECT_REPORT_2026-11-12.md).

The evidence-locked camera component is now frozen as
`trace-camera-module-v1.0-research`. Run it with
`scripts/run_trace_camera_module.py`; release notes and the immutable file
manifest are under `release/trace_camera_module_v1/`. Completion boundaries are
recorded in `CAMERA_MODULE_COMPLETE.md`.

This workspace is the data and preprocessing foundation for TRACE. The precise
name is **TRACE Preprocessing and Harmonization Module**.

Its intended job is to apply the same **image-adaptive rule** to any supported
colour fundus photograph without requiring a camera-model label:

1. validate the input and retain the untouched original;
2. detect/crop/centre the retinal field and apply a common mask and size;
3. measure each image's colour, saturation, brightness, contrast, gamma/tonal
   response, and illumination non-uniformity;
4. optionally test bounded corrections toward a train-derived reference range;
5. measure blur, noise, glare, compression, field coverage, and correction
   magnitude;
6. reject or flag unsupported/unsafe corrections instead of forcing every image
   to look acceptable;
7. use the frozen DINOv2/RFMiD disease baseline to enforce a task-performance
   non-inferiority guard and report model scores separately from quality flags.

The currently active default is geometry plus quality control; photometric
correction is disabled because it failed the frozen-model selection criteria.
This is **camera-ID-agnostic preprocessing**, not proven universal camera
generalization. Without real per-image camera labels, experiments can test
robustness across datasets, resolution/appearance strata, quality levels, and
controlled synthetic shifts, but cannot honestly claim performance on every
camera.

## Current verified data

| Source | Canonical records | TRACE role | Important limitation |
|---|---:|---|---|
| RFMiD | 3,200 | multi-label retinal disease development/evaluation | no patient IDs; 497 images remain camera-unresolved because archive resolution counts disagree with the paper |
| IDRiD grading + localisation | 516 | DR/DME labels and optic-disc/fovea coordinates | overlaps RFMiD; not independent |
| IDRiD segmentation | 81 | lesion and optic-disc preservation | possible same-source links to grading records |
| FIVES | 800 released rows / 794 unique photographs | vessel preservation, quality flags, AMD/DR/glaucoma/normal checks | single-camera role; no public camera or patient IDs |
| PALM | 1,200 released / 1,184 unique photographs | two-camera pathologic-myopia and annotation-preservation evidence | duplicate copies cross official splits; cameras are imbalanced |
| JSIEC-1000 | 1,000 photographs / 39 class folders | rare and multi-condition label coverage; graph mapping | no verifiable patient or per-image camera IDs; licence is only `other-open` |
| Paraguay DR V03 | 1,437 photographs | external hospital/population DR transfer and trust evaluation | DR-focused, not a rare-disease training source |

FIVES contains six pairs of byte-identical photographs with different vessel
mask files, plus two empty vessel masks. These are explicitly flagged in its
manifest. The published archive and all original ZIP files remain unchanged.

The selected open external addition is **PALM** because it has 1,200 fundus
images from 720 subjects, two documented camera models, pathologic-myopia
labels, and optic-disc, fovea, atrophy and detachment annotations.  Its camera
counts are imbalanced, so it adds useful external evidence but cannot prove
generalization to every camera. The archive contains 16 exact duplicate pairs,
11 crossing official splits; the TRACE manifest keeps 1,184 unique primary
analysis photographs and excludes duplicate copies with test-first priority.
BRSET remains inaccessible and is not required.

EyeQ's 28,792 quality-label rows are also present, but the corresponding
EyePACS photographs remain access-gated on Kaggle. The labels are therefore
metadata-only for now. The machine-readable knowledge-graph foundations
(Mondo rare, HPO, Uberon, RO, ECO, Orphadata, HGNC, ClinGen and Biolink) are
snapshotted under `data/knowledge_sources/`; no medical assertions have yet
been curated from them.

## Manifests

- `data/manifests/master_manifest.csv`: 3,797 canonical RFMiD/IDRiD records.
- `data/manifests/duplicate_pairs.csv`: confirmed/candidate visual relations.
- `data/manifests/manifest_validation.json`: independent source/annotation audit.
- `data/manifests/trace_analysis_manifest.csv`: conservative split-grouped
  analysis manifest.
- `data/manifests/trace_split_summary.json`: split policy and movement counts.
- `data/manifests/fives_manifest.csv`: 800 FIVES rows with quality, mask,
  SHA-256, and duplicate status.
- `data/manifests/fives_manifest_summary.json`: FIVES audit results.

The final TRACE split guarantees that known duplicate/related groups do not
cross splits. It cannot guarantee patient-independent splitting because patient
identifiers were not supplied.

## Research-safety boundary

There is currently no medical safety reviewer. Initial preservation checks will
therefore use public expert/annotated proxies: IDRiD lesion/optic-disc masks and
FIVES vessel masks, plus bounded corrections, an abstention gate, and the frozen
model's label-based non-inferiority test once supplied. This supports a research
prototype; it does not establish clinical safety or replace a blinded retinal
specialist review.

## Environment

The local `.venv` contains the preprocessing, optimisation, statistics, testing,
and GPU-learning packages in `requirements-core.txt` and
`requirements-pytorch-cu128.txt`. PyTorch uses CUDA 12.8, which is compatible
with the installed driver and RTX 4060 Laptop GPU.

Recheck it with:

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe scripts\validate_master_manifest.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Implemented preprocessing structure

The model-independent development structure is implemented in
`trace_preprocessing/`:

- manifest-driven ZIP/local image and annotation loading;
- default blocking of the final test split;
- RGB/resolution validation and fundus-field detection;
- crop, centre, resize and common-mask geometry;
- coordinate and lesion/vessel-mask transformation;
- colour, saturation, brightness, contrast, gamma and illumination measurements;
- blur, noise, glare, field-coverage and compression measurements;
- train-only robust reference-profile construction;
- bounded adaptive photometric correction as an experimental, disabled path;
- before/after safety comparison and provisional accept/warn/reject routing;
- annotation-aware lesion/vessel preservation measurements;
- image-free JSON audit records;
- a separate DINOv2 tensor adapter.

The current thresholds are explicitly development guards, not clinically
validated acceptance limits. The smoke reference under
`outputs/preprocessing_smoke/` uses six training images only to verify code and
must never be presented as the final TRACE reference standard.

### Camera-robustness experiment decision

The development decision is recorded in
`outputs/camera_module_decision/current_decision.json`. The important result is
negative: **no photometric filter or learned adapter has earned deployment**.

- Geometry-only DINOv2 embeddings identified the three RFMiD native-resolution
  groups at 97.64% balanced accuracy versus 33.33% chance. The official paper
  maps two groups cleanly, but the 4288x2848 group remains camera-unresolved due
  to a 30-image paper/archive count mismatch.
- The proxy is confounded: disease labels alone identified it at 52.33%.
- Acquisition leakage is nevertheless real: among 134 images with no positive
  RFMiD disease label, embedding-probe accuracy was still 98.13%.
- Three train-derived reference profiles and 12 reference/strength combinations
  were screened; six individual photometric components were then ablated.
- No direct correction simultaneously preserved disease performance and reduced
  embedding-domain identifiability. Photometric correction therefore defaults
  to off in `trace-preprocessing-dev-v0.2`.
- A residual paired-view adapter improved bounded synthetic-shift AUPRC from
  0.5019 to 0.5065 and reduced mean score change from 0.0186 to 0.0121, but clean
  sensitivity fell from 66.26% to 60.74%. Its checkpoint is retained for audit
  and explicitly marked `not_selected`.

These results do not establish unseen-camera generalization. The strongest next
evidence would be a camera-labelled external dataset plus a fresh development
split; repeatedly tuning on the current validation split would only overfit it.

Experiment 4A now characterises the frozen pipeline under paired camera-artifact
stress. Mild mixed stresses reduced micro sensitivity from 66.20% to 60.90%,
although their macro-AUPRC interval crossed zero. In the isolated v2 benchmark,
glare, resolution loss and narrow field of view produced established paired
macro-AUPRC losses of 0.0365, 0.0214 and 0.0378 respectively. Current QC found
only 3.4% of glare views and 2.0% of resolution-loss views that removed a
previously correct positive decision; its field-of-view rule detected 71.4%
but flagged 478/639 views. The next target is therefore train-derived,
artifact-aware QC—not stronger colour harmonisation or a new encoder. See
`outputs/rfmid_artifact_stress_v2/EXPERIMENT_4A_REPORT.md`.

Experiment 4B trained three grouped, train-only logistic QC detectors from 33
single-image features. The predeclared combined module was rejected: its glare
threshold flagged 84.5% of clean validation images, reducing clean acceptance
to 15.5%. Resolution-loss and narrow-field detectors were nearly perfect on
synthetic train views and accepted 99.84% of clean validation images when glare
was removed, but this partial rule was defined after observing the combined
failure and is only exploratory. The full QC module remains `not_selected`;
resolution and field-of-view detectors are frozen candidates for fresh real
validation, while glare remains unsolved. See
`outputs/artifact_qc_v1/EXPERIMENT_4B_REPORT.md`.

Experiment 4C performed a frozen, no-retuning real-image proxy validation on
794 unique FIVES photographs. The synthetic resolution detector ranked the
automatic poor-blur labels moderately (AUROC 0.7620), but its frozen warning
threshold detected only 2/131 poor-blur images (1.53% sensitivity, 100%
specificity). It failed the predeclared AUROC and sensitivity gates and is not
selected. FIVES lacks a matching field-completeness label, so that detector
also remains disabled. No learned artifact-QC detector is currently active.
See `outputs/fives_frozen_artifact_qc_v1/EXPERIMENT_4C_REPORT.md`.

PALM Experiment 2A now adds a disease-transfer baseline using its official
Canon CR-2 and ZEISS Visucam 500 labels. Disease ranking transferred well, but
a ZEISS-only source threshold fell to 42.86% specificity on Canon despite
0.9796 AUROC, indicating score/threshold shift. A paired 5,000-bootstrap
ablation found no established disease-transfer advantage for TRACE geometry
over direct resize-only in any of 30 metric/condition comparisons; the same
ZEISS-to-Canon specificity failure occurred under both. Geometry remains
selected for field standardisation, annotation transformation and QC, not as a
camera-invariance claim. See
`outputs/palm_camera_transfer_v1/EXPERIMENT_2A_REPORT.md`.

### Trust-layer experiment decision

PALM Experiment 3A evaluates source-only Platt calibration, kNN and
Mahalanobis camera-shift scores, confidence/OOD abstention, risk–coverage and a
class-conditional conformal diagnostic. OOD distance detected ZEISS-to-Canon
shift (about 0.88 AUROC) but flagged only 20–29% of Canon images at a
source-training 95th-percentile threshold and ranked case-level errors worse
than ordinary prediction confidence. Nominal 90% conformal coverage fell to
76.8% for ZEISS-trained predictions on Canon and to 42.9% for Canon negatives.
Confidence abstention improved Canon-to-ZEISS risk at useful coverage, but no
method improved the unresolved ZEISS-to-Canon direction at 50% or greater
coverage. The trust layer is therefore retained as diagnostics, not selected
as a universal safety mechanism. See
`outputs/palm_trust_layer_v1/EXPERIMENT_3A_REPORT.md`.

### Geometry audit status

The complete non-test geometry audit is under `outputs/geometry_audit/`:

- 3,605 train/validation records processed;
- zero processing errors and zero wrong output dimensions;
- 817 lesion, optic-disc or vessel masks evaluated;
- zero nonempty annotation masks lost after transformation;
- zero final-test records accessed;
- 24 provisional low-confidence/low-fill records visually reviewed;
- no visible crop or centring failure found in those 24 records.

The visual inspection was an AI engineering review, not a clinical safety
review. Very dark or partially visible photographs remain quality-control cases
for the later accept/warn/reject policy.

## Frozen disease-model baseline

The development baseline is now implemented with the pinned official
**DINOv2-S/14** encoder (all encoder weights frozen) and a trained 45-output
linear RFMiD head. Only the RFMiD training and validation splits were used;
the final test split was not opened.

- Training records: 1,917; validation records: 639.
- Encoder representation: 384 values per geometry-normalized image.
- Learnable outputs: 43/45. RFMiD has no positive training examples for HR or
  ODPM, so those two outputs are explicitly marked unsupported and return only
  a smoothed prior.
- Thresholded outputs: 21/45. A provisional yes/no threshold is saved only when
  validation contains at least five positives.
- Validation macro AUPRC: 0.515 across those 21 better-supported classes.
- Validation macro AUROC: 0.846 across 38 classes with at least one positive.
- These are development/model-selection figures, not final-test or clinical
  performance claims. Scores are not the same as diagnostic certainty.

Artifacts are under `data/features/` and
`models/rfmid_dinov2_linear_v1/`. The callable image-to-score interface is
`trace_models.TraceRFMiDScorer`; the command-line entry point is:

```powershell
.\.venv\Scripts\python.exe scripts\predict_rfmid_image.py --image C:\path\to\fundus.jpg --top-k 10
```

PALM is the next open external dataset. Even after it is added, the filter can
be evaluated only on the documented devices and domains; it cannot support a
universal-camera claim.

## Frozen few-shot protocol

Experiment 0 is stored under `data/protocols/trace_fewshot_v1/`. It defines a
count-only base/novel label split, fixed validation queries, and 50 paired
support draws for each novel label at K = 1, 3 and 5. Every future encoder must
use the same episode plan. Shared adapters are allowed to learn from base labels
only; training records positive for any novel label are excluded from the base
adapter pool. Dataset support counts are an operational definition of low data,
not evidence that a condition is clinically rare. No final-test features are
loaded by the protocol builder.

The Experiment 0 controls passed on 2026-11-11: 24 code tests passed, no known
duplicate/related group crossed train and validation, the label-shuffle control
remained within 0.0099 macro AUPRC of prevalence (0.02 tolerance), and no test
features were accessed. The full development-only report is at
`outputs/fewshot_protocol_v1/EXPERIMENT_0_REPORT.md`.

Experiment 1A is complete under `outputs/dinov2_fewshot_v1/`. Base-label tuning
selected regularised logistic regression, but it was statistically tied with
kNN on novel labels. Under the main abundant-negative protocol, logistic macro
AUPRC was 0.1233, 0.1988 and 0.2266 at K = 1, 3 and 5. Its paired improvement
over a simple prototype was established at K = 3 and 5, not K = 1. Under the
strict K-positive/K-negative protocol the heads were not distinguishable. No
test features were accessed. See `EXPERIMENT_1A_REPORT.md` in that output
directory for the full limitations and intervals.

Experiment 1B compares the generic encoder with the official FLAIR retinal
vision-language model. FLAIR was numerically higher in all six image-only
few-shot comparisons, but none of the paired 95% confidence intervals excluded
zero. Its base-selected text-only zero-shot method achieved 0.2514 novel-label
macro AUPRC; however, RFMiD is listed in FLAIR's pretraining assembly, so this
result is tagged `possibly_seen` and is not an unseen-dataset claim. Adding an
image cache did not establish an improvement over text-only zero-shot and was
rejected from the primary method. The locked report is at
`outputs/encoder_comparison_v1/EXPERIMENT_1B_REPORT.md`.

The size-matched capacity control is also complete. Official DINOv2-L/14
(304,368,640 parameters) was evaluated at the same 448 px resolution and on
the same support/query episodes. None of its six paired comparisons against
DINOv2-S excluded zero, and none of FLAIR's six comparisons against DINOv2-L
excluded zero. DINOv2-S therefore remains the primary generic baseline;
DINOv2-L is retained only as a documented capacity control.

The gated official RETFound-DINOv2 MEH checkpoint was then obtained and
verified. Its teacher backbone strictly matched the same 304,368,640-parameter
DINOv2-L/14 architecture, and both encoders were evaluated at 448 px on the
identical episodes. RETFound was numerically lower under all three
abundant-negative K settings and slightly higher or tied under strict K+/K−;
none of the six paired 95% intervals excluded zero. A fixed-logistic control
gave the same conclusion. RETFound therefore does not earn selection, and the
much smaller DINOv2-S remains TRACE's primary image encoder. See
`outputs/encoder_comparison_v1/EXPERIMENT_1C_RETFOUND_REPORT.md`.

## External-site and trust validation

Experiment 3A tested calibration, OOD warnings, conformal sets and selective
prediction under the two documented PALM cameras. Platt scaling helped in some
directions, but no universal trust gate survived both shift directions.
Confidence was generally a stronger ranking signal than distance on PALM, yet
the cross-camera coverage failures prevented a safety claim. The report is at
`outputs/palm_trust_layer_v1/EXPERIMENT_3A_REPORT.md`.

Experiment 3B then froze the existing RFMiD DINOv2-S head and evaluated it on
all 1,437 audited Paraguay DR photographs without external fitting. Any-DR
ranking transferred (AUROC 0.8876, AUPRC 0.9141), but the frozen RFMiD threshold
did not: sensitivity fell to 0.3044 while specificity rose to 0.9958. ECE was
0.2402 and nominal 90% conformal coverage fell to 72.7%, including only 46.7%
coverage for diseased eyes. Confidence-only abstention retained confident
false negatives; embedding-distance rejection was less bad but not sufficient.
This supports external-site representation transfer, not universal-camera or
clinical reliability. See
`outputs/paraguay_external_shift_v1/EXPERIMENT_3B_REPORT.md`.

## Rare-aware adapter decision

Experiment 5A trained a small residual adapter on frozen DINOv2-S features
using ten base RFMiD labels. All training records positive for any of the eleven
novel evaluation labels were excluded. The adapter improved internal base-label
macro AUPRC from 0.3623 to 0.5213 and slightly improved synthetic paired-view
consistency, but those gains did not transfer to unseen labels. Under the main
abundant-negative protocol, novel-label macro AUPRC changed from 0.1233 to
0.1090 at K=1, 0.1988 to 0.1774 at K=3 and 0.2266 to 0.2075 at K=5. All paired
95% intervals included zero and all six point estimates favored the unadapted
encoder. The adapter is therefore `not_selected`; frozen DINOv2-S remains the
primary TRACE image encoder. No protected-test records were accessed. See
`outputs/trace_rare_adapter_fewshot_v1/EXPERIMENT_5A_REPORT.md`.

## Selected label-hidden A1 encoder — 13 November 2026

The separately frozen source-balance study continued DINOv2-S with rank-8 LoRA, an EMA teacher and the existing contrastive queue for 9,580 updates on the 1,917 label-hidden RFMiD-v1 training photographs. Across three seeds, A1 improved fixed development macro AUPRC over its own frozen initialization by a mean **+0.0341 at K=3** and **+0.0474 at K=5**. All three seeds cleared the +0.02 screen, although EDN worsened consistently and the result remains development-only.

The audited RFMiD+FIVES expansion did not justify a more complex model. Natural sampling added only +0.0040/+0.0052 AUPRC at K=3/5, and source balancing added only +0.0024/+0.0017 beyond that. Both uncertainty intervals crossed zero, so A2/A3 were stopped rather than replicated. See `data/protocols/trace_source_balance_v1/DEVELOPMENT_RESULT.md`, `EXPANDED_POOL_RESULT.md` and `MODEL_CARD.md`.

The selected predeclared seed has a strict image-to-embedding interface:

```powershell
.\.venv\Scripts\python.exe scripts\embed_trace_a1_image.py --image C:\path\to\fundus.jpg --output embedding.npy
```

It verifies the checkpoint/configuration, runs frozen geometry-only preprocessing and emits 384 values. A real-image smoke test produced a finite `(384,)` float32 vector. The embedding is not a calibrated probability or diagnostic confidence, and no camera-generalization or clinical-safety claim follows from this result.

`scripts/score_trace_fewshot_embeddings.py` fits the same balanced `C=1` logistic head used in the locked evaluation from positive and negative support matrices, then writes query ranking margins. A real-feature smoke test with three positives, 20 negatives and five queries passed. These margins are intentionally not presented as probabilities.

For a complete folder-based research run, use `scripts/run_trace_fewshot_images.py` with positive-support, negative-support and query folders. It performs exact-duplicate role checks, preprocessing, embedding, QC logging and few-shot ranking in one audited command. An end-to-end six-image smoke run completed with two queries scored and none rejected. The caller remains responsible for label validity and patient/eye separation.
