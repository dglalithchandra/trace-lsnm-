# TRACE complete audit and next-steps review prompt

**Prepared:** 11 November 2026  
**Workspace:** `C:\Users\D Lalith Chandra\Desktop\eyeimage`  
**Purpose:** self-contained evidence dossier and prompt for an independent AI reviewer

---

## Copy everything below into the other AI

You are acting as a brutally honest independent medical-AI research reviewer.
Review the TRACE project described below and recommend the strongest next course
of action. Do not praise the project merely because it contains many components.
Distinguish verified results from interpretations and proposals. Identify
leakage, multiple-comparison, dataset-overlap, clinical-validity, novelty and
evaluation problems. If internet access is available, independently check the
current literature using primary papers, official model cards and official
dataset pages; cite sources and dates. Do not claim novelty without a literature
comparison.

### 1. Challenge and project objective

The challenge theme is **“Contrastive Learning for Rare Eye Diseases — Learn
More From Less Data.”** Suggested concepts include CLIP-style learning,
self-supervised learning, foundation models and few-shot learning. Suggested
problems include rare retinal disease detection, OCT biomarker discovery,
cross-device generalisation and domain adaptation.

TRACE has been narrowed to colour fundus photographs. Its intended research
question is:

> Can a retinal representation learn a new low-data disease from one, three or
> five labelled positive examples while remaining reliable under acquisition,
> site and camera shift?

It is a research prototype, not a clinical diagnostic system.

### 2. Current TRACE system

The currently selected pipeline is:

1. Validate the input image.
2. Detect the retinal/fundus field.
3. Crop, centre and resize it to a standard circular field with a common mask.
4. Measure image quality and issue an accept/warning status using conservative
   deterministic checks only.
5. Encode the image using official frozen DINOv2-S/14 at 448 px.
6. Use either:
   - a frozen RFMiD multi-label linear disease head for development disease
     scores; or
   - a few-shot classifier fitted from 1/3/5 labelled examples for a requested
     disease.
7. Treat score calibration, OOD distance and abstention as diagnostics, not
   guaranteed safety mechanisms.

The following are disabled because experiments did not support them:

- six-characteristic photometric harmonisation;
- the synthetic-shift consistency adapter;
- learned glare, resolution-loss and field-of-view QC detectors;
- the rare-aware residual adapter from Experiment 5A.

### 3. Data actually used

#### Core manifest

- **RFMiD:** 3,200 images. Duplicate-aware working split: 1,917 train, 639
  validation and 644 protected test. The protected test has not been used in
  the reported model-selection experiments.
- **IDRiD grading/localisation:** 516 canonical images.
- **IDRiD segmentation:** 81 images with lesion/optic-disc masks.
- The unified core manifest contains 3,797 canonical records.
- There are 56 confirmed duplicate groups involving 116 records. Duplicate
  groups and conservatively linked IDRiD records do not cross the generated
  split boundaries.
- Patient independence cannot be guaranteed because patient IDs are absent.

#### External datasets used for specific validation

- **PALM:** pathologic-myopia labels with documented ZEISS Visucam 500 and
  Canon CR-2 domains; used for cross-camera transfer and trust experiments.
- **Paraguay DR:** 1,437 images from a new hospital/ZEISS domain; used only for
  frozen external DR evaluation.
- **FIVES:** 794 unique photographs; used for frozen real-image proxy
  validation of artifact QC. Its quality labels in this experiment were
  automatically derived proxies, not clinician grades.

#### Available locally but not equivalent to completed evidence

- External-data folders exist for EyeQ and JSIEC, and downloaded archives or
  knowledge-source files exist. Their presence must not be represented as a
  completed training or validation experiment.
- A knowledge graph was discussed as a separate optional project, but no graph
  result is part of the validated TRACE pipeline described here.

Camera metadata limitation: RFMiD per-image camera identity is not available.
Native-resolution groups were used as acquisition proxies. Two mappings are
supported by the paper/table, but the 4288×2848 group remains unresolved due to
a 30-image archive/paper count mismatch. Proxy results must not be called
verified camera-model results.

### 4. Completed preprocessing and camera work

#### Geometry

The camera module detects the fundus, crops and centres it, standardises the
field and size, produces a common circular mask, and transforms available
lesion/vessel/optic-disc masks into the same coordinates.

Audit results:

- 3,605 non-test development records processed;
- 0 processing errors;
- 817 non-empty annotation masks evaluated;
- 0 non-empty masks lost after transformation;
- 24 low-confidence/low-fill cases visually inspected with no visible crop or
  centring failure;
- the inspection was an AI engineering review, not a clinician safety review.

The camera module is frozen as `trace-camera-module-v1.0-research`.

#### Acquisition leakage

- A linear probe predicted three RFMiD resolution/acquisition proxy groups
  from geometry-only DINOv2 embeddings with 0.9764 balanced accuracy versus
  0.3333 chance.
- Disease labels alone predicted the proxy at 0.5233, showing confounding.
- On 134 images with no positive RFMiD disease label, the embedding probe still
  reached 0.9813 balanced accuracy.

Interpretation: strong acquisition information remains in the representation,
but these are resolution/acquisition proxies rather than verified cameras.

#### Photometric harmonisation

Three train-derived reference profiles, 12 reference/strength combinations and
six individual photometric components were evaluated. No candidate both:

1. preserved disease ranking/sensitivity; and
2. reduced embedding-domain identifiability.

Decision: photometric correction is **not selected** and is off by default.

#### Synthetic-shift consistency adapter

- Clean macro AUPRC: 0.51502 baseline versus 0.51502 adapter.
- Clean micro sensitivity: 0.6626 baseline versus 0.6074 adapter.
- Stressed macro AUPRC: 0.5019 baseline versus 0.5065 adapter.
- Mean paired score change: 0.0186 baseline versus 0.0121 adapter.

It improved synthetic stability slightly but caused an unacceptable clean
sensitivity loss. Decision: **not selected**.

#### Artifact stress and QC

Controlled paired stresses reduced macro AUPRC:

- glare: −0.0365;
- resolution loss: −0.0214;
- narrow field of view: −0.0378.

The learned combined QC module accepted only 15.49% of clean validation images,
mainly because its glare detector over-flagged images. On FIVES, the frozen
resolution detector achieved AUROC 0.7620 against an automatic poor-blur proxy,
but only 1.53% sensitivity at its frozen warning threshold with 100%
specificity. FIVES did not provide matching field-completeness ground truth.

Decision: all learned artifact-QC detectors are disabled. Fresh expert-labelled
real quality development and held-out evaluation data are required.

### 5. Frozen disease-model baseline

The official DINOv2-S/14 encoder is frozen. A 45-output RFMiD linear head was
trained using RFMiD training data and selected using validation data.

- Representation width: 384.
- Learnable outputs: 43/45; HR and ODPM had no positive training examples.
- Provisional thresholds: 21/45 outputs with at least five validation positives.
- Validation macro AUPRC: 0.515 across the 21 better-supported classes.
- Validation macro AUROC: 0.846 across 38 classes with positives.

These scores are not calibrated diagnostic certainty and are not clinical
performance claims.

### 6. Locked rare-disease few-shot protocol

The protocol is stored as `trace-fewshot-v1`.

- Base labels (10): DR, ARMD, MH, DN, MYA, BRVO, TSLN, ODC, ODP, ODE.
- Novel labels (11): ERM, LS, MS, CSR, CRVO, AION, RT, RS, CRS, EDN, RPEC.
- K values: 1, 3 and 5 positives.
- 50 fixed support episodes for every novel label and K.
- The same 639 RFMiD validation queries are used for paired comparisons.
- Main protocol: abundant negatives.
- Sensitivity protocol: K positive and K negative support examples.
- Any shared supervised adapter must exclude every training record positive for
  a novel label.
- Known duplicate/related train-validation crossings: 0.
- Near-duplicate candidates at cosine >0.9999: 0.
- Label-shuffle macro-AUPRC stayed within 0.0099 of prevalence, inside the
  predeclared 0.02 tolerance.

Important caveat: low dataset support is an operational low-data definition,
not proof that each condition is clinically rare. Repeated use of the same
validation labels for many model comparisons creates model-selection pressure,
even though paired statistics and locked episodes are used.

### 7. Few-shot encoder results

#### DINOv2-S baseline

Base-label tuning selected regularised logistic regression. Under the main
abundant-negative novel-label protocol, macro AUPRC was:

- K=1: 0.1233;
- K=3: 0.1988;
- K=5: 0.2266.

Its improvement over a simple prototype was established at K=3 and K=5, but
not K=1. Under strict K+/K− support, the heads were not distinguishable.

#### FLAIR retinal vision-language encoder

FLAIR was numerically higher in all six image-only comparisons, but none of the
paired 95% confidence intervals excluded zero. Text-only zero-shot macro AUPRC
was 0.2514. However, RFMiD is listed in FLAIR's pretraining assembly, so these
results are tagged **possibly seen** and cannot establish clean unseen-dataset
generalisation. Adding an image cache did not establish improvement and was
rejected.

#### DINOv2-L capacity control

Official DINOv2-L/14 has 304,368,640 parameters and was evaluated at the same
448 px resolution on identical episodes. None of six paired comparisons versus
DINOv2-S excluded zero. The much smaller DINOv2-S remains preferred.

#### RETFound-DINOv2 MEH

The verified official checkpoint matches the DINOv2-L/14 architecture. Against
size-matched DINOv2-L, RETFound was numerically lower for all abundant-negative
K settings and slightly higher/tied under strict K+/K−; none of six paired
intervals excluded zero. A fixed-logistic control gave the same conclusion.
RETFound is not selected.

### 8. Experiment 5A: rare-aware adapter

A small residual adapter was trained over frozen DINOv2-S features:
LayerNorm → Linear 384→128 → GELU → Linear 128→384, with an identity skip and
L2-normalised output. The final layer was zero-initialised so it began as the
original representation.

- Eligible base-only records: 1,200.
- Internal fit/dev: 1,020/180.
- Novel-positive records used: 0.
- Validation records used for adapter fitting/selection: 0.
- Protected-test records accessed: 0.
- Selected objective: episodic base-label metric training with conservative
  identity and paired-view consistency penalties.
- Internal base-label macro AUPRC: 0.3623 frozen versus 0.5213 adapted.
- Mean validation cosine to original representation: 0.9686.
- Synthetic paired-stress cosine: 0.9928 frozen versus 0.9951 adapted.

But unseen novel-label results were worse in all six settings:

| Protocol | K | DINOv2-S | Adapter | Difference | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Abundant negatives | 1 | 0.1233 | 0.1090 | −0.0143 | [−0.0911, 0.0516] |
| Abundant negatives | 3 | 0.1988 | 0.1774 | −0.0214 | [−0.0993, 0.0510] |
| Abundant negatives | 5 | 0.2266 | 0.2075 | −0.0191 | [−0.1001, 0.0559] |
| Strict K+/K− | 1 | 0.0713 | 0.0665 | −0.0048 | [−0.0724, 0.0594] |
| Strict K+/K− | 3 | 0.1058 | 0.0956 | −0.0102 | [−0.0880, 0.0668] |
| Strict K+/K− | 5 | 0.1140 | 0.1098 | −0.0041 | [−0.0781, 0.0701] |

The intervals used 5,000 paired hierarchical bootstrap replicates over labels,
support episodes and stratified validation-query resamples. The adapter failed
the predeclared gate and is **not selected**. The likely explanation is
base-task specialisation without transfer to unseen rare labels.

### 9. Real camera/site transfer and trust evidence

#### PALM camera transfer

Using documented ZEISS and Canon domains for pathologic myopia:

- ZEISS-trained ranking on Canon: AUROC 0.9796.
- The ZEISS-derived threshold achieved only 42.86% specificity on Canon.
- In 30 paired metric/condition comparisons, TRACE geometry did not establish
  an advantage over direct resize-only; zero intervals excluded zero.
- Canon validation contained only seven negative images.

Interpretation: ranking transferred, but threshold/calibration shifted.
Geometry is retained for field standardisation and annotation handling, not as
proven camera invariance.

#### PALM trust layer

Platt calibration, kNN/Mahalanobis distances, confidence/OOD abstention and a
class-conditional conformal diagnostic were tested. OOD distance detected
domain shift but did not reliably rank individual prediction errors. Nominal
90% conformal coverage failed under camera shift. No trust mechanism solved
both camera directions at useful coverage. Decision: diagnostics only.

#### Paraguay external DR

The frozen RFMiD DINOv2-S DR output was evaluated without Paraguay fitting:

- 1,437 records; 726 any-DR positive and 711 negative.
- AUROC: 0.8876 [0.8686, 0.9049].
- AUPRC: 0.9141 [0.8980, 0.9278].
- Frozen-threshold sensitivity: 0.3044.
- Frozen-threshold specificity: 0.9958.
- ECE: 0.2402.
- Nominal 90% conformal coverage: 72.72%; diseased-eye coverage: 46.69%.

Interpretation: ranking transferred fairly well, but calibration and the fixed
decision threshold failed badly. Confidence-only abstention retained confident
false negatives. This supports representation transfer, not clinical or
universal-camera reliability.

### 10. What is genuinely complete

- Reproducible geometry and mask transformation pipeline.
- Duplicate-aware manifests and development split controls.
- Frozen DINOv2-S/RFMiD scorer and image-to-score interface.
- Locked 1/3/5-shot protocol with paired episodes and sanity controls.
- DINOv2-S, DINOv2-L, FLAIR and RETFound comparisons.
- PALM cross-camera development evaluation.
- Paraguay frozen external-site evaluation.
- Artifact stress tests and honest rejection of failed QC models.
- Rare-aware adapter training, paired evaluation and rejection.
- Integrated camera-module CLI and 32 passing code tests.

### 11. What is not solved

- Universal or unseen-camera generalisation.
- Verified camera invariance.
- Clinically validated image-quality rejection.
- Reliable cross-site calibration or conformal coverage.
- Strong K=1 rare-disease performance.
- A learned TRACE encoder that has beaten frozen DINOv2-S under the locked
  novel-label protocol.
- Patient-independent evaluation where patient identifiers are unavailable.
- Clinical safety/reader validation; no medical safety reviewer is available.
- Knowledge-graph construction or validated graph reranking in this workspace.
- A clean claim that FLAIR never saw the evaluation dataset.

### 12. Current claim that is supportable

TRACE currently supports this narrow claim:

> A camera-ID-agnostic geometric preprocessing and audit pipeline, combined
> with a frozen generic foundation encoder, can support reproducible few-shot
> retinal experiments and diagnose substantial acquisition, calibration and
> artifact failures. The experiments do not establish universal camera
> normalisation, camera-invariant representations or clinical safety.

### 13. My proposed next steps for you to judge

#### Step 0 — freeze and version the evidence

Create an immutable experiment snapshot, hashes and source-control repository.
The current workspace is not a Git repository. Keep the RFMiD protected test
closed until a final method and success gate are frozen.

#### Step 1 — decide between a research route and a delivery route

**Research route (recommended if enough time and GPU budget remain):** attempt
one materially different representation-learning experiment.  
**Delivery route (recommended if the deadline is close):** stop model search,
integrate the selected geometry + DINOv2-S + few-shot head + explicit warning
layer, and present the negative findings honestly.

Do not keep making small adapters against the same RFMiD validation episodes;
that would increase selection overfitting without adding independent evidence.

#### Step 2 — strongest proposed research experiment

Build a **multi-domain retinal self-supervised, pathology-preserving
parameter-efficient adaptation of DINOv2-S**, not another base-disease
supervised adapter.

Proposed design:

1. Acquire a separate legal training pool from retinal datasets that will not
   be used as final evaluations (candidate sources: EyePACS, APTOS, ODIR and
   JSIEC, after licence and duplicate checks).
2. Keep PALM, Paraguay and FIVES out of pretraining if they remain external
   validation sets.
3. Start from official DINOv2-S. Train only small LoRA blocks or a limited
   upper-layer adapter to fit laptop-GPU compute.
4. Use two mild, clinically bounded views of the same image. Avoid transforms
   that erase colour-dependent lesions or crop away peripheral disease.
5. Optimise a combination of:
   - teacher-preservation/distillation to prevent catastrophic drift;
   - global self-supervised view consistency or contrastive learning;
   - patch/local consistency so small lesions are not averaged away;
   - bounded artifact-pair consistency for glare/resolution/FOV stresses known
     to hurt the frozen model.
6. Do not use novel RFMiD labels in training or hyperparameter selection.
7. Predeclare one primary configuration or use training-only nested selection;
   avoid a large new sweep on the already reused validation set.
8. Evaluate on the already locked episodes against DINOv2-S, FLAIR, RETFound
   and the rejected adapter.

Why this may be better than Experiment 5A: it learns from diverse retinal
images and image structure instead of optimising ten base disease labels, so it
has a more plausible path to unseen-label transfer. This is still only a
hypothesis and may fail.

Suggested selection gate:

- Primary: abundant-negative novel-label macro-AUPRC improvement over frozen
  DINOv2-S at K=3 or K=5, with paired 95% CI above zero.
- No material regression at the other primary K.
- Preserve clean RFMiD disease ranking/sensitivity within a predeclared
  non-inferiority margin.
- Improve or at least not worsen paired artifact stability.
- Do not select based on PALM/Paraguay after viewing their final results.

#### Step 3 — obtain genuinely fresh evidence

The greatest weakness is not architecture; it is reused validation and limited
independent rare-disease evidence. Prefer obtaining an external dataset with
rare conditions and documented device/site metadata. If unavailable, create a
new untouched validation partition before training the next method and reserve
RFMiD test for one final locked evaluation.

#### Step 4 — final test only once

After the method, preprocessing, heads, thresholds, abstention rules and report
templates are frozen, run the duplicate-audited RFMiD protected test once.
Report class support, per-class AUPRC/AUROC, K-specific paired effects,
calibration and failure cases. Do not hide unsupported or zero-support labels.

#### Step 5 — final product integration

The demo should show:

- original image preserved;
- geometric standardisation and mask;
- deterministic quality measurements/warnings;
- requested disease ranking or few-shot score;
- support examples used;
- confidence/calibration warning and abstention when appropriate;
- provenance and an audit record;
- prominent research-only disclaimer.

The optional knowledge graph should remain separate and small. If built, it
should only rerank or explain disease scores using curated, source-located
assertions. It must not invent findings that the encoder does not output, and
LLM-generated medical edges must not be treated as evidence.

### 14. Questions you must answer

1. Is the proposed self-supervised parameter-efficient adaptation the strongest
   next experiment, or is another direction better given the evidence?
2. What exact loss, augmentations, trainable layers and sampling scheme would
   you use, and why should they transfer to unseen rare labels?
3. Which parts are already established in published research, and what—if
   anything—could be a defensible TRACE contribution?
4. How should we control for repeated use of RFMiD validation and possible
   pretraining overlap?
5. Which datasets should be training-only, development-only and untouched
   final tests? Check licences and known overlap.
6. Is the proposed success gate statistically adequate? Improve it if needed.
7. With one RTX 4060 laptop GPU, what is the minimum credible experiment and
   expected compute time?
8. If the deadline is short, what should be cut entirely?
9. What is the strongest honest final claim if no learned method beats frozen
   DINOv2-S?
10. Give a numbered next-action plan, ordered by scientific value and
    dependency. Mark each item must-do, optional or reject.

Conclude with one of these recommendations and justify it:

- proceed with the proposed retinal self-supervised adaptation;
- replace it with a better specified experiment;
- stop training and integrate the current evidence;
- redesign the evaluation before any further model work.

Do not recommend a complicated combination merely because it sounds novel.
Prefer the smallest experiment capable of answering the central question.

