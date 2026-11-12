# TRACE two-laptop execution plan

Status date: 2026-11-12

## 1. The project in one sentence

TRACE is a research system for learning useful retinal fundus-image representations from mostly unlabelled data, then recognising rare or low-data diseases with a few labelled examples while remaining reliable across image quality, site and camera changes.

The intended final claim is narrower than "diagnosis from any camera": a TRACE representation and inference protocol improves few-shot rare-disease ranking or is statistically equivalent to the strongest frozen foundation-model baseline, with explicit calibration, abstention and cross-source evaluation.

## 2. Complete TRACE component map

| Component | Purpose | Current state | Required final output |
|---|---|---|---|
| Research contract | Define the exact question, endpoints and permitted claims | Partial | Frozen protocol, primary endpoint and smallest useful effect |
| Data registry | Record dataset, licence, source, labels, patient/device fields and intended role | Partial | Versioned registry and immutable manifests |
| Leakage control | Detect duplicates, patient overlap and public-pretraining contamination risk | Partial | Exact and near-duplicate ledger with review decisions |
| Camera and quality preprocessing | Standardise geometry and expose blur, noise, glare, compression and missing retina | V1 complete with limited claim | Frozen module, QC report and ablation against raw images |
| Foundation baselines | Establish the strongest model that TRACE must beat | Partial | Same-protocol results for general and retinal foundation encoders |
| TRACE contrastive encoder | Learn retinal features from unlabelled fundus photographs | Development models exist; confirmatory training absent | Pre-registered encoder comparison with seeds and ablations |
| Few-shot inference | Adapt to a new disease using 1, 3, 5, 10 or 20 examples | Partial | Frozen heads/retrieval rules and full learning curves |
| Cross-source/device evaluation | Test support examples from one source and queries from another | Missing or blocked by metadata | Source-shift episodes and cautious device claims where metadata exists |
| Statistical evaluation | Quantify uncertainty across labels, episodes and images | Partial | AUROC, normalised AUPRC, sensitivity at fixed specificity, confidence intervals and paired tests |
| Calibration and abstention | Convert rankings to usable confidence and refuse uncertain cases | Partial | Calibration error, Brier score, risk-coverage curve and OOD policy |
| Retrieval and explainability | Show similar support images and where evidence occurs | Development retrieval exists | Leakage-safe retrieval, patch evidence and qualitative audit |
| Efficiency | Measure compute and deployment cost | Missing final measurements | Parameters, latency, memory, training time and energy/compute record |
| Demo and model card | Make the research understandable and reproducible | Partial | Research-only demo, model card, data card and audit trail |
| TRACE evidence graph | Curated disease/finding/artifact reasoning, reranking and explanations | Source snapshots present; graph and evaluation not built | Versioned graph, evidence-aware reranker, abstention, provenance display and controlled ablation |
| Vision-language extension | Optional CLIP-style text alignment | Not started | Run only with verified text and image-only/scrambled-text controls |
| OCT work | Different imaging modality | Out of current scope | Separate future study; do not mix with colour fundus claims |

## 3. Honest current position

Completed or usable:

- A frozen camera/preprocessing V1: retinal field detection, crop, centre, resize, circular mask and quality measurements.
- Photometric harmonisation is disabled in the frozen release because the evidence did not justify forcing colour changes.
- Frozen DINOv2-S, DINOv2-L and RETFound feature paths, fixed few-shot code, retrieval tooling and several external exploratory datasets.
- Two matched 20-epoch SSL development students. Their K=3 macro AUPRC difference was about +0.0017 for full-field views versus crop views, which is too small and uncertain to claim a win.
- A locked five-model RFMiD v1 scoring pipeline is prepared.

Not completed:

- The 12 human duplicate/identity decisions needed before final RFMiD v1 scoring.
- A confirmatory result on untouched data.
- A sufficiently powered multi-label, multi-episode benchmark with multiple seeds.
- A clean RFMiD 2.0 evaluation mapping; current image/label IDs do not line up, and overlap concerns remain.
- Reliable calibration, abstention and source/device shift results.
- A final encoder that beats the strongest frozen comparator.
- Final demo, model card, compute report and submission package.

## 4. Two-laptop ownership

### Laptop A: Lalith / primary repository owner

Own the critical path and protected evaluation:

1. Finish the 12 duplicate-review decisions.
2. Freeze and hash the final RFMiD v1 analysis manifest.
3. Run the locked five-model comparison once.
4. Select the primary baseline on development data only.
5. Train the next TRACE encoder experiments with multiple seeds.
6. Run final calibration, abstention and confirmatory scoring.
7. Integrate accepted branches and create the final report/demo.

Protected files and data must remain on Laptop A:

- `data/protocols/trace_final_v1/`
- `data/manifests/trace_analysis_manifest.csv`
- `scripts/extract_trace_final_v1_test_features.py`
- `scripts/evaluate_trace_final_v1.py`
- Any untouched test labels, scoring releases or final test predictions

### Laptop B: friend / independent research workstream

First assignment: build TRACE-Bench development infrastructure and trust evaluation without opening protected test data.

1. Audit external development datasets and create a dataset registry.
2. Standardise label mappings while preserving exact/broad/narrow relationships.
3. Add exact and near-duplicate audit reports against allowed development manifests.
4. Build repeated few-shot episode generation for K=0/1/3/5/10/20.
5. Add normalised AUPRC, AUROC, sensitivity at fixed specificity, calibration and risk-coverage metrics.
6. Add tests using synthetic fixtures or non-protected development samples.
7. Produce one reproducible command and a machine-readable result schema.

Second assignment, after the first merges: compare frozen baselines on development episodes only. Candidate models are DINOv2-L, DINOv3-L if weights and code are available, RETFound-DINOv2/MAE and one theme-relevant retinal vision-language model if it can be run legally and reproducibly. Every model must use identical images, preprocessing, episodes and heads.

Parallel independent assignment: build the TRACE evidence graph described in Section 5. It must remain independently testable so that graph failures cannot change the encoder score. The graph becomes part of the final TRACE system only after it passes its development ablation.

## 5. TRACE evidence graph: its exact job and design

The graph is useful when it answers a question that the image encoder cannot answer alone: "Given the model's candidate diseases, the few verified findings we can actually detect, and the image-quality limitations, which candidates are better supported, which are contradicted, and why?"

### Inputs

The graph receives only recorded evidence:

- Calibrated `disease_scores` from the image encoder/few-shot head. This is the primary input.
- `finding_scores` only for findings with a genuinely trained and evaluated head, such as disc pallor, disc oedema, exudation, media haze or a validated vessel finding. Missing finding heads remain unknown, not negative.
- Camera/source and quality-control observations such as blur, glare, compression, uneven illumination and missing retinal field.
- Optional patient context only when the dataset supplies it lawfully and consistently. Missing age, sex, symptoms or genetics remain unknown.

### Graph objects

- Disease: stable MONDO/ORPHA identifier and synonyms.
- Finding: HPO identifier plus `visible_on_fundus`, `visibility_status` and curation state.
- Artifact/quality condition: glare, blur, compression, uneven illumination and field loss.
- Dataset label: the exact local label, never silently treated as an ontology equivalent.
- Assertion: a first-class record joining subject, predicate and object with source, locator, evidence type, curator, review status, version and date.
- Mapping: exact, broad, narrow or related mapping with justification and confidence.

Core relations include `HAS_FINDING`, `EXCLUDES_FINDING`, `MAY_MIMIC`, `MAY_OBSCURE_FINDING`, `VISIBLE_ON_FUNDUS`, and dataset-to-ontology mappings. Each relation has one direction and a documented meaning.

### Evidence sources already prepared

The repository contains pinned MONDO Rare Disease, HPO, UBERON, Relation Ontology, ECO, Orphadata, HGNC, ClinGen and Biolink snapshots with checksums and licence notes. Ontology associations seed candidates; they do not prove that a finding is visible in a fundus photograph. Visibility and important differential-diagnosis assertions require manual literature/guideline curation with a precise locator.

### Practical implementation

Start small: about 10 target rare retinal diseases, 5 common mimics, and the findings that TRACE can measure or must preserve. Use TSV/CSV tables validated with Pydantic or Pandera, NetworkX for traversal and a small JSON API. Export SSSOM TSV for mappings and JSON-LD/Turtle for interoperability. Neo4j is unnecessary for this scale.

Recommended tables:

- `entities.tsv`
- `assertions.tsv`
- `dataset_mappings.sssom.tsv`
- `fundus_visibility.tsv`
- `sources.tsv`
- `curation_reviews.tsv`

### Reranking

The graph must not replace the encoder. It adjusts the encoder's calibrated disease logits using verified supporting and opposing evidence:

```text
graph_score(d) = calibrated_encoder_logit(d)
               + alpha * supported_findings(d)
               - beta  * excluded_findings(d)
               - gamma * unreliable_due_to_artifact(d)
```

HPO frequency bands become ordinal evidence weights, not invented probabilities. `alpha`, `beta`, `gamma`, minimum evidence and abstention thresholds are tuned on development data and frozen before the final test. Scores are corrected for annotation-density bias so well-studied diseases do not win simply because they have more edges.

If a relevant finding is hidden by glare or field loss, the graph lowers the reliability of that evidence rather than calling the finding absent. If graph evidence is missing or contradictory, it returns the encoder ranking plus an abstention/explanation flag.

### Evaluation gate

Compare all four on exactly the same frozen development cases:

1. Encoder alone.
2. Encoder plus class-prior/co-occurrence reranking.
3. Encoder plus plain HPO-overlap ranking.
4. Encoder plus the TRACE evidence graph.

Measure ranking AUPRC/AUROC, top-k recall, calibration, risk-coverage, performance by source/quality group and between-source performance gaps. The graph passes only if it improves a predeclared endpoint without violating a clean-ranking non-inferiority margin. Also run edge-removal, artifact-reasoning and frequency-weight ablations. Report graph coverage: the fraction of cases for which verified evidence was available.

The graph remains a proper part of the submitted system even if it fails the performance gate: its curated evidence and explanation view can be delivered, but the failed reranker must be disabled and reported honestly.

### Friend's graph branch

Use `codex/kg-trace-evidence-v1`. The friend owns source parsing, schema, curation interface/tables, validators, NetworkX graph, reranker, explanation payload and graph-specific tests. She must not edit encoder training or protected evaluation files. Lalith supplies frozen development predictions and permitted finding/QC fields through a versioned interface; final evaluation stays on Laptop A.

## 6. Branch and merge rules

Use a private Git repository. Never work directly on `main`.

- Lalith branch pattern: `codex/core-<task>`
- Friend branch pattern: `codex/bench-<task>` or `codex/trust-<task>`
- One branch should have one clear deliverable.
- Pull `main` before starting a branch.
- Commit source code, manifests without protected labels, tests, small configs and documentation.
- Do not commit datasets, model weights, feature arrays, predictions, credentials, access cookies or final test labels.
- Exchange large permitted artifacts through an agreed private storage location and verify SHA-256 hashes.
- Merge only after the owner reruns the documented command and checks the result schema.
- Never resolve a scientific conflict by choosing the better score. Fix the protocol first and rerun both methods.

Recommended daily flow:

```text
git switch main
git pull --ff-only
git switch -c codex/bench-trace-bench-v2
# work, test, document
git add <only intended files>
git commit -m "Build repeated few-shot development benchmark"
git push -u origin codex/bench-trace-bench-v2
```

After review, merge the branch on the repository host. Both laptops then run `git switch main` and `git pull --ff-only`.

## 7. Data sharing rules

Git transfers the code, not the research data. Each laptop should download allowed datasets from the official source when possible. A shared registry records:

- Dataset name and version
- Official URL and licence/terms
- Archive SHA-256
- Local extraction layout
- Patient, eye, camera and site metadata availability
- Label type and mapping confidence
- Role: SSL training, development, external evaluation or prohibited/protected
- Known or possible overlap with model pretraining and other datasets

No dataset should silently change roles. In particular, a dataset used to tune a method cannot later be described as independent confirmation.

## 8. Experiment gates

An experiment enters the final comparison only if all of these are true:

1. The images and labels are permitted for that role.
2. Duplicate and overlap checks have completed.
3. The episode file was frozen before model comparison.
4. Every method uses the same support/query examples and scoring head.
5. Hyperparameters were selected on development data.
6. Results include uncertainty, per-label behaviour and failure cases.
7. Camera claims use real camera metadata; dataset identity is described as source shift.
8. The protected test is read only by the final locked script.

## 9. Research priority order

1. Complete duplicate adjudication and lock the benchmark.
2. Establish the strongest same-protocol frozen baseline.
3. Fix statistical power: repeated episodes, more eligible labels and multiple seeds.
4. Test source/device-shift episodes and site recalibration.
5. Train retinal SSL variants only after the benchmark is trustworthy.
6. Add calibration, OOD detection, abstention and retrieval.
7. Run controlled ablations of preprocessing, objective terms and resolution.
8. Run one confirmatory comparison.
9. Build the demo, model/data cards and submission evidence.
10. Merge the evidence graph and run its four-way controlled ablation.
11. Add vision-language alignment only if image-only representation remains the bottleneck.

## 10. Definition of a strong final TRACE result

The strongest successful outcome is not simply a high number. It is a result showing that, under one locked and adequately powered protocol, TRACE improves rare-disease few-shot performance over the best frozen baseline, preserves clean-source ranking, remains useful under source/device shift, produces calibrated uncertainty or abstains, and can be reproduced from released code and manifests.

If the trained encoder does not win, a rigorous benchmark showing statistical equivalence, the exact shift failure, and the small local-label budget needed for recalibration is still a defensible research contribution.

## 11. First handoff prompt for the friend's Codex

```text
You are working on TRACE, a retinal fundus representation-learning research project. Work on two isolated branches, one at a time: codex/bench-trace-bench-v2 for benchmark infrastructure, then codex/kg-trace-evidence-v1 for the evidence graph. Do not read or modify protected final-test files, test labels, scoring releases, model weights or cached final-test features.

Your assignment is to build TRACE-Bench v2 for development data. First inspect the repository and its existing manifests, scripts and tests. Then implement:

1. A versioned dataset registry containing source, licence reference, hashes, label type, patient/eye/camera/site metadata availability, allowed role and contamination/overlap risk.
2. Deterministic repeated few-shot episodes for every eligible label and K in 0, 1, 3, 5, 10 and 20, with support/query separation and explicit random seeds.
3. Metrics: AUROC, AUPRC, normalised AUPRC, sensitivity at fixed specificity, Brier score, calibration error and risk-coverage curves. Return per-label, per-episode and aggregate records; never hide failed/undefined labels.
4. Bootstrap or hierarchical uncertainty suitable for paired model comparisons.
5. Exact and near-duplicate audit hooks that output candidate pairs for human review without automatically deleting scientific data.
6. Synthetic-fixture tests and one reproducible development-only command.
7. Documentation of assumptions, exclusions, output schema and remaining risks.

Do not optimise any method using protected data. Do not claim dataset identity is camera identity. Do not invent missing patient, camera or disease metadata. Preserve exact/broad/narrow label mappings. Make small commits, show test evidence, and finish with a handoff containing changed files, commands, outputs and unresolved issues.

After the benchmark branch is reviewed, build the TRACE evidence graph on codex/kg-trace-evidence-v1:

1. Read data/knowledge_sources/SOURCE_INVENTORY.md and verify checksums. Never modify raw snapshots.
2. Create validated tables for entities, assertions, sources, curation reviews, fundus visibility and SSSOM dataset mappings.
3. Start with about 10 rare retinal diseases, 5 mimics and only findings relevant to fundus images or available TRACE heads.
4. Treat every edge as an assertion with provenance, source locator, evidence code and review state. Do not allow an LLM to author medical edges.
5. Implement encoder-disease-score input first. Accept finding scores only through an explicit typed interface, and treat missing findings as unknown.
6. Implement quality-aware reasoning so glare, blur or missing field reduce evidence reliability rather than becoming negative clinical evidence.
7. Implement the development-frozen reranker and abstention/explanation payload.
8. Evaluate encoder-only, prior/co-occurrence, HPO overlap and evidence-graph reranking on identical development cases, including coverage and ablations.
9. Export SSSOM TSV plus JSON-LD or Turtle, and provide a small human-readable evidence report for each ranked disease.
```

## 12. Information required before the first push

The local repository currently has no remote configured, and the GitHub CLI is unavailable. To publish safely, the owner must provide the URL of an existing private Git repository and add the friend's GitHub account as a collaborator through the repository host. No password or access token should be pasted into Codex chat.
