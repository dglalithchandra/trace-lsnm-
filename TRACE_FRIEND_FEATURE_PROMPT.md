# Paste this into the friend's Codex

```text
Implement the friend workstream for TRACE. Repository: https://github.com/dglalithchandra/trace-lsnm-.git

Current scope: build a small, reproducible development benchmark, recalibration analysis and an EXPLANATION-ONLY knowledge graph. Lalith owns model training, protected data, final scoring and integration. This instruction supersedes earlier suggestions to build graph reranking, train a new backbone, or use multiple friend branches.

GIT
Use exactly feature for every commit and push. Never commit, merge, rebase or push into main. Only Lalith updates main. No force pushes, destructive resets or discarding local changes. If an existing checkout is dirty, report its state and preserve it. Do not touch main even for local integration.

If needed, clone the repository and enter its directory. Run git status --short, git remote -v, git branch --show-current, then git fetch origin. If feature exists locally, switch to it and fast-forward from origin/feature if present. If only origin/feature exists, create its tracking branch. Otherwise create feature from origin/main. Incorporate origin/main into feature using a normal merge if needed. Stop and report unexpected conflicts instead of overwriting work. Verify branch=feature before each commit and push. Use git push -u origin feature, never push --all.

READ ONLY WHAT IS NEEDED
Read TRACE_CURRENT_SPRINT_ORDER.md and TRACE_VERIFIED_PROJECT_REPORT_2026-11-12.md first. If the first file is not on your clone yet, this prompt is the current scope. The broad TRACE_TWO_LAPTOP_EXECUTION_PLAN.md is historical where it conflicts with this prompt. Read existing trace_protocols/, trace_trust/metrics.py, their relevant tests, development protocol metadata, and data/knowledge_sources/SOURCE_INVENTORY.md selectively. Reuse existing functions before adding new ones.

PROTECTED BOUNDARY
You may read IMPLEMENTATION_STATUS.md and ANALYSIS_PLAN.md as policy context, but do not inspect row-level final-test labels/images, release decisions, final-test features/predictions or run protected scoring. Do not modify data/protocols/trace_final_v1/, data/manifests/trace_analysis_manifest.csv, final-test scripts, existing training scripts, frozen releases, original manifests or checkpoints. No human duplicate adjudication. Do not alter old episode schedules or prior results.

OWN THESE NEW PATHS
trace_bench_v2/
trace_explanations/
data/contracts/trace_dev_exchange_v1.json
data/registries/trace_dataset_registry_v2.json
data/knowledge_sources/curated/trace_explanation_graph_v1.json
scripts/run_trace_bench_v2.py
scripts/run_trace_recalibration_dev.py
scripts/build_trace_explanation_graph.py
tests/test_trace_bench_v2.py
tests/test_trace_explanations.py
docs/TRACE_FEATURE_HANDOFF.md

Prefer additive implementation in these paths. Propose changes to shared modules in the handoff rather than editing them concurrently with Lalith. Raw snapshots/data/weights/features are ignored and will not arrive through cloning. Do not fabricate them or bulk-download replacements.

MILESTONE 1: EXCHANGE CONTRACT AND REGISTRY
First produce a versioned exchange schema and a checked-in synthetic example. Real exports will be provided by Lalith outside Git. Record IDs, split, source/dataset, optional known patient/eye/group/device IDs, class order and labels must align exactly with embeddings/predictions. Include model/checkpoint hash, preprocessing hash, precision, software version, feature width, pooling/normalisation, score type and calibration status. Reject ambiguous row order, duplicate IDs, non-finite scores and misaligned labels.

Build a registry from repository-supported facts: dataset/version, source and licence reference, access availability, proposed role, verified metadata availability, overlap and pretraining-exposure risk. Missing facts stay unknown; proposed training admissions remain proposed until Lalith approves the record-level pool. EyePACS/APTOS are not ready locally. Preserve Paraguay, PALM, EDID and JSIEC as development/evaluation-only; RFMiD 2.0 test remains blocked.

MILESTONE 2: DEVELOPMENT BENCHMARK
Implement deterministic K=1/3/5 episodes; K=10/20 only when enough allowed support records exist. K=0 is not part of this image-only sprint. Preserve existing fixed episode files. Use explicit seeds, positive/negative policy, disjoint support/query records and known duplicate/patient groups. Report unsupported labels/K settings with a reason. Never claim patient independence when IDs are absent. Keep classifier hyperparameters fixed or select them on a distinct development-tuning partition, never on the evaluated queries.

Metrics: AUROC, average precision/AUPRC, prevalence, AP/prevalence (name this explicitly), sensitivity/specificity at a threshold fitted on a separate permitted partition, Brier score, documented ECE, and risk-coverage. Brier/ECE require genuine probability inputs and recorded calibration state; do not treat cosine scores or decision margins as probabilities. Undefined labels/metrics must retain failure reasons and sample counts. Provide per-episode/per-label and macro outputs with hashes/seeds.

Implement paired uncertainty preserving identical support/query selections across models. Account for labels, support draws and shared query resampling; preserve the same multilabel query/image resample across labels where possible. Resample known patient/groups if available, otherwise report image-level limitations. Repeated support draws are not independent query cohorts. Do not call an interval crossing zero equivalence.

MILESTONE 3: TRUST AND RECALIBRATION
Using synthetic inputs first, implement budgets 0/5/10/20/50 for calibration/threshold repair. Require a separate calibration set and evaluation set with known-group separation where available. Use repeated seeded draws, report class counts, and mark one-class/insufficient fits infeasible. Budget zero means no target fitting; use the frozen source rule. Keep ranking, threshold, calibration and selective-risk results separate. No threshold selection on evaluation labels. These externals are exploratory after prior use.

MILESTONE 4: EXPLANATION GRAPH
Build a small validated JSON graph and deterministic lookup API, not a reranker. Inputs: unchanged model label IDs/scores plus optional measured QC flags and provenance. Outputs: original ranking/scores, label mappings, definitions, cited general disease/finding relationships, source/version/locator, mapping/review state, missing knowledge and technical quality limitations. Test exact preservation of scores/order. Do not generate new confidence, diagnosis, findings or abstention decisions from the graph.

Target 20-40 label mappings only where available source records justify them. Use MONDO/ORPHA for diseases, HPO for findings, and exact/broad/narrow/related mappings with justification. Do not force disease/finding equivalence. Check available pinned-source hashes; if files are absent, build schema/tests and an unresolved curation queue and record the missing source. Small primary-source lookup is allowed to resolve a specific mapping. No LLM-authored medical edges or memory-only clinical assertions. Imported source relationships retain provenance and an accurate unreviewed/imported status. Do not claim clinician verification. General knowledge about a disease does not prove a finding is visible in the supplied photograph. Missing findings are unknown. QC measurements do not establish a medical causal relationship by themselves. Retain licences and do not copy restricted source text. JSON and standard-library traversal are sufficient; avoid Neo4j, servers and unnecessary dependencies.

MILESTONE 5: TESTS, HANDOFF AND PUSH
Create synthetic tests for deterministic episodes, support/query and group separation, insufficient data, score/label alignment, AP normalisation, separated threshold fitting, calibration fit failures, paired-identical zero difference, undefined metrics, graph provenance validation, distinct mapping predicates, unknown evidence, unchanged graph scores/order and protected-data rejection. Use independently computed small examples for numerical checks.

Run the existing baseline suite once if dependencies permit, record its result, then run focused tests while implementing and the full suite once at the end. Run compileall for changed modules and git diff --check. Provide a tiny synthetic smoke command that runs without images, weights, network or GPU and saves outputs under ignored outputs/. Do not silently skip failures. Install only missing small dependencies necessary for your task; do not attempt the Windows CUDA requirements on the Mac.

Commit at each stable milestone on feature, push feature, and give Lalith the hashes. Keep real predictions, images, feature arrays, raw ontologies, outputs, weights and credentials out of Git. Before every push inspect the intended file list and branch.

Handoff in docs/TRACE_FEATURE_HANDOFF.md: changed files, exact install/run/test commands and results, exchange schema, real inputs still needed, imported versus manually reviewed knowledge, scientific limits, remaining failures and how Lalith integrates it. Do not claim real evaluation, graph benefit or clinical validation from synthetic tests.

CREDIT AND TIME BUDGET
Do not repeatedly re-read all planning documents, re-run completed training, sweep models, or implement optional dashboards. Use rg, targeted reads, batched independent checks, existing functions and concise progress updates. Stop at a tested minimal deliverable. Prioritise contract -> benchmark -> recalibration -> explanation graph -> handoff. Continue with synthetic inputs while waiting for permitted development exports. If access is essential, report the exact missing file or permission instead of creating substitute evidence.
```
