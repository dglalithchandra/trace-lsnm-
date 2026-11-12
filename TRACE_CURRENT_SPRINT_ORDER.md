# TRACE current sprint order

Decision date: 12 November 2026. This order supersedes the broad training, graph-reranking and multi-branch suggestions in earlier planning documents. It does not amend any locked analysis plan or reinterpret any completed experiment.

## Objective

Test whether label-hidden adaptation of DINOv2-S benefits from an audited multi-source pool and source-balanced sampling. Separately measure ranking, threshold, calibration and coverage transfer. Deliver an explanation graph that preserves model scores and rankings.

The operational baseline remains frozen DINOv2-S. The existing trainer uses rank-8 LoRA in the last four attention blocks, an EMA teacher, a projection head and a MoCo-style queue. Replacing this objective with DINO/iBOT is deferred to a separate study. No new training is authorised or started merely by creating this work order.

## Machine and Git ownership

| Owner | Work | Git rule |
|---|---|---|
| Lalith, RTX 4060 Laptop GPU / 8 GB | Source-pool admission, preprocessing cache, SSL training, reference feature extraction, protected review/release/scoring, integration and final demo | Only Lalith may update/push main; ordinary development can use codex/core-source-balance-v1 |
| Friend, Mac M3 Pro / 18 GB | Development benchmark, statistics, recalibration analysis, small explanation graph, tests and handoff | All commits and pushes go to the branch named exactly feature; no push or merge into main |

Short synthetic FP16 inference: PC 262.0 images/s; Mac 51.1 images/s (user-reported). These numbers do not establish training time or cross-device numerical agreement. Mac inference is optional after an identical-real-image check; reference model comparisons stay on one machine and implementation.

## File ownership and exchange contract

Friend-owned new paths:

- `trace_bench_v2/`
- `trace_explanations/`
- `data/registries/trace_dataset_registry_v2.json`
- `data/knowledge_sources/curated/trace_explanation_graph_v1.json`
- `data/contracts/trace_dev_exchange_v1.json`
- `scripts/run_trace_bench_v2.py`
- `scripts/run_trace_recalibration_dev.py`
- `scripts/build_trace_explanation_graph.py`
- `tests/test_trace_bench_v2.py`
- `tests/test_trace_explanations.py`
- `docs/TRACE_FEATURE_HANDOFF.md`

PC-owned new paths:

- `data/protocols/trace_source_balance_v1/`
- `data/manifests/trace_ssl_pool_v2.csv`
- New source-balance trainer/benchmark/extraction scripts; do not overwrite `scripts/train_trace_ssl_v1.py`.
- Local checkpoints, caches and feature exports under the already ignored directories.

The friend reads existing development code for reuse but avoids editing it. Cross-cutting changes should be proposed in the handoff for Lalith to integrate.

The first friend deliverable is the small exchange schema. It must describe model/checkpoint hash, preprocessing hash, embedding implementation/pooling/normalisation, precision, feature width, software version and aligned rows containing record ID, development split, dataset/source, optional verified patient/eye/group/device IDs, label ID/value, score or calibrated probability, and optional QC fields. Explicitly distinguish a ranking score from a probability. Arrays and real predictions are private exchanged artifacts, outside Git. Use synthetic rows for the checked-in example.

## PC work order

1. **Preserve a safe starting point.** Record commit and checkpoint/protocol hashes, check the working tree, create a core branch for new work, and keep the epoch-five and epoch-twenty artifacts intact. Do not merge unfinished feature work.
2. **Handle the old final experiment independently.** Lalith completes the 12 image-identity decisions in the local review page. Validate the review, freeze the release and run only the locked five-model extraction/evaluation. Uncertain decisions keep this gate closed; other development work can continue. Do not use these scores to select the new source-balance model. Do not add epoch-twenty or new candidates to the old test.
3. **Freeze the new study before screening.** Write the new protocol under trace_source_balance_v1. Set K=3 macro AUPRC as primary, K=5 as secondary, identical development queries/support/head budget, the paired uncertainty procedure, seed list, selection rule and a justified clean-performance margin. All currently reused cohorts remain development/exploratory. Reserve genuinely fresh confirmation if available.
4. **Admit data at record/group level.** Start from the 1,917 admitted RFMiD training records. Review additional legally available training-only images. All queries, test records, known related groups and near duplicates are excluded from SSL/support as appropriate. Never admit broad 'non-test' splits automatically. EyePACS/APTOS images are not currently present locally; do not count their metadata or hypothetical downloads as a ready pool. Paraguay, PALM, EDID and JSIEC stay evaluation/development-only. RFMiD 2.0 test remains blocked. IDRiD/FIVES additions require an explicit future role and overlap decision. If the expanded pool is not ready, stay core-only and record why A2/A3 are deferred.
5. **Cache preprocessing once.** Reuse frozen 448-pixel geometry and mask, QC measurements and correction-off setting. Create a versioned expanded image manifest with source labels, admission decisions and hashes. Keep disease labels out of the SSL dataloader.
6. **Measure the actual training cost.** Run a separate 50-step training benchmark, after warm-up, with the intended views, student/teacher, backward, optimizer and data loading. Record effective batch size, step-time distribution, peak allocated/reserved memory and numerical stability. Include feature extraction/checkpointing overhead in the run estimate. Freeze updates and precision after this check. Do not infer a training duration from 262 images/s inference.
7. **Export the initial baseline.** Reuse matching frozen DINOv2-S/L development features where hashes, pooling, precision and preprocessing agree. Export the minimal development feature/prediction pack using the agreed schema. A new backbone is optional and cannot delay this study. Mac runs statistics from the pack.
8. **Screen A0–A3.** A0=frozen S; A1=current SSL recipe on core training; A2=same recipe on expanded pool with natural sampling; A3=same expanded pool with source-balanced sampling. A1–A3 share initial weights, augmentations, optimizer, batch/accumulation, update count and evaluation. Use the same random seed for the initial screen. Implement new manifests/samplers/resumable seeds in separate scripts. Repeat support episodes identically across encoders.
9. **Select and replicate.** A3 minus A2 tests balancing; A2 minus A1 tests pool expansion at fixed updates; adapted minus A0 tests overall adaptation. A numerical development gain is a screening signal. Repeat the selected candidate and the control required for its claimed mechanism across three matched seeds if feasible. Sign consistency alone is not a superiority test. Include uncertainty and per-label harms. If only the overall adaptation is claimed, A0 is deterministic; if balancing is claimed, replicate both A2 and A3.
10. **Integrate trust analysis.** Give Mac frozen development/exploratory scores. Separate calibration samples from evaluation samples, using patient/group separation where available. Test budgets 0/5/10/20/50 only where feasible. Report class counts and failed one-class fits; never fit thresholds or calibration on the scored queries. Paraguay/PALM remain exploratory after repeated use.
11. **Integrate friend work and demo.** Review feature commits, run the relevant tests, merge accepted work from Lalith's account and push main. Connect QC warnings, few-shot scores, retrieved training examples and explanation graph through the agreed interface. Display cited general knowledge separately from measured case findings. Preserve scores/ranking through the graph.
12. **Freeze and deliver.** Record all results including losses, seeds, runtime, memory, dataset roles, provenance and contamination risk. Freeze the new model and analysis before any fresh confirmation. Without fresh confirmation, label new model gains development-only. Package reproducible commands and model/data cards.

## Minimum experiment set

| Arm | Weights/data | Purpose |
|---|---|---|
| A0 | Frozen DINOv2-S | Operational starting point |
| A1 | Core-pool SSL | Adaptation control |
| A2 | Expanded-pool SSL, natural sampling | Effect of the additional pool |
| A3 | Same expanded pool, source-balanced sampling | Effect of balancing |

Frozen DINOv2-L is a secondary comparator. Keep existing RETFound/FLAIR results as context with exposure tags. New DINO/iBOT, lesion-guided views, learned QC, feature fusion, graph reranking and a compulsory 45,000-image target are outside this sprint.

## Friend milestone order

1. Pull safely and work only on feature. Read this order and the verified report.
2. Deliver exchange schema, synthetic example and evidence-based registry.
3. Deliver deterministic development episodes, metrics and paired comparisons, with synthetic tests.
4. Deliver recalibration/threshold/risk-coverage analysis using synthetic data first, real development exports when supplied.
5. Deliver the explanation graph schema/validator/lookup, conservative source-backed mappings and unresolved curation queue.
6. Deliver retrieval/presentation outputs only after the above. No new GPU training or bulk acquisition.
7. Push feature and supply commits, test results, commands, unresolved issues and merge instructions; never merge into main.

## Decisions and limits

- K=1/3/5 are required. K=10/20 are conditional on support availability. K=0 applies only to an actual text/zero-shot method and is outside the primary sprint.
- More support draws do not add independent query images. Unknown patient grouping remains a limitation.
- A bootstrap must preserve model pairing and shared multilabel query dependence. Resample known patient/groups where available; otherwise identify the analysis as image-level.
- Brier/ECE require probabilities with known calibration status; ranking margins/cosine similarity are not automatically calibrated probabilities.
- Explanation mappings must retain exact/broad/narrow/related meaning, original source/version/locator, provenance and review state. Absent sources or unreviewed mappings are pending rather than invented.
- Git branch instructions are workflow rules. Enforcing owner-only main writes requires repository branch protection/rules with appropriate permissions; no protection change is implied by these documents.
- No changes to locked protected protocols, no score-based edits to test exclusions, no clinical/universal-camera claim.
