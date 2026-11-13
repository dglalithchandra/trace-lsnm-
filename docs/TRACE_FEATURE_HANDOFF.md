# TRACE feature workstream handoff

Date: 12 November 2026

Branch: `feature`

Starting point: `397ff54`
Scope: development-only benchmark, recalibration analysis and explanation-only graph

## Delivered commits

| Commit | Milestone |
|---|---|
| `79308d4` | Development exchange contract, semantic validator, synthetic exchange and dataset registry |
| `78eb88e` | Deterministic episodes, metrics, macro summaries and paired hierarchical uncertainty |
| `55d6a05` | Separated calibration/evaluation and repeated label-budget analysis |
| `3b8d536` | Explanation-only graph validator, lookup, conservative graph and curation queue |
| `9ef23bb` | Synthetic smoke command, final tests and handoff |
| `60036ef` | Patient/duplicate connected support units, stricter exchange hashes and macro provenance |

The follow-up documentation commit only removes Markdown trailing whitespace and records `9ef23bb` in this table.

## Changed files and interfaces

- `data/contracts/trace_dev_exchange_v1.json` is the JSON Schema for private development feature/prediction packs.
- `trace_bench_v2/examples/trace_dev_exchange_v1.synthetic.json` is the checked-in no-data example.
- `trace_bench_v2.exchange.load_exchange` applies cross-field semantic checks that JSON Schema alone cannot express.
- `data/registries/trace_dataset_registry_v2.json` records evidence, availability, role, metadata, overlap and exposure risk. Proposed sources remain unapproved.
- `trace_bench_v2.episodes.build_episode_plan` builds deterministic K=1/3/5 episodes and conditional K=10/20 episodes. It retains infeasible settings and reasons.
- `trace_bench_v2.metrics.evaluate_binary` reports AUROC, average precision, prevalence, `ap_over_prevalence`, threshold metrics, Brier score, equal-width ECE and risk-coverage with explicit undefined reasons.
- `trace_bench_v2.paired.paired_bootstrap_difference` resamples labels, support draws and shared query groups or images while preserving model pairing.
- `trace_bench_v2.calibration.run_label_budget_analysis` runs budgets 0/5/10/20/50 with calibration/evaluation separation and repeated seeded draws.
- `trace_explanations.lookup_explanations` returns original model outputs unchanged, cited mappings and definitions when present, missing knowledge, measured QC values and technical limitations.
- `scripts/run_trace_bench_v2.py` is the no-image smoke entry point.
- `scripts/run_trace_recalibration_dev.py` accepts a JSON list of separated development records.
- `scripts/build_trace_explanation_graph.py` validates the graph and checks pinned source presence and hashes.

## Exchange required from Lalith

The real development pack must follow `trace_dev_exchange_v1` and remain outside Git. It needs:

1. Explicit zero-based `row_index` and unique `record_id` values.
2. Development split, dataset and source for every row.
3. Patient, eye, duplicate/group and device identity values with evidence status, including explicit unknowns.
4. One fixed `class_order`, aligned target label IDs/values and aligned output label IDs/values.
5. Model/checkpoint hash, code commit, precision and software versions.
6. Preprocessing hash and implementation, feature implementation, pooling, normalisation and width.
7. A private feature-array URI, SHA-256, dtype and exact shape.
8. Output type as `ranking_score` or `probability`, with the true calibration state.

The validator rejects duplicate records, ambiguous row order, class misalignment, non-finite values, invalid probability ranges and protected/non-development split names.

## Scientific safeguards

- The existing fixed episode files and all locked final-test paths are unchanged.
- K=0 is absent. K=10/20 are emitted only when positive and negative support units are sufficient.
- Support uses one record per known patient/group unit. Known support/query group overlap is an error. Patient independence is not claimed when IDs are incomplete.
- Repeated support draws are resampled as support variation and are not described as independent query cohorts.
- Average precision is named explicitly. `ap_over_prevalence` is separate and never substituted for AP.
- Brier score and ECE are defined only for recorded probability inputs.
- Threshold fitting uses calibration labels only. Evaluation labels are never passed to fitting code.
- Budget zero performs no target fit and uses the frozen source threshold/calibration rule.
- An interval containing zero is described as inconclusive, never equivalent.
- The explanation layer returns the exact input order and score values. It creates no confidence, diagnosis, case finding, causal QC relationship or abstention decision.

## Knowledge status

The checked-in graph contains 21 TRACE label nodes and a 21-item unresolved curation queue. It contains zero asserted mappings and zero medical relationships. The pinned MONDO, HPO and Orphadata raw snapshots are ignored by Git and absent from this clone, so their expected hashes are recorded and every candidate remains unresolved. No imported assertion is represented as manually reviewed, and no clinician verification is claimed.

To curate mappings later, Lalith should place the pinned files at their documented ignored paths, run the graph source/hash check, then add only source-located mappings with `exact`, `broad`, `narrow` or `related` predicates and an accurate review state.

## Reproduction commands

From the repository root:

```bash
/Users/lalith/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install pytest==9.2.0
.venv/bin/python -m pip install scikit-learn==1.9.1 opencv-python-headless==5.0.0.93 scikit-image==0.26.0
PYTHONPATH=. .venv/bin/python scripts/run_trace_bench_v2.py --synthetic
PYTHONPATH=. .venv/bin/python scripts/build_trace_explanation_graph.py
.venv/bin/python -m pytest -q tests/test_trace_bench_v2.py tests/test_trace_explanations.py
.venv/bin/python -m compileall -q trace_bench_v2 trace_explanations scripts/run_trace_bench_v2.py scripts/run_trace_recalibration_dev.py scripts/build_trace_explanation_graph.py tests/test_trace_bench_v2.py tests/test_trace_explanations.py
```

The smoke output is written under ignored `outputs/trace_bench_v2/`.

## Verification results

- Focused suite: `29 passed in 0.11s`.
- Complete applicable suite: `57 passed, 4 deselected in 1.82s`.
- Unfiltered complete suite: `57 passed, 4 failed in 1.94s`. The four failures are `test_palm_loader_uses_effective_split_and_palm_masks`, `test_manifest_loader_blocks_test_and_streams_training_image`, `test_manifest_loader_reads_idrid_annotation_masks` and `test_fives_loader_reads_image_and_vessel_mask`. Each fails with `FileNotFoundError` because the ignored PALM, IDRiD or FIVES image archive is absent from this clone. No failure was silently skipped or changed in source.
- Synthetic smoke: `PASS synthetic TRACE-Bench v2 smoke: 12 episodes, 10 recalibration rows`.
- Graph validation: `PASS_GRAPH_SCHEMA`, 21 unresolved curation items, zero asserted mappings, zero knowledge edges, and all four pinned source snapshots reported absent.
- `compileall` over every new package, script and test: pass with no output.
- `git diff --check`: pass with no output.

The exact applicable-suite command was:

```bash
PYTHONPATH=/private/tmp/eye-dinov2-bench/lib/python3.12/site-packages:. .venv/bin/python -m pytest -q -k 'not test_palm_loader_uses_effective_split_and_palm_masks and not test_manifest_loader_blocks_test_and_streams_training_image and not test_manifest_loader_reads_idrid_annotation_masks and not test_fives_loader_reads_image_and_vessel_mask'
```

The additional `PYTHONPATH` entry reuses the temporary PyTorch installation already created for the M3 inference benchmark. It does not add weights, checkpoints or generated outputs to Git.

## Remaining issues and integration

1. Real development features and predictions have not been supplied, so every numerical result produced here is synthetic.
2. Raw ontology snapshots are absent, so mappings and general medical relationships remain unresolved rather than fabricated.
3. The tested work is published only to `origin/feature`; `main` was not modified. Lalith should review the registry and real exchange, run the reproduction commands, then integrate accepted `feature` commits. Only Lalith may merge or push `main`.
4. The Git Credential Manager authorization used for the feature push is account-level persistent access and should be reviewed or revoked in GitHub settings when it is no longer needed.
5. No graph benefit, real evaluation result, clinical validation or patient-independent result is claimed.
