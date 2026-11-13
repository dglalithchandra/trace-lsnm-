# TRACE feature integration report

Date: 13 November 2026

Target branch: `codex/core-source-balance-v1`

Reviewed source: `origin/feature` at `088362c`

Merge commit: `a2f2b11`

## Integrated work

The feature workstream added the development exchange contract and validator, deterministic TRACE-Bench v2 episodes and metrics, paired hierarchical uncertainty, separated label-budget recalibration, an explanation-only graph scaffold, dataset registry, synthetic examples, tests and handoff documentation.

No feature commit changed the locked protected-test protocol, A1 model implementation, camera release or existing core experiment files. The merge had no content conflicts.

## Review conclusions

- Calibration and evaluation partitions are enforced separately.
- Known group overlap between calibration and evaluation is rejected.
- Evaluation labels are not passed into threshold or calibration fitting.
- Ranking scores must be marked `not_probability`; probability-only metrics are undefined for ranking scores.
- Paired comparisons preserve model pairing and shared query/group resampling.
- Explanation lookup returns the original score values and ordering unchanged.
- The graph does not infer case findings, diagnoses, confidence, causality or abstention.

The knowledge graph is currently a **safe scaffold**, not a completed medical graph: 21 TRACE labels remain in the curation queue, with zero asserted mappings and zero knowledge edges. This must not be presented as a demonstrated graph benefit.

## Local verification

- Feature-focused tests: **29 passed**.
- Synthetic TRACE-Bench smoke: **passed**, producing 12 episodes and 10 recalibration rows.
- Explanation graph schema: **passed**.
- All four locally available pinned source snapshots—MONDO rare, HPO, Orphanet nomenclature and Orphanet phenotypes—matched their expected SHA-256 values.
- Feature packages and entry scripts compiled successfully.
- `git diff --check`: passed.

The three advertised scripts originally required `PYTHONPATH=.` on Windows. The integrated branch now inserts the repository root itself, so these work directly:

```powershell
.\.venv\Scripts\python.exe scripts\run_trace_bench_v2.py --synthetic
.\.venv\Scripts\python.exe scripts\build_trace_explanation_graph.py
.\.venv\Scripts\python.exe scripts\run_trace_recalibration_dev.py --help
```

## Full-suite environment interruption

Before this merge, the core suite passed 56/56. During the post-merge complete run, Windows Application Control began blocking PyTorch DLLs (`caffe2_nvrtc.dll`, `shm.dll` and `_C`) during test collection. Four PyTorch-importing modules were reported before collection stopped. This is an environment-policy failure, not a Python assertion failure or evidence that the merged feature code changed model behavior.

The 29 feature tests were rerun afterward and still passed. A complete combined PyTorch suite must be rerun after the Windows policy/runtime issue is cleared. Do not report the combined suite as passing yet.

## Next integration step

The merge belongs only to `codex/core-source-balance-v1`. It must not be merged into `main` until Lalith reviews this report, the PyTorch runtime is available again, the combined suite passes, and any real development exchange/recalibration input is frozen. The graph should remain explanation-only and disabled from scoring.
