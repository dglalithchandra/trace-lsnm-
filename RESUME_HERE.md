# TRACE resume point — 2026-11-11

## Camera module finished

The evidence-supported camera/preprocessing component is complete and frozen as
`trace-camera-module-v1.0-research`. See `CAMERA_MODULE_COMPLETE.md` and
`release/trace_camera_module_v1/`. Do not reopen rejected filters or QC models
without genuinely new evidence.

## Completed and locked

- Experiment 4A camera-artifact stress benchmark is complete.
- Report: `outputs/rfmid_artifact_stress_v2/EXPERIMENT_4A_REPORT.md`.
- Glare, resolution loss and narrow field of view all caused established paired
  macro-AUPRC losses under the frozen DINOv2-S/RFMiD pipeline.
- Current QC missed most harmful glare and resolution-loss cases.
- DINOv2-S and the disease head remain frozen; photometric harmonisation and the
  old shift adapter remain disabled.
- Full suite before Experiment 4B changes: 30 tests passed.

## Experiment 4B completed

- Full report: `outputs/artifact_qc_v1/EXPERIMENT_4B_REPORT.md`.
- The combined module is `not_selected` because clean acceptance was only
  15.49%, driven by the glare detector.
- Glare is rejected. Resolution-loss and field-of-view detectors are frozen
  exploratory candidates requiring fresh real validation.
- Complete train and validation QC feature files now exist under
  `data/features/`.

## Previous interruption is resolved

- Checkpoint/resume support was added to the extractor.
- Train and validation extraction both completed.
- No external or protected-test records were used in Experiment 4B.

## Experiment 5A completed

The rare-aware residual adapter experiment is complete. It trained only on ten
base labels after excluding every novel-positive training record. It improved
the internal base-label development score, but lost novel-label macro AUPRC in
all six 1/3/5-shot comparisons. A 5,000-replicate paired hierarchical bootstrap
found no evidence of improvement. The adapter is `not_selected`, its checkpoint
is retained only for audit, and frozen DINOv2-S remains primary. Report:
`outputs/trace_rare_adapter_fewshot_v1/EXPERIMENT_5A_REPORT.md`.

Protected-test records accessed across Experiment 5A: **0**.

## Next valid work

RFMiD 2.0 official archive is downloaded and checksum-verified. A development-only audit found 38 shared abbreviation candidates, a nonbreaking-space `ON` header, and an unnamed positive CSV column. Four local-feature overlap alarms were visually rejected as false positives, so this screen has zero verified cross-release duplicates. Source-paper review provisionally aligned `CRS`, `MS`, and `RT`; `EDN` is excluded because the v2 paper calls it `EX` while the supplied CSV uses `EDN`. The cross-release development comparison is complete for DINOv2-S, DINOv2-L, and RETFound-DINOv2. It found label-dependent results with only 6–8 positives per label, so no general encoder winner is selected. See `data/RFMiD2_DEVELOPMENT_AUDIT.md` and `data/protocols/trace_bench_v2/DEVELOPMENT_RESULTS.md`. The v2 test archive remains sealed.

Do not deploy or further tune the rejected adapter on the same novel validation
episodes. The next encoder experiment must be a separately specified method
with a locked success gate, or the project can proceed to final integration and
presentation using frozen DINOv2-S. Camera/QC limitations remain unchanged:
photometric harmonisation and all learned QC filters are disabled, and stronger
QC claims require fresh expert-labelled real quality data.

## EDID external cohort preparation

The official, unaugmented Eye Disease Image Dataset ZIP is downloaded and SHA-256 verified. Its 5,335 original images contain extensive exact copying across class folders: 464 cross-class groups, including 201 involving the `Healthy` folder. After exact-copy and near-copy review, the revised exploratory manifest retains 4,281 fundus files. Target/control counts are macular scar 326, retinitis pigmentosa 124, disc edema 106, and healthy-folder 777. All 76 local-geometry matches were reviewed side by side and logged. A priority check of 75 EDID-to-RFMiD-v1-training candidate pairs found no local-geometry matches, but is not an exhaustive overlap guarantee. See `data/EDID_SOURCE.md`, `data/manifests/edid_exploratory_manifest_v2.csv`, and `data/protocols/trace_bench_v2/EDID_EXPLORATORY_PLAN.md`. The deterministic RFMiD-v1-training support schedule is frozen at `data/protocols/trace_bench_v2/edid_support_schedule.csv`. Preprocessing accepted all 1,333 queries, including two with warnings; none were rejected. Frozen features and exploratory scores are complete for DINOv2-S, DINOv2-L, and RETFound-DINOv2; see `data/protocols/trace_bench_v2/EDID_EXPLORATORY_RESULTS.md`. No encoder wins all labels or support sizes. A post-hoc same-head check shows RETFound MS AUPRC also declines from K=1 to K=3/5 with KNN held fixed; the cause remains unresolved. Next, seek trustworthy disease/mimic controls before stronger claims. No protected RFMiD test images were opened. BRSET is not available without PhysioNet credentialed access.

## External mimic-source feasibility

JSIEC-1000 was audited as a candidate disease-control cohort. It has 22 RP, eight Bietti and 37 normal distinct images after exact deduplication. The RP/Bietti comparison is relevant but too small and lacks public patient IDs; it can only be descriptive. See `data/protocols/trace_bench_v2/EXTERNAL_MIMIC_FEASIBILITY.md`. Continue with the encoder's controlled label-hidden contrastive experiment using accessible RFMiD v1 training images. Keep EDID and JSIEC out of that SSL training pool if they remain external evaluations.

## TRACE SSL v1 starting point

The label-hidden 1,917-image RFMiD-v1-training manifest is frozen at `data/protocols/trace_ssl_v1/unlabeled_train_manifest.csv`. A 60-image view preflight measured nearly complete retinal-mask retention for the mild full-field view versus 95.8% mean retention for the plain random crop. The corrected matched plain-crop and full-field MoCo/InfoNCE pilots each completed five epochs and 2,395 updates. An earlier crop run used inconsistent colour gain in epoch one and was excluded. The corrected K=3 development macro AUPRCs were 0.2164 and 0.2189; their paired difference was small and uncertain. The best frozen comparator, DINOv2-L, scored 0.2044 on the same fixed-head episodes; neither adapted model reliably beat it. No adapted encoder is selected. See `data/protocols/trace_ssl_v1/FIVE_EPOCH_PILOT.md`. No protected test or EDID/JSIEC image entered SSL training.

The two adapted encoders were also evaluated on the existing 177-image RFMiD 2.0 **validation** split against three frozen encoders using identical K=3 support draws and head. On provisional mapped labels CRS/MS/RT (only 7/6/8 positives), neither adaptation showed a clear cross-release gain. See `data/protocols/trace_ssl_v1/RFMID2_CROSS_RELEASE_DEV.md`. No RFMiD v1 or v2 protected-test member was opened. Next, lock a single final comparison and audit the sealed test archive before a one-time test run; stop retuning on development episodes.

## Final evaluation plan

`data/protocols/trace_final_v1/ANALYSIS_PLAN.md` fixes the primary full-field SSL versus frozen DINOv2-L comparison, K=3 support draws, the 11-label eligibility rule, uncertainty method, secondary RFMiD 2.0 cross-release cohort, duplicate audit and stop conditions. `plan_lock.json` hashes the plan, training features, checkpoints, manifests, episode schedule and source archives without opening test members. Status: plan locked locally; test audit and final evaluation implementation pending. Do not change the plan in response to test labels or scores.

## Paused checkpoint — 12 November 2026

The user requested a pause during the **pre-scoring** test-input audit. `scripts/audit_trace_final_test_inputs.py` was interrupted with Ctrl-C after logging progress through 2,000 of 3,200 RFMiD v1 records. It had verified the locked archive/manifest hashes and begun decoding v1 images for exact/dHash overlap screening. The run had not reached RFMiD 2.0, had not created `outputs/trace_final_v1_audit/`, and had not extracted encoder features or scored any test image. There is **no completed audit result**. The process is stopped. On resumption, rerun the audit from the beginning with `--open-test-for-audit`; do not score until its candidate pairs receive local-detail and visual review, and final extraction/evaluation code is frozen. The v1 test images are no longer accurately described as never opened: some may have been read during this interrupted audit. No model selection was made from them.

