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

The official, unaugmented Eye Disease Image Dataset ZIP is downloaded and SHA-256 verified. Its 5,335 original images contain extensive exact copying across class folders: 464 cross-class groups, including 201 involving the `Healthy` folder. The conservative manifest excludes those groups, within-class extra copies, and anterior-segment Pterygium, leaving 4,370 distinct fundus image files for exploratory use. See `data/EDID_SOURCE.md` and `data/protocols/trace_bench_v2/EDID_EXPLORATORY_PLAN.md`. Next, screen near duplicates and RFMiD support/query overlap, then freeze the query set before any EDID model evaluation. BRSET is not available without PhysioNet credentialed access.

