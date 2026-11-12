# TRACE completion order on this Windows PC

Decision: use the existing RTX 4060 Windows environment; the Mac and AnyDesk are not needed for the remaining local work. Status checked 12 November 2026.

## Already working

- Camera/preprocessing module v1: 448-pixel geometry, mask, quality diagnostics, frozen disease-scoring guardrail. Photometric harmonisation and learned QC are disabled after failing their gates.
- Frozen DINOv2-S/L and RETFound-DINOv2 baselines, fixed RFMiD few-shot protocol, development retrieval and exploratory external checks.
- Two matched 20-epoch label-hidden SSL students. Development K=3 macro AUPRC: crop 0.2363; full-field 0.2380. Their difference is uncertain. Original five-epoch checkpoint hashes are unchanged.
- Locked five-model RFMiD-v1 final analysis and runnable scoring scripts; the development-only preflight passes.

## Required, in order

1. **Human image-identity review.** Review all 12 pairs in `data/protocols/trace_final_v1/REVIEW_IMAGES.html` and record duplicate/distinct with reviewer name. Uncertain pairs remain a gate, not guessed decisions. This does not require disease diagnosis.
2. **Run the current locked comparison once.** Validate the review CSV, freeze the scoring release, extract only the five specified test feature sets and run the fixed analysis. Report all labels and uncertainty, regardless of outcome. Do not insert epoch-20 models into this locked comparison.
3. **Create a new evaluation gate for an improved model.** Resolve the RFMiD 2.0 test image/label ID mismatch and exact overlaps through an authoritative mapping, or acquire an independent low-support retinal cohort. If neither is possible, use the organizer's authorized hidden evaluation if available and describe local epoch-20 findings as development-only.
4. **Choose the strongest eligible baseline on development.** Existing DINOv2-L and RETFound are ready; add another foundation model only if access, usage terms, input protocol and likely pretraining overlap can be verified. Keep support/query IDs and head budget identical.
5. **Improve the encoder only with a clean control.** The best current signal is longer SSL training, not full-field views. Audit and deduplicate a suitable, licensed unlabeled pool before adding images. Compare RFMiD-only versus pooled data at matched updates and source-balanced sampling. Repeat finalists across at least three seeds if time and a fresh test permit. Add a patch-level objective only after a lesion-scale failure is measured on development annotations.
6. **Complete trust and presentation.** Evaluate calibration/threshold transfer, quality/OOD warnings and risk versus coverage; document few-shot K=1/3/5, per-label counts, retrieval examples, UMAP, ablations, latency/memory and a model card. Ship one callable image-to-embedding and support-to-score interface plus a small research demo.

The knowledge graph is a separate optional study, not a prerequisite for TRACE's encoder claim. No unsupported universal-camera or clinical-diagnosis claim should be made.

## Current blockers

- `audit_decisions.csv` and `scoring_release.json` do not exist because the 12-image human review is unfinished.
- RFMiD 2.0 test labels have mismatched IDs and three exact overlaps with its development images; secondary test scoring remains blocked.
- No fresh independent rare-condition cohort with verified patient/device metadata is ready.
- The 20-epoch students have one seed and reused validation queries; they have no confirmatory result.
