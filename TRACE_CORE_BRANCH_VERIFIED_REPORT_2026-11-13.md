# TRACE core branch — verified report

Date: 13 November 2026

Branch: `codex/core-source-balance-v1`

Verified evidence head before this report commit: `2157a4b`
Remote: `origin/codex/core-source-balance-v1`

This report distinguishes completed evidence from remaining work. The branch has not been merged into `main`; only Lalith should make that integration decision.

## Completed model study

The A0–A3 study used fixed RFMiD-v1 development queries, the same support episodes and head, and a fixed 9,580-update budget.

- **A0:** frozen DINOv2-S baseline.
- **A1:** label-hidden retinal adaptation on 1,917 RFMiD-v1 training images.
- **A2:** the same method and updates after adding 595 audited FIVES training images under natural sampling.
- **A3:** the A2 pool with exactly balanced RFMiD/FIVES source draws.

A1 was repeated at seeds 20261111, 20261112 and 20261113. Mean A1-minus-A0 macro-AUPRC was **+0.0341 at K=3** and **+0.0474 at K=5**. All three seeds cleared the +0.02 screening gate at both K values. The result was heterogeneous: EDN worsened in all seeds; CSR and CRVO improved strongly.

A2 added only +0.0040 at K=3 and +0.0052 at K=5 over matched A1. A3 added only +0.0024 and +0.0017 over A2. Their intervals crossed zero and both failed the fixed +0.02 gates, so they were stopped instead of replicated. **A1 is the selected development arm.**

All A1–A3 training and evaluation runs recorded zero protected-test access.

## Cross-release development check

All three A1 seeds were evaluated on the already-used 177-image RFMiD 2.0 validation cohort with RFMiD-v1 support. The three provisional mapped labels have only 7, 6 and 8 positive queries.

| Model | Macro AUPRC | Macro AUROC | AUPRC delta vs frozen S |
|---|---:|---:|---:|
| Frozen DINOv2-S | 0.2689 | 0.6486 | — |
| A1 seed 20261111 | 0.2758 | 0.6952 | +0.0069 |
| A1 seed 20261112 | 0.2911 | 0.7184 | +0.0222 |
| A1 seed 20261113 | 0.2792 | 0.7068 | +0.0102 |

All macro point differences were positive, but most per-label intervals crossed zero. This is encouraging, underpowered development evidence—not fresh confirmation or a camera-generalization result.

## Usable research package

The branch now provides:

1. `TraceA1Encoder`: strict image-to-384-value-embedding interface with frozen geometry-only preprocessing and artifact/configuration verification.
2. `TraceFewShotHead`: the locked balanced `C=1` logistic support-to-ranking-margin method. Its output is not called a probability.
3. `TraceRetrievalIndex`: provenance-bearing cosine retrieval that does not alter the model score.
4. `scripts/run_trace_fewshot_images.py`: one-command positive-support, negative-support and query-folder workflow with exact-duplicate role checks, QC audit, scoring and retrieval.
5. `scripts/generate_trace_a1_embedding_map.py`: low-support-label and reported-camera-domain projection. Windows blocked Numba, so the local output is explicitly PCA fallback; the script produces deterministic UMAP where that runtime is allowed.
6. Model, deployment, inference, embedding-map and cross-release reports under `data/protocols/trace_source_balance_v1/`.

The end-to-end folder smoke test used two positive supports, two negative supports and two queries. Both queries were scored, none was rejected, and four retrieved supports per query were recorded.

## Compact artifact and efficiency

The full A1 training checkpoint is 183,049,039 bytes. The verified LoRA-only adapter contains 73,728 learned parameters and is 300,979 bytes: **608× smaller**. On a real-image equivalence check, the compact and full artifacts produced exactly equal float32 embeddings (`max_abs_difference = 0.0`).

On the RTX 4060 laptop GPU, FP16 encoder-only performance was:

| Model | Batch 1 median | Batch 16 throughput | Batch 16 peak allocated |
|---|---:|---:|---:|
| Frozen DINOv2-S | 8.23 ms | 225.5 img/s | 337.85 MB |
| TRACE A1 | 8.55 ms | 213.3 img/s | 338.34 MB |

This excludes disk I/O and preprocessing. TRACE A1 added about 0.46 MiB peak allocation and reduced batch-16 throughput by 5.4%.

## Verification status

- Test suite: **56 passed** at report creation.
- Environment verifier: **failed for two Windows-policy-blocked optional extension DLLs**—Albumentations/StringZilla and UMAP/Numba. The tested core TRACE pipeline still passed; the embedding-map script records an explicit PCA fallback. Do not describe the full environment check as passing.
- Working tree: clean before this report was added.
- Adapter, checkpoints, feature arrays and generated outputs are Git-ignored.
- The remote has no `feature` branch yet; no friend work has been integrated.
- `main` remains at `397ff54`; this branch is not merged.

## Remaining blockers and next actions

1. **Protected final comparison remains blocked.** Twelve image-identity decisions and the existing release gate must be resolved by a human. Do not manufacture these decisions or score the protected test early.
2. **Fresh confirmation is absent.** The A1 claim remains development-only until a genuinely protected organizer/hospital cohort is run once under a frozen analysis.
3. **Friend branch is absent.** Recalibration/threshold analysis and the explanation-only graph were assigned to branch `feature`, but that branch has not been pushed. Fetch and review it before duplicating those owned paths.
4. **True UMAP is pending on a permitted runtime.** Run the checked-in embedding-map script on the Mac; do not present the Windows PCA fallback as UMAP.
5. **Explanation graph must not rerank scores.** If delivered, it should expand verified label names and cite curated sources while preserving the model's margin and ordering. No unvalidated finding heads exist.
6. **Final integration and presentation remain.** After review, merge accepted commits, add the optional explanation/recalibration outputs, run the full suite, freeze hashes and prepare the research-only demo/report.

## Commands

```powershell
# Image to selected A1 embedding
.\.venv\Scripts\python.exe scripts\embed_trace_a1_image.py --image C:\path\fundus.jpg --output embedding.npy

# Complete few-shot image-folder workflow
.\.venv\Scripts\python.exe scripts\run_trace_fewshot_images.py --positive-dir support_positive --negative-dir support_negative --query-dir queries --output-dir trace_run

# Compact adapter export
.\.venv\Scripts\python.exe scripts\export_trace_a1_adapter.py

# Reference inference benchmark
.\.venv\Scripts\python.exe scripts\benchmark_trace_a1_inference.py

# UMAP where permitted; otherwise explicitly labelled PCA fallback
.\.venv\Scripts\python.exe scripts\generate_trace_a1_embedding_map.py

# Tests
.\.venv\Scripts\python.exe -m pytest -q
```

## Claims that are not supported

Do not claim universal camera normalization, unseen-camera generalization, clinical diagnostic safety, calibrated disease certainty, patient-independent performance, a proven knowledge-graph benefit, or fresh external confirmation. The strongest supported claim is narrower: under the leakage-audited RFMiD development protocol, three A1 seeds improved few-shot macro AUPRC over the same frozen initialization, with small inference overhead; a tiny, previously used cross-release validation check had positive macro point differences in all three seeds but remained inconclusive.
