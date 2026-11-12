# TRACE labelled-example retrieval — development check

Status: **descriptive development component; no final-test images accessed; not selected as evidence of cross-camera generalization**.

## What was configured

The index contains 1,917 RFMiD v1 training images represented by the frozen DINOv2-L 448-pixel TRACE-geometry embeddings. For each of 639 RFMiD v1 validation queries, the system returns five training examples ranked by cosine similarity. It excludes a candidate with the same known split-group ID and records each example's source ID, positive RFMiD labels, similarity and reported acquisition domain. The example labels are taken from the supplied annotations; the encoder does not discover or verify them.

The source files and full top-five output are recorded under `outputs/trace_retrieval_v1_dev/`. `scripts/build_trace_retrieval_dev.py` creates that result. The different-reported-domain diagnostic is under `outputs/trace_retrieval_cross_domain_dev/` and comes from `scripts/analyze_trace_retrieval_cross_domain_dev.py`.

## What the development data show

Across all 639 queries, **91.6%** of the unrestricted top-five examples share the query's reported domain. Random draws from the training pool would share it about **63.0%** of the time, given the validation domain mix. This is evidence of strong acquisition-domain clustering in the frozen embeddings under this metadata scheme. It is **not** proof that camera hardware alone caused the clustering: some RFMiD domain labels are inferred from image resolution, and disease mix, image quality and acquisition details may differ by domain.

The recorded-label overlap is highly uneven across the eleven fixed low-support labels. On positive validation queries, unrestricted top-five hit rates include RT 6/6, RS 12/14, CSR 8/11, and zero for MS 0/5 and EDN 0/5. These small counts are descriptive and should not be presented as clinical retrieval precision. An image without a recorded label is not necessarily a true negative for that condition.

Forcing every retrieved example to come from another reported domain substantially weakens the observed same-label hit rates. CSR goes from 8/11 to 0/11; RT from 6/6 to 1/6. RS goes from 12/14 to 0/14, but **none** of those 14 queries has any training example with the RS label in another reported domain. Only one of six RT queries and one of eight CRVO queries has a recorded positive support example in another domain. The comparison is therefore confounded by label availability and pool size. It is a feasibility warning, not an encoder performance verdict.

## Decision

Keep retrieval as an **inspectable development demo** with source and label provenance. Do not call its cosine similarity a disease probability, and do not claim the returned examples are from independent patients; RFMiD lacks patient IDs. Do not use this validation analysis to change the locked final encoder comparison. A credible cross-camera retrieval claim needs verified per-image devices, case-matched positive support in multiple domains, patient identifiers or a defensible related-image grouping, and an independent evaluation set.
