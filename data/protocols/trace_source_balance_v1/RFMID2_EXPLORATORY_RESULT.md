# TRACE A1 RFMiD 2.0 exploratory result

Status: **three-seed cross-release validation completed; encouraging but underpowered development evidence**.

All three selected A1 replication seeds were frozen before this check. The evaluation used the same 1,917 RFMiD-v1 training supports, 50 fixed K=3 draws, abundant negatives and the fixed `C=1` logistic head. Queries were the same 177 RFMiD 2.0 **validation** images used in prior TRACE development. Neither RFMiD-v1 nor RFMiD-2.0 test members were opened.

| Model | CRS AUPRC | MS AUPRC | RT AUPRC | Macro AUPRC | Macro AUROC |
|---|---:|---:|---:|---:|---:|
| Frozen DINOv2-S | 0.0427 | 0.1177 | 0.6464 | 0.2689 | 0.6486 |
| A1 seed 20261111 | 0.0585 | 0.1367 | 0.6323 | 0.2758 | 0.6952 |
| A1 seed 20261112 | 0.0691 | 0.1417 | 0.6626 | 0.2911 | 0.7184 |
| A1 seed 20261113 | 0.0756 | 0.1176 | 0.6443 | 0.2792 | 0.7068 |

A1-minus-frozen macro-AUPRC differences were +0.0069, +0.0222 and +0.0102, so all three seeds had a positive macro point difference; the mean was **+0.0131**. CRS improved in all seeds, MS improved in two and was flat in one, and RT was mixed.

This result is not statistically strong. CRS, MS and RT have only 7, 6 and 8 positive queries. Eight of nine per-seed/per-label paired image-bootstrap intervals crossed zero; the only exception was seed 20261112 on CRS, with a lower bound barely above zero, and no multiplicity-adjusted claim was prespecified. The validation set is not fresh, patient IDs are unavailable, and per-image device metadata is not verified.

Therefore the result supports only this narrow statement: A1's internal development gain did not disappear in the three-label RFMiD 2.0 validation check, and its macro point difference was positive across seeds. It does **not** establish external superiority, camera generalization, patient-independent performance or clinical benefit.

Reproduce with `scripts/extract_trace_a1_rfmid2_validation.py` for each seed and `scripts/evaluate_trace_a1_rfmid2_k3.py`. Generated features and episode outputs remain Git-ignored.
