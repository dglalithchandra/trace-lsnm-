# TRACE final pre-scoring input audit — 12 November 2026

Status: **scoring gate not passed**. This report records a review of inputs before any final-test encoder feature extraction or disease scoring. It does not report model performance. The locally locked analysis plan remains unchanged.

## What was checked

- Locked SHA-256 values for the RFMiD v1 and RFMiD 2.0 source ZIPs and the RFMiD v1 split manifest matched `plan_lock.json`. The archive contents were read for this audit; the test data can no longer be described as unopened.
- The RFMiD v1 split still contains 1,917 training, 639 validation and 644 test records. Previously known duplicate groups do not cross splits.
- All 3,200 RFMiD v1 images and all 860 RFMiD 2.0 images were screened for exact SHA-256 copies. A 256-bit difference hash generated candidate pairs, followed by SIFT local-feature matching for 1,939 v1 test-to-development pairs and an alignment check for high-priority and near-threshold matches. Candidate search is not a proof that no other duplicate exists.
- No final-test encoder features, predictions, episode scores or performance metrics were computed.

## Source integrity findings

The RFMiD 2.0 training ZIP has 509 image files and 509 label rows, but their ID sets differ: CSV-only IDs `89, 216`, image-only IDs `79, 222`. The validation ZIP has 177 matched IDs. The test ZIP has 174 image files and 174 label rows, but CSV-only IDs `151, 191, 405, 490` and image-only IDs `251, 291, 404, 489`. Equal file and row counts therefore do **not** establish a valid image-to-label join. No ID correction was inferred.

Three RFMiD 2.0 test images are byte-identical to RFMiD 2.0 development images: test `886` matches train `798`; test `887` matches validation `799`; test `888` matches validation `800`. These would need exclusion even if the label join were fixed. The RFMiD 2.0 secondary test cohort is blocked for scoring under the locked schema and overlap gate.

## RFMiD v1 cross-split candidate review

No byte-identical v1 test-to-development image was found. Six strong local-feature alarms were reviewed against the original images, and the alarm images were registered and compared using high-frequency image detail. Four test/training pairs appear to be duplicate photographs with matching retinal vessel and lesion patterns:

| Test ID | Training ID | Aligned high-pass correlation | Current audit decision |
| --- | --- | ---: | --- |
| `541` | `1877` | 0.669 | visually confirmed duplicate photograph |
| `571` | `1818` | 0.795 | visually confirmed duplicate photograph |
| `617` | `1852` | 0.674 | visually confirmed duplicate photograph |
| `620` | `1787` | 0.805 | visually confirmed duplicate photograph |

Two other strong SIFT alarms, test `377`/train `835` and test `506`/train `1577`, had weak aligned detail correlation (0.219 and 0.184) and did not visually match; they are false alarms. This demonstrates why SIFT inliers alone are insufficient for exclusion.

A sensitivity screen of 98 pairs below the original 20-inlier alarm threshold found additional uncertain candidates. Test `226`/train `1292` is especially difficult because both images are heavily blurred; the aligned detail correlation is 0.650, but the visible retinal detail does not independently establish identity. Test `175`/train `709` is also unresolved because the training photograph has very little discernible retinal detail. Test `36`/validation `186`, `261`/train `347`, `113`/validation `79`, and `8`/train `1201` were visually different despite moderate correlations. A high correlation in low-quality images can reflect shared coarse structures or acquisition artifacts.

## Decision and next gate

Do **not** use the four confirmed duplicate v1 test records as evaluation queries. Do not treat the uncertain pairs as confirmed duplicates or silently exclude them. The locked plan requires human review of ambiguous local-detail matches, so the primary v1 scoring gate remains closed pending documented adjudication. Since patient IDs are absent, even a completed image-duplicate audit cannot prove patient independence.

For a clean resumption: review the uncertain v1 pairs against original images and registration evidence, record a fixed decision for each before any scoring, then freeze/hash the extraction and evaluation code. Score v1 once only if the audit gate passes. Keep v2 secondary scoring blocked unless an authoritative image-to-label mapping is obtained and its exact overlaps are handled under the fixed rule. No encoder or preprocessing setting may be retuned from this audit.

Raw machine-readable evidence is in the ignored `outputs/trace_final_v1_audit/` directory. The scripts that produced it are `scripts/audit_trace_final_test_inputs.py`, `scripts/review_trace_v1_test_detail.py`, `scripts/measure_trace_alarm_alignment.py`, and `scripts/measure_trace_near_threshold.py`.
