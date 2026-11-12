# TRACE final evaluation plan v1

Status: **locally locked analysis plan; implementation and test audit pending**. Written before extracting or scoring either RFMiD v1 or RFMiD 2.0 test images. This is a repository commitment, not an external preregistration or a clinical validation study.

## Question and fixed methods

Does the five-epoch full-field retinal SSL student improve few-shot ranking over the strongest frozen development comparator, DINOv2-L, on RFMiD v1 test images? Compare five already fixed encoders: frozen DINOv2-S, frozen DINOv2-L, frozen RETFound-DINOv2, matched plain-crop SSL epoch 5 and full-field SSL epoch 5. No further encoder training, checkpoint choice, resolution choice, head tuning, preprocessing tuning or label choice is permitted after this plan.

Use the existing RFMiD v1 training feature files and `trace_fewshot_v1/episode_plan.csv`. For each label, use its 50 existing K=3 positive-support draws, all available RFMiD v1 training negatives, and logistic regression with C=1, selected previously using base labels. Use the same support records, query records, preprocessing (448-pixel frozen TRACE geometry; photometric correction off), head and query order for every encoder. The SSL encoder output is the 384-dimensional pre-projection embedding. The 11 labels are fixed by `trace_fewshot_v1/protocol.json`: ERM, LS, MS, CSR, CRVO, AION, RT, RS, CRS, EDN and RPEC. These are **operational low-support labels**, not eleven independently proven rare diagnoses.

## Primary cohort and endpoint

The primary query cohort is the RFMiD v1 final `test` split from the existing duplicate-aware `trace_analysis_manifest.csv`, restricted to RFMiD records. Its membership must be read from the existing split assignment, never recomputed after seeing scores. No test image may appear in the training/support feature IDs, SSL training manifest, or a confirmed duplicate group crossing those sets. If a new independently verified duplicate is found, exclude its entire affected query group using the same rule for every model, report it, and do not revisit model choice.

A label is primary-eligible if it has at least five positive and five negative query images after the fixed duplicate rule. Apply this count-only rule to the 11 named labels. If fewer than eight labels qualify, **do not make a primary superiority claim**; report all evaluable labels descriptively. For each eligible label and each of 50 K=3 support draws, compute average precision (AUPRC) on the identical query set. Average the 50 AUPRCs within each label, then average labels equally. This macro AUPRC is the primary endpoint. It matches the development pilot's statistic and treats repeated support draws as repeated measurements, not independent patients.

The single primary contrast is **full-field SSL minus frozen DINOv2-L** macro AUPRC. A 95% paired interval will resample labels, support draws within each sampled label, and positive/negative query images within each label (5,000 seeded resamples). Use one shared set of sampled indices for both models. Claim evidence of superiority only if the interval's lower bound exceeds zero; claim an effect of at least +0.02 AUPRC only if that lower bound exceeds +0.02. Otherwise state that superiority was not established. Do not call an interval crossing zero equivalence. Patient clustering cannot be modeled because IDs are unavailable.

## Secondary analyses

Report per-label positives, negatives, prevalence, per-label mean episode AUPRC and AUROC for all five encoders. Report macro AUROC, AUPRC divided by prevalence, and K=1 and K=5 learning-curve points as descriptive analyses using the previously fixed support draws and the same C=1 head. Report paired full-field versus matched plain-crop, versus frozen DINOv2-S, and versus frozen RETFound-DINOv2 contrasts, but do not use them to override the primary decision. No threshold or calibration claim follows from ranking metrics.

The RFMiD 2.0 test split is a **secondary cross-release cohort only**. Before any scoring, audit its label schema and image overlap against RFMiD v1 support and v2 development images. Score only provisional mapped CRS, MS and RT, with the same 50 K=3 v1 training support draws and C=1 head. For each finding, report query count, prevalence, AUPRC, AUROC and uncertainty; if fewer than five positives or negatives remain, mark it descriptive-only. This cohort cannot prove camera invariance because per-image camera metadata and patient IDs are unavailable, and its source mappings remain provisional. Do not pool it with v1 test in the primary endpoint.

## Audit gate before scoring

1. Check source archive hashes, ZIP integrity, split membership and expected schemas. Do not infer label meaning from matching abbreviations alone.
2. Complete exact-hash and local-retinal-detail duplicate screening against training/support and development pools. Human-review ambiguous local-detail matches before releasing test scores. A coarse perceptual-hash hit alone is not grounds for exclusion.
3. Freeze and hash the final extraction and evaluation code. Test feature files and scores are written only after this gate passes; log every test-image read and any exclusions.
4. Run the locked comparison once. Save all per-image scores, per-episode metrics, excluded IDs, input hashes and software versions. Report favorable and unfavorable labels.

If the schema, overlap or preprocessing audit fails and cannot be resolved by the fixed rules, stop without a performance claim and record the reason. The test set is not used to change any model or preprocessing decision. Original images remain untouched.

## Scope of claims

A win on RFMiD v1 test would support few-shot ranking improvement on this release under this support protocol. A consistent RFMiD 2.0 result would add exploratory cross-release evidence. Neither result alone establishes patient-independent diagnosis, safety, or generalization to arbitrary cameras. If the primary contrast fails, retain the frozen baseline and report the negative result.
