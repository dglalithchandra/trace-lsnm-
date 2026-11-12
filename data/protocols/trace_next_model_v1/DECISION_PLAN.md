# TRACE next model study — decision plan

Status: proposed on 12 November 2026. This does not amend the locked `trace_final_v1` comparison and does not authorize use of its protected test scores to select a new model.

## What is established locally

- The frozen camera module supplies 448-pixel geometry, retina mask, and quality measurements. The optional photometric correction has not earned inclusion.
- On 11 RFMiD v1 low-support labels, K=3 development macro AUPRC was 0.2044 for frozen DINOv2-L, 0.2164 for the matched five-epoch crop SSL student, and 0.2189 for the five-epoch full-field student. The full-field minus DINOv2-L interval was -0.0484 to +0.0620. This is inconclusive. One training seed and fixed validation queries limit the evidence.
- The RFMiD v1 test is behind an image-identity review gate. RFMiD 2.0 test has an unresolved image/label mapping and overlap audit. Neither is available for new model selection.
- Development retrieval clusters by reported acquisition domain. That observation is confounded by class availability and source characteristics; it does not establish a camera-specific failure mechanism.

## Research question and success condition

Can a label-hidden retinal adaptation improve few-shot ranking on diseases with few labelled examples, including under acquisition shift, over the **best eligible frozen foundation-model baseline** at matched image input and support budget?

A strong model result requires a fresh, sufficiently sized held-out query cohort, matched baselines, at least three training seeds for finalists, and a prespecified paired uncertainty interval. An attractive development score is a model-selection signal, not the result. Record label counts, patient/device metadata availability, duplicate rules and likely pretraining exposure for each model and cohort.

## Sequence

1. **Finish the existing experiment unchanged.** Obtain the twelve image-identity decisions, validate the audit file, release and run the five locked encoders once. Report all results, including a null result. Keep this test out of the next study's model selection.
2. **Create a fresh study cohort.** Audit RFMiD 2.0's authoritative image-to-label mapping and overlaps before any scoring. Seek a second independently labelled retinal cohort with patient and per-image device IDs. Use IDRiD only for its supported DR/lesion tasks; do not relabel common DR as a rare disease. If only RFMiD low-support labels are available, name the endpoint accordingly. Set aside the new query cohort before comparing new models.
3. **Find the strongest frozen comparator on development.** Evaluate existing DINOv2-L and RETFound-DINOv2, plus DINOv3-L and RetiZero only if their weights, usage terms and input protocol are verified. Use the same support/query IDs, head-selection budget and clinically valid resolution. Include an ImageNet or plain SSL control. Log possible public-dataset pretraining overlap.
4. **Diagnose the actual model failure.** On development images only, compare whole-image against patch/multiscale features on small-lesion examples; assess source bias conditional on label and quality; check ranking versus transferred threshold/calibration separately. Choose one failure that is both measured and fixable.
5. **Train one primary candidate and matched controls.** Start with label-hidden continued retinal SSL on the chosen backbone. Expand the legal, deduplicated unlabeled training pool if available, with source-balanced sampling. Compare a plain SSL control and one mechanism tied to step 4 (for example lesion-preserving views or patch consistency). Hold data, resolution, updates, optimizer and head budget equal. Run three seeds for finalists, save all checkpoints/recipes and report compute. Do not add mechanisms after viewing the fresh test.
6. **Select on development; confirm once.** Prespecify eligible labels, K=1/3/5 support policy, macro AUPRC and per-label normalized AUPRC, AUROC, sensitivity at fixed specificity, threshold calibration, paired intervals and a minimum useful gain. Freeze the model and code, then score the fresh held-out cohort once. If the candidate does not clear the gate, retain the best frozen baseline and report the result.
7. **Package the submission.** Deliver the encoder and image-to-embedding interface, few-shot head, audited retrieval, quality/OOD warning, model card, dataset/contamination ledger, UMAP, ablation table, latency/memory and a small demo. The friend's knowledge graph remains an optional independent study until verified finding inputs and a controlled graph-versus-no-graph comparison exist.

## Decisions that must be made from evidence

- **More data:** likely useful, but the local five-epoch pilot cannot show whether data scale, adaptation method or training duration is the bottleneck. Run a matched scale/duration ablation; no pooled dataset is admitted without license and duplicate checks.
- **Architecture:** do not assume an extra graph, a camera-adversarial head or a complex contrastive term improves transfer. Compare against the strongest frozen model first. Prior retinal SSL adaptation exists, so the contribution must be the demonstrated failure diagnosis and improvement under a sound protocol.
- **Claims:** without verified per-image device IDs, report source or acquisition-domain shift, not camera invariance. Without patient IDs, disclose possible related-image dependence. Without a truly rare-disease cohort, report low-label transfer, not clinical rare-disease validation.

## Primary research sources to recheck when acquiring weights/data

- DINOv3 official implementation: https://github.com/facebookresearch/dinov3
- RETFound-DINOv2 versus generalist ocular models: https://arxiv.org/abs/2509.03421
- Block-expanded DINORET retinal adaptation: https://arxiv.org/abs/2409.17332
- RetiZero vision-language model: https://arxiv.org/abs/2406.09317
