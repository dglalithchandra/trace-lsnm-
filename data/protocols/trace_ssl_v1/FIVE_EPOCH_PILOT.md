# TRACE SSL v1 five-epoch development pilot

Status: **pilot completed; no encoder selected**. Conducted 11 November 2026 with zero protected-test images accessed. EDID and JSIEC images were excluded from SSL training.

## What ran

- The label-hidden RFMiD-v1 training manifest contained 1,917 photographs. Frozen geometry preprocessing accepted all: 1,901 directly and 16 with warnings. The cached files are lossless PNGs. A 24-image sample from each split reproduced the original frozen DINOv2-S features with minimum cosine similarity above 0.999998.
- Both runs started from the same verified DINOv2-S/14 checkpoint. Each trained rank-8 LoRA on the last four blocks' attention QKV and output projections, plus a 384→384→128 projection head. A no-gradient EMA teacher and 4,096-entry negative queue provided the MoCo-style InfoNCE objective. The encoder's 384-dimensional output, not the projection head, was used for evaluation.
- The corrected `plain_crop_matched` and `full_field` runs used the same images, seed, shuffled order, batch size 4, learning rate 0.0001, queue settings and **five epochs / 2,395 updates**. Both used the same mild per-channel gain range throughout training. Geometry differed: random resized crop versus ±5° rotation and ±4-pixel translation without cropping. Projection spread remained nonzero and losses finite. Training loss is not a transfer metric.
- An earlier `plain_crop` checkpoint had no colour gain in epoch one and was therefore **excluded from the controlled comparison**. It remains local only as an audit artifact; every table and interval below uses the fresh `plain_crop_matched` run.
- K=3 development evaluation used 11 existing novel labels, 50 frozen support draws each, 639 fixed RFMiD-v1 validation queries, abundant training negatives and the same base-label-selected logistic head (C=1) for every encoder. The frozen DINOv2-S result reproduced the prior 0.198776 macro AUPRC.

## Results

| Encoder | K=3 macro AUPRC | Macro AUROC |
|---|---:|---:|
| Frozen DINOv2-S | 0.1988 | 0.7946 |
| Frozen DINOv2-L | 0.2044 | 0.7826 |
| Frozen RETFound-DINOv2 | 0.1793 | 0.7867 |
| Matched plain-crop adaptation, epoch 5 | 0.2164 | 0.8039 |
| Full-field adaptation, epoch 5 | 0.2189 | 0.8023 |

The corrected paired full-field minus plain-crop macro AUPRC difference was **+0.0025**. A 5,000-resample label-and-support bootstrap gave **−0.0051 to +0.0088**. Full-field improved 6 of 11 labels; it reduced CSR AUPRC by 0.0274 relative to plain crops. Against the best frozen comparator, DINOv2-L, full-field's observed gain was **+0.0144**, with bootstrap interval **−0.0484 to +0.0620**.

These intervals resample labels and support draws while **holding the same validation queries fixed**. They omit query and patient uncertainty; only one training seed was run. RFMiD lacks patient IDs. The episodes have been used in prior development, so this is not a fresh confirmatory result. The full per-label and per-episode results and checkpoint hashes are under `outputs/trace_ssl_v1_k3_dev/` and `models/trace_ssl_v1/` locally; those large artifacts are ignored by git.

Reproduction identifiers: the frozen training-cache index SHA-256 is `6ab2e4315724799f8e475b1ab8bc24e4c0ce9518e36b62126f8a3d8021eec575`; the corrected crop checkpoint SHA-256 is `4a241d7f76e5cd451c194bfd2987a8486911b1947e9647dd9f2e7f8dc6f9153d`; the full-field checkpoint SHA-256 is `3c6b17c70f4224803c9e945bde89479df220e5114146554a4d354494e3335da7`. Each produced 550 episode rows, with no duplicate model/episode pairs.

## Decision

The full-field augmentation preserved more retinal area geometrically in the preflight, but this pilot **does not show that it improves few-shot transfer over ordinary crops**. Both adapted encoders have encouraging development scores relative to DINOv2-S, yet neither has established a reliable win over DINOv2-L or on a fresh cohort. Keep frozen DINOv2-S as the operational baseline and DINOv2-L as the stronger frozen comparison. Do not add neighbor weighting or patch-consistency terms on the basis of this pilot.

Next: complete a fixed longer-duration and multi-seed comparison only if the project can preserve a fresh evaluation gate; otherwise package this as a negative/uncertain augmentation ablation and focus on the organizer's hidden evaluation. A new method should be justified by a measured failure mode, not by the visual appeal of preserving the full retinal circle.
