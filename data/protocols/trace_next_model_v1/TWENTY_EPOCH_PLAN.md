# Matched twenty-epoch SSL continuation

Written before the continuation run and its evaluation, 12 November 2026. **Development only.** The five-epoch checkpoints remain unchanged and remain the only SSL candidates in the locked `trace_final_v1` test.

## Question

Was the five-epoch pilot simply too short to reveal a stable difference between ordinary matched crops and lesion-preserving full-field views?

Resume each existing five-epoch RFMiD-v1 training checkpoint into a **new run directory** and continue to epoch 20. Keep its mode, seed, 1,917 label-hidden training images, image order, batch size 4, learning rate 0.0001, LoRA rank 8, momentum teacher, queue and optimizer state. Neither EDID, JSIEC nor any validation/test image enters training. Save the epoch-20 feature files under new names. Evaluate the same fixed 550 K=3 development episodes and C=1 head, with both modes paired by support/query IDs. Compare their macro AUPRC and the frozen DINOv2-L baseline; report every label and a paired label/support interval. Do not select a checkpoint by searching many intermediate epochs.

## Interpretation gate

An epoch-20 development gain is a reason to retain a candidate for a **new** future confirmatory cohort, not evidence of superiority and not permission to amend the current locked final comparison. If longer training helps one mode but hurts several labels or clean performance, report that. One seed, reused validation queries and missing patient IDs remain material limits. A stronger study would need three seeds and a fresh evaluation set.
