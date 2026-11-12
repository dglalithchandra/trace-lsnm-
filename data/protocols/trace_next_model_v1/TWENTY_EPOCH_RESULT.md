# Twenty-epoch matched SSL continuation: development result

12 November 2026. **Training and development evaluation complete; no new encoder selected for the locked final test.** The original five-epoch checkpoint hashes were verified unchanged after both continuations. No RFMiD v1 or RFMiD 2.0 protected-test encoder features or disease scores were computed.

## Design

Both students resumed from their separate epoch-five checkpoints to epoch 20, ending at 9,580 updates each. They used the same 1,917 label-hidden RFMiD-v1 training images, batch size 4, seed, LoRA configuration, optimizer, momentum teacher, queue and learning rate. The intended difference was ordinary matched crops versus full-field views. New 384-dimensional pre-projection train/validation embeddings were extracted from the epoch-20 students. The existing 550 K=3 episodes on 11 low-support labels, 639 fixed validation queries and C=1 logistic head were replayed without retuning.

The crop continuation was interrupted and resumed from saved checkpoints during a runtime-thread experiment: a four-thread CPU setting was slower, so the run returned to 24 threads. Its optimizer, random state, queue and image sequence were restored from checkpoints. CPU execution scheduling therefore differed briefly across the two runs; this is a small reproducibility limitation, even though the statistical training recipe remained fixed.

| Encoder | K=3 macro AUPRC | Macro AUROC |
|---|---:|---:|
| Frozen DINOv2-L | 0.2044 | 0.7826 |
| Matched crop SSL, epoch 5 | 0.2164 | 0.8039 |
| Full-field SSL, epoch 5 | 0.2189 | 0.8023 |
| Matched crop SSL, epoch 20 | **0.2363** | **0.8128** |
| Full-field SSL, epoch 20 | **0.2380** | 0.8123 |

Paired macro AUPRC differences and 95% label/support bootstrap intervals, with validation queries held fixed:

| Contrast | Difference | Interval |
|---|---:|---:|
| Full-field 20 minus crop 20 | +0.0017 | -0.0102 to +0.0139 |
| Full-field 20 minus DINOv2-L | +0.0335 | -0.0278 to +0.0849 |
| Crop 20 minus DINOv2-L | +0.0319 | -0.0344 to +0.0902 |
| Full-field 20 minus full-field 5 | +0.0191 | +0.0021 to +0.0362 |
| Crop 20 minus crop 5 | +0.0199 | +0.0022 to +0.0391 |

The epoch-20 full-field model beat the epoch-20 crop model on **4 of 11** labels. Its largest loss relative to crops was CSR (-0.0404 AUPRC); its largest gain was LS (+0.0405). Relative to its own epoch-five checkpoint, full-field improved 9 labels but worsened EDN (-0.0374) and ERM (-0.0033). Crop improved 10 labels and worsened EDN (-0.0326).

## Decision

**Longer label-hidden training is the stronger development signal.** This study still does not establish superiority to the best frozen comparator because that interval crosses zero. It also does not support the claim that full-field views are better than matched crops. The validation queries have been reused across development experiments, only one training seed exists, patient IDs are absent and the intervals do not resample queries. No epoch-20 model may be added to the already locked `trace_final_v1` five-model comparison.

Next: preserve a fresh cohort before further model selection. If that is available, repeat the plain crop and full-field finalists with multiple seeds and a larger, licensed unlabeled training pool, with matched updates and a data-scale ablation. Spend architecture effort on a measured lesion-scale failure rather than adding a contrastive term because it sounds novel. Until a fresh cohort exists, package the epoch-20 models as development candidates and keep the strongest *confirmed* model claim open.

Reproduction: `scripts/train_trace_ssl_v1.py` with `--run-dir models/trace_ssl_long_v1/<mode> --resume --epochs 20`; `scripts/extract_trace_ssl_v1_features.py` with the corresponding new checkpoint and output tag; `scripts/evaluate_trace_ssl_long_v1_k3.py`. Checkpoint, feature and episode-output hashes are in local generated summaries under `models/trace_ssl_long_v1/`, `data/features/` and `outputs/trace_ssl_long_v1_k3_dev/`.
