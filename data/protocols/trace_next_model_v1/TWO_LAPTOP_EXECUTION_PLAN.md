# TRACE: two-laptop execution plan

12 November 2026. This is a proposed division of work, not evidence that the Mac has been configured or benchmarked. Windows has an RTX 4060 laptop GPU and the existing CUDA-based TRACE environment. The user reports a MacBook Pro with M3 Pro and 18 GB unified memory. The project currently has no Git remote.

## Immediate finding

The completed matched 20-epoch RFMiD-v1 SSL study reached 0.2363 macro AUPRC for ordinary crops and 0.2380 for full-field views on reused K=3 development episodes. Both improved by about 0.02 over epoch five; the full-field advantage was only 0.0017 with an interval crossing zero. This makes training duration a more promising near-term lever than another augmentation term. Neither model has a confirmed win over frozen DINOv2-L.

## Machine roles

| Windows RTX 4060 | Mac M3 Pro / 18 GB |
|---|---|
| Keep the existing CUDA SSL trainer, checkpoint and extraction pipeline. Run matched control/candidate training and, later, three seeds. | Audit and prepare new datasets, source terms, labels, related-image groups and overlap screens. Run CPU statistics and documentation. |
| Own the locked RFMiD-v1 final scoring release after human image-identity review. Do not add the new epoch-20 models to its fixed five-model set. | Independently reproduce frozen-feature scoring from transferred NPZ files. Compute paired intervals, per-label plots, acquisition/confounding diagnostics and model-card tables. |
| Run GPU-heavy image extraction for models already verified on Windows. | Optionally run a separate frozen-backbone inference job through PyTorch MPS **only after** a 100-image correctness/throughput/memory smoke test. Do not assume CUDA checkpoints or mixed-precision code run unchanged on MPS. |

Do not distribute one MoCo queue or a single optimizer across both laptops for this sprint. That adds communication and reproducibility work. Give each machine independent jobs whose input and output hashes can be checked.

## Mac readiness gate

1. Install a supported Python/PyTorch build, verify `torch.backends.mps.is_available()`, record macOS, Python, Torch and Metal device information. Apple currently documents macOS 14+, Python 3.10+ and Xcode command-line tools for its stable MPS setup. MPS remains a backend with possible operation gaps.
2. Transfer the committed code as a Git bundle or establish an authenticated private remote. There is **no remote configured** in the present Windows repository. Transfer only the task-specific raw data, features or checkpoints needed by the Mac; do not copy protected test archives for convenience.
3. Compare SHA-256 hashes for every transferred source archive, protocol, feature file and checkpoint. Maintain one dataset-role ledger on both machines. Keep generated features/checkpoints outside Git.
4. Run 100 identical preprocessed development images through a frozen model on both devices, compare record order and cosine similarity of embeddings, and measure images/second and peak memory. Exact bit equality across CUDA and MPS is not expected. If MPS is unstable or slower, use the Mac for data/analysis jobs and keep model inference on CUDA.
5. Porting the current trainer requires device-neutral autocast/scaling, RNG save/restore, tensor placement and a smoke test. `train_trace_ssl_v1.py` and `extract_trace_ssl_v1_features.py` currently explicitly require CUDA. A Mac training run is a **new implementation** and must be compared to the CUDA run before its scores are pooled.

## Highest-value work after setup

1. **Complete the existing final gate.** A human reviews the 12 flagged RFMiD-v1 image pairs. Windows then runs the locked five-encoder comparison once. The result is reported regardless of whether it wins.
2. **Protect a fresh next-study query cohort.** Resolve RFMiD 2.0's image/label mismatch and exact overlaps, or obtain a genuinely independent retinal set with useful low-support labels, patient IDs and device/site metadata. FDR26 can only support DR-specific external checks and its image archive is not yet fully downloaded.
3. **Baseline sweep before new training.** On development data, compare the existing DINOv2-L and RETFound-DINOv2 and any legally accessible DINOv3/RetiZero candidate under identical preprocessing, support IDs and head budget. Record likely public-data pretraining exposure. Pick the best eligible comparator, not the easiest one to beat.
4. **Data-scale study.** Add a licensed, deduplicated fundus pool only after assigning every source a role. Avoid letting a single DR or glaucoma set dominate source-balanced batches. Compare RFMiD-only versus pooled training at equal updates, with the same backbone and plain SSL control.
5. **One measured model change.** Diagnose whether small-lesion information is lost in whole-image embeddings using development lesion annotations and patch-level retrieval. If so, test one local/multiscale objective against the plain matched control. If not, spend effort on better data and training instead.
6. **Evidence package.** Repeat finalists over at least three seeds; report K=1/3/5 AUPRC and AUROC with per-label counts, sensitivity at fixed specificity, calibration and risk/coverage under shift. Confirm once on the protected new cohort. Package the encoder, retrieval, quality warning, model card, UMAP, ablations, latency/memory and contamination ledger. The knowledge graph remains optional until it adds verified findings and wins a controlled ablation.

## What would count as beating the friend's reported system

Their supplied AUROC values and TRACE's K=3 AUPRC are different tasks and cannot be ordered numerically. Obtain the friend's checkpoint if possible and evaluate both systems on **the same query images, disease definitions, support budgets, preprocessing rules and head selection budget**, with training/evaluation overlap disclosed. If the checkpoint is unavailable, compare against their published-like baselines and describe their numbers as external context rather than a direct win.

Official platform/model references: https://developer.apple.com/metal/pytorch/ ; https://docs.pytorch.org/docs/stable/notes/mps.html ; https://github.com/facebookresearch/dinov3 .
