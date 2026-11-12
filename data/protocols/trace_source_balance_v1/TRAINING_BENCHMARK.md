# TRACE source-balance training benchmark

## Expanded RFMiD+FIVES smoke check (13 November 2026)

The admitted 2,512-record manifest was exercised for 50 updates in each sampling mode before any A2/A3 full run.

| Mode | Steps/s | FIVES draws | RFMiD draws | Peak reserved CUDA memory |
|---|---:|---:|---:|---:|
| Natural frequency | 5.3248 | 43 | 157 | 503,316,480 bytes |
| Source balanced | 5.2804 | 100 | 100 | 503,316,480 bytes |

Natural sampling produced 21.5% FIVES draws in this small deterministic smoke sample, close to the manifest share of 23.7%. Source-balanced sampling produced the required exact 50:50 draw count. Both runs stayed finite and opened zero protected-test images.

Run date: 12 November 2026. Device: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB reported total memory. This measures the existing DINOv2-S LoRA/EMA/queue recipe on the frozen 1,917-image cache. It does not measure full-backbone DINO/iBOT or a future multi-source dataloader.

## Commanded work

- Mode: `plain_crop_matched`
- Micro/effective batch: 4/4; no accumulation
- Precision: CUDA FP16 autocast with gradient scaling
- Learning rate: 0.0001
- Optimizer steps: 50
- Two augmented image views per record, student plus no-gradient EMA teacher, optimizer update and queue update
- Fresh run directory under ignored `models/trace_ssl_training_benchmark_v1/`

## Result

| Measurement | Value |
|---|---:|
| Process exit | 0 |
| Full process wall time | 14.723 s |
| Training-loop time at step 1 | 1.12 s |
| Training-loop time at step 50 | 9.55 s |
| Steps 2-50 elapsed | 8.43 s |
| Mean for steps 2-50 | 0.1720 s/step |
| Approximate training images per second at batch 4 | 23.25 |
| Maximum externally sampled GPU memory | 603 MiB |
| Maximum externally sampled utilization | 64% |
| Samples at roughly 100 ms intervals | 73 |
| Final loss | 3.6118 |
| Final embedding standard-deviation diagnostic | 0.0792 |
| Benchmark checkpoint SHA-256 | `f8915ea9743efe9ee4ad7d7807a3a73e9865253791e04fd4b663c2581169ad3d` |

All recorded losses were finite and the embedding-spread diagnostic remained nonzero. The official DINOv2 code warned that xFormers was unavailable; the run completed using the fallback implementation.

At the measured post-startup rate, 9,580 updates would take about 27.5 minutes of training-loop time before periodic checkpoint and other overhead. The completed earlier 9,580-step full-field run recorded 1,119.28 seconds (18.65 minutes) inside its training loop. The current external 100 ms GPU polling may have slowed the short benchmark. Use an approximate **19-28 minute range** for this exact core LoRA recipe, then remeasure after the multi-source sampler exists. This estimate does not apply to full DINO/iBOT training.

The memory number is sampled process/device usage rather than `torch.cuda.max_memory_allocated`; short activation peaks may be missed. It establishes ample observed headroom for the current batch-4 LoRA recipe, not a safe batch size for a different objective or backbone.

## Decision

Freeze batch 4, no gradient accumulation and FP16 autocast for the first A1-A3 screen unless the expanded dataloader fails or a recorded protocol revision occurs before A2/A3 scores exist. Keep the 9,580-update budget for matched arms. Rebenchmark the actual multi-source implementation before starting A2/A3.

## New fixed-update trainer smoke test

The separate `scripts/train_trace_source_balance_v1.py` implementation then ran 50 natural-frequency steps using the new core v2 manifest:

| Measurement | Value |
|---|---:|
| Records/sources | 1,917 / 1 |
| Steps / source draws | 50 / 200 |
| Training-process time | 9.937 s |
| Steps per second | 5.0317 |
| Approximate images per second at batch 4 | 20.13 |
| PyTorch peak allocated memory | 448,646,656 bytes (427.9 MiB) |
| PyTorch peak reserved memory | 503,316,480 bytes (480 MiB) |
| Protected-test images opened | 0 |

The checkpoint resumed from step 50 to step 60 without resetting the cumulative source-draw count (240 after resume). The resume process reported 628,193,792 allocated bytes and 696,254,464 reserved bytes at peak, reflecting restored optimizer/checkpoint state. All recorded losses remained finite and embedding spread remained nonzero. The new trainer refuses source-balanced mode until at least two admitted `source_id` values exist.

At 5.0317 steps/s, 9,580 updates would require about 31.7 minutes of training-process time before extraction and evaluation. Treat this as a core-pool planning estimate. Multi-source I/O and sampling require their own benchmark.
