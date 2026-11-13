# TRACE A1 inference benchmark

Status: **reference-machine encoder-forward benchmark complete**.

The benchmark used 16 real, already preprocessed RFMiD validation tensors at 448×448. It ran PyTorch 2.11.0 with CUDA 12.8 and FP16 autocast on the NVIDIA GeForce RTX 4060 Laptop GPU. Each arm received 10 warm-up iterations. Timings use explicit CUDA synchronization and exclude disk I/O and preprocessing.

| Model | Batch | Median batch latency | P95 batch latency | Throughput | Peak allocated |
|---|---:|---:|---:|---:|---:|
| Frozen DINOv2-S | 1 | 8.23 ms | 10.78 ms | 115.0 img/s | 112.87 MB |
| TRACE A1 | 1 | 8.55 ms | 8.74 ms | 115.7 img/s | 113.36 MB |
| Frozen DINOv2-S | 16 | 72.07 ms | 74.12 ms | 225.5 img/s | 337.85 MB |
| TRACE A1 | 16 | 74.07 ms | 79.28 ms | 213.3 img/s | 338.34 MB |

TRACE A1's median latency overhead was 3.9% at batch 1 and 2.8% per batch at batch 16. Batch-16 throughput was 5.4% lower. Peak allocated memory increased by about 0.46 MiB (0.14%) at batch 16. The tiny batch-1 throughput reversal is timing noise and is not evidence that LoRA makes the base model faster.

These numbers describe one machine and one process without xFormers. They are not end-to-end latency, a Mac comparison or a deployment guarantee. Zero protected-test records were accessed. Reproduction: `scripts/benchmark_trace_a1_inference.py`.
