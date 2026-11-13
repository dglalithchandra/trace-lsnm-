"""Benchmark frozen DINOv2-S versus TRACE A1 on real cached retinal tensors."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_models import TraceA1Encoder
from trace_models.dinov2 import load_dinov2_vits14
from trace_preprocessing.adapters import prepare_dinov2_tensor


def summarize_timings(seconds: list[float], batch_size: int) -> dict[str, float]:
    if not seconds or batch_size < 1 or any(value <= 0 for value in seconds):
        raise ValueError("timings and batch_size must be positive")
    values = np.asarray(seconds, dtype=np.float64)
    return {
        "iterations": len(seconds),
        "batch_size": batch_size,
        "median_batch_milliseconds": float(np.median(values) * 1000),
        "p95_batch_milliseconds": float(np.quantile(values, 0.95) * 1000),
        "images_per_second": float(batch_size * len(values) / values.sum()),
    }


def load_real_tensors(count: int) -> torch.Tensor:
    index_path = ROOT / "data/protocols/trace_ssl_v1/cached_validation_index.csv"
    with index_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))[:count]
    if len(rows) != count:
        raise RuntimeError("Not enough cached validation images for benchmark")
    tensors = []
    for row in rows:
        bgr = cv2.imread(str(ROOT / row["cache_path"]), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (448, 448, 3):
            raise RuntimeError(f"Invalid benchmark image: {row['cache_path']}")
        tensors.append(prepare_dinov2_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    return torch.stack(tensors)


def benchmark_model(
    model: torch.nn.Module,
    tensors: torch.Tensor,
    *,
    batch_size: int,
    iterations: int,
    warmup: int,
) -> dict[str, float]:
    device = torch.device("cuda")
    batch = tensors[:batch_size].to(device)
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(warmup):
            model(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(iterations):
            started = time.perf_counter()
            output = model(batch)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
    if output.shape != (batch_size, 384) or not torch.isfinite(output).all():
        raise RuntimeError(f"Invalid benchmark output: {tuple(output.shape)}")
    return {
        **summarize_timings(timings, batch_size),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations-batch1", type=int, default=100)
    parser.add_argument("--iterations-batch16", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this reference benchmark")
    if min(args.warmup, args.iterations_batch1, args.iterations_batch16) < 1:
        raise ValueError("Warm-up and iteration counts must be positive")
    tensors = load_real_tensors(16)
    device = torch.device("cuda")
    results = []

    baseline = load_dinov2_vits14(ROOT, device=device)
    for batch_size, iterations in ((1, args.iterations_batch1), (16, args.iterations_batch16)):
        results.append(
            {
                "model": "frozen_dinov2_s",
                **benchmark_model(
                    baseline,
                    tensors,
                    batch_size=batch_size,
                    iterations=iterations,
                    warmup=args.warmup,
                ),
            }
        )
    del baseline
    torch.cuda.empty_cache()

    trace = TraceA1Encoder(ROOT, device=device)
    for batch_size, iterations in ((1, args.iterations_batch1), (16, args.iterations_batch16)):
        results.append(
            {
                "model": "trace_a1_seed_20261111",
                **benchmark_model(
                    trace.encoder,
                    tensors,
                    batch_size=batch_size,
                    iterations=iterations,
                    warmup=args.warmup,
                ),
            }
        )

    output = ROOT / "outputs/trace_source_balance_v1/a1_inference_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "REFERENCE_MACHINE_MODEL_FORWARD_BENCHMARK",
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "cuda_float16_autocast",
        "input": "real_cached_RFMiD_validation_tensors_448x448",
        "scope": "encoder_forward_only; excludes disk IO and preprocessing",
        "warmup_iterations": args.warmup,
        "results": results,
        "trace_artifact_kind": trace.artifact_kind,
        "trace_artifact_sha256": trace.checkpoint_sha256,
        "protected_test_records_accessed": 0,
        "limitations": [
            "Single machine and one process; not a multi-machine performance guarantee.",
            "Cached tensors avoid disk and preprocessing costs.",
            "No xFormers acceleration is available in this environment.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
