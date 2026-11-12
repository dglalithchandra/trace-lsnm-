"""Extract development-only RFMiD features from a completed source-balance run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_trace_ssl_v1 import add_lora
from trace_models.dinov2 import load_dinov2_vits14, sha256_file
from trace_preprocessing.adapters import prepare_dinov2_tensor
from trace_protocols import load_feature_split


def extract_split(
    split: str,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    output: Path,
) -> dict[str, object]:
    baseline_path = ROOT / "data/features" / f"rfmid_{split}_dinov2_vits14_geometry_v1.npz"
    baseline = load_feature_split(baseline_path, split)
    index_name = "cached_image_index.csv" if split == "train" else "cached_validation_index.csv"
    index_path = ROOT / "data/protocols/trace_ssl_v1" / index_name
    with index_path.open(newline="", encoding="utf-8") as stream:
        index = {row["record_id"]: row for row in csv.DictReader(stream)}
    if set(index) != set(baseline.record_ids):
        raise RuntimeError(f"Cached {split} IDs do not match frozen baseline IDs")

    features: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            encoded = encoder(torch.stack(batch).to(device))
        if not torch.isfinite(encoded).all():
            raise RuntimeError(f"Non-finite {split} encoder output")
        features.extend(encoded.float().cpu().numpy())
        batch.clear()

    for number, record_id in enumerate(baseline.record_ids, 1):
        path = ROOT / index[str(record_id)]["cache_path"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (448, 448, 3):
            raise RuntimeError(f"Invalid cached image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        batch.append(prepare_dinov2_tensor(rgb))
        if len(batch) >= batch_size:
            flush()
        if number % 200 == 0 or number == len(baseline.record_ids):
            print(f"source-balance features {split}: {number}/{len(baseline.record_ids)}", flush=True)
    flush()

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.shape != (len(baseline.record_ids), 384):
        raise RuntimeError(f"Unexpected {split} embedding shape: {matrix.shape}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        record_ids=baseline.record_ids,
        embeddings=matrix,
        labels=baseline.labels,
        class_order=np.asarray(baseline.class_order),
        split=np.asarray([split]),
    )
    return {
        "split": split,
        "records": len(baseline.record_ids),
        "embedding_shape": list(matrix.shape),
        "feature_sha256": sha256_file(output),
        "baseline_schema_sha256": sha256_file(baseline_path),
        "cache_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Completed checkpoint under models/")
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.batch_size < 1:
        raise RuntimeError("CUDA and a positive batch size are required")
    if not re.fullmatch(r"[a-z0-9_]+", args.output_tag):
        raise RuntimeError("Output tag must use lowercase letters, digits and underscores")

    checkpoint = (ROOT / args.checkpoint).resolve()
    if not checkpoint.is_relative_to((ROOT / "models").resolve()):
        raise RuntimeError("Checkpoint must be under models/")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state.get("config", {})
    if config.get("version") != "trace-source-balance-v1":
        raise RuntimeError("Not a TRACE source-balance-v1 checkpoint")
    if int(state.get("step", -1)) != int(config.get("updates", -2)):
        raise RuntimeError("Only a completed fixed-update checkpoint may be evaluated")

    device = torch.device("cuda")
    encoder = load_dinov2_vits14(ROOT, device=device)
    add_lora(encoder)
    encoder.load_state_dict(state["student"], strict=True)
    encoder.eval().requires_grad_(False).to(device)

    reports = []
    for split in ("train", "validation"):
        output = ROOT / "data/features" / f"rfmid_{split}_trace_source_balance_{args.output_tag}.npz"
        if output.exists():
            raise RuntimeError(f"Refusing to overwrite existing features: {output}")
        reports.append(extract_split(split, encoder, device, args.batch_size, output))

    summary = {
        "status": "DEVELOPMENT_FEATURES_COMPLETE",
        "output_tag": args.output_tag,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": config,
        "splits": reports,
        "protected_test_records_accessed": 0,
    }
    summary_path = ROOT / "data/features" / f"trace_source_balance_{args.output_tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
