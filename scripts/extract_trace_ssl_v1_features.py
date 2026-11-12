"""Extract RFMiD development features from a trained TRACE SSL student."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def extract_cached_split(split: str, encoder: torch.nn.Module, device: torch.device, batch_size: int, output: Path) -> dict[str, object]:
    baseline = load_feature_split(ROOT / "data/features" / f"rfmid_{split}_dinov2_vits14_geometry_v1.npz", split)
    index_path = ROOT / "data/protocols/trace_ssl_v1" / ("cached_image_index.csv" if split == "train" else "cached_validation_index.csv")
    with index_path.open(newline="", encoding="utf-8") as stream:
        index = {row["record_id"]: row for row in csv.DictReader(stream)}
    if set(index) != set(baseline.record_ids):
        raise RuntimeError(f"Cached {split} IDs do not match frozen baseline IDs")
    features = []
    batch = []

    def flush() -> None:
        if not batch:
            return
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            result = encoder(torch.stack(batch).to(device))
        if not torch.isfinite(result).all():
            raise RuntimeError(f"Nonfinite {split} encoder output")
        features.extend(result.float().cpu().numpy())
        batch.clear()

    for number, record_id in enumerate(baseline.record_ids, 1):
        row = index[str(record_id)]
        path = ROOT / row["cache_path"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (448, 448, 3):
            raise RuntimeError(f"Invalid cached image: {path}")
        batch.append(prepare_dinov2_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if len(batch) >= batch_size:
            flush()
        if number % 200 == 0 or number == len(baseline.record_ids):
            print(f"TRACE SSL feature extraction {split}: {number}/{len(baseline.record_ids)}", flush=True)
    flush()
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.shape != (len(baseline.record_ids), 384):
        raise RuntimeError(f"Unexpected {split} embedding shape: {matrix.shape}")
    np.savez_compressed(output, record_ids=baseline.record_ids, embeddings=matrix, labels=baseline.labels, class_order=np.asarray(baseline.class_order), split=np.asarray([split]))
    return {"split": split, "records": len(baseline.record_ids), "embedding_shape": list(matrix.shape), "feature_sha256": sha256_file(output), "cache_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plain_crop", "plain_crop_matched", "full_field"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or not torch.cuda.is_available():
        raise RuntimeError("Positive batch size and CUDA required")
    checkpoint = ROOT / "models/trace_ssl_v1" / args.mode / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config"]["mode"] != args.mode or int(state["epoch"]) < 1 or int(state["cursor"]) != 0:
        raise RuntimeError("Only completed-epoch checkpoints may be evaluated")
    epoch = int(state["epoch"])
    device = torch.device("cuda")
    encoder = load_dinov2_vits14(ROOT, device=device)
    add_lora(encoder)
    encoder.load_state_dict(state["student"], strict=True)
    encoder.eval().requires_grad_(False).to(device)
    reports = []
    for split in ("train", "validation"):
        out = ROOT / "data/features" / f"rfmid_{split}_trace_ssl_{args.mode}_epoch{epoch}.npz"
        reports.append(extract_cached_split(split, encoder, device, args.batch_size, out))
    summary = {"status": "FROZEN_STUDENT_FEATURES_DEVELOPMENT_ONLY", "mode": args.mode, "epoch": epoch, "checkpoint_sha256": sha256_file(checkpoint), "splits": reports, "test_records_accessed": 0}
    (ROOT / "data/features" / f"trace_ssl_{args.mode}_epoch{epoch}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
