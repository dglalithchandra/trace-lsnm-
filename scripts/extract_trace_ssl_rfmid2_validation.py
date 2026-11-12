"""Extract RFMiD 2.0 validation features from a completed TRACE SSL student.

This reads Validation_set.zip only. No v2 image enters encoder training.
"""

from __future__ import annotations

import argparse
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

from scripts.extract_rfmid2_validation_features import LABELS, validation_records
from scripts.train_trace_ssl_v1 import add_lora
from trace_models.dinov2 import load_dinov2_vits14, sha256_file
from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.adapters import prepare_dinov2_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plain_crop_matched", "full_field"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    checkpoint = ROOT / "models/trace_ssl_v1" / args.mode / "last.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config"]["mode"] != args.mode or int(state["epoch"]) != 5 or int(state["cursor"]) != 0:
        raise RuntimeError("Expected the completed matched five-epoch checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = load_dinov2_vits14(ROOT, device=device)
    add_lora(encoder)
    encoder.load_state_dict(state["student"], strict=True)
    encoder.eval().requires_grad_(False).to(device)
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    records = validation_records()
    embeddings: list[np.ndarray] = []
    image_hashes: list[str] = []
    statuses: dict[str, int] = {}
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = encoder(torch.stack(batch).to(device))
        if not torch.isfinite(output).all():
            raise RuntimeError("Nonfinite encoder output")
        embeddings.extend(output.float().cpu().numpy())
        batch.clear()

    for number, (record_id, _, raw) in enumerate(records, 1):
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot decode {record_id}")
        result = pipeline.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image_id=record_id)
        if result.status == ProcessingStatus.REJECT or result.processed_image is None:
            raise RuntimeError(f"Preprocessing rejected {record_id}")
        statuses[result.status.value] = statuses.get(result.status.value, 0) + 1
        image_hashes.append(hashlib.sha256(raw).hexdigest())
        batch.append(prepare_dinov2_tensor(result.processed_image))
        if len(batch) >= args.batch_size:
            flush()
        if number % 50 == 0 or number == len(records):
            print(f"{args.mode}: {number}/{len(records)}", flush=True)
    flush()
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.shape != (177, 384):
        raise RuntimeError(f"Unexpected feature shape {matrix.shape}")
    output = ROOT / "data/features" / f"rfmid2_validation_trace_ssl_{args.mode}_epoch5.npz"
    np.savez_compressed(
        output,
        record_ids=np.asarray([record_id for record_id, _, _ in records]),
        embeddings=matrix,
        labels=np.stack([labels for _, labels, _ in records]),
        class_order=np.asarray(LABELS),
        split=np.asarray(["validation"]),
    )
    summary = {
        "status": "EXPLORATORY_RFMiD2_VALIDATION_ONLY",
        "mode": args.mode,
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_archive_sha256": sha256_file(ROOT / "data/external/RFMiD2.0/RFMiD2_0.zip"),
        "query_image_hashes_sha256": hashlib.sha256("\n".join(image_hashes).encode()).hexdigest(),
        "feature_sha256": sha256_file(output),
        "records": len(records),
        "preprocessing_status_counts": statuses,
        "protected_test_members_opened": 0,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
