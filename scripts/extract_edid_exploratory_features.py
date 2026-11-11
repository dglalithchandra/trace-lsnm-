"""Extract frozen encoder features for the locked EDID exploratory query set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_models.dinov2 import load_dinov2_vits14, load_dinov2_vitl14, sha256_file
from trace_models.retfound import load_retfound_dinov2_meh
from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.adapters import prepare_dinov2_tensor

ARCHIVE = ROOT / "data/external/EDID/Original_Dataset.zip"
AUDIT = ROOT / "data/protocols/trace_bench_v2/edid_preprocessing_audit.csv"
MODEL_CONFIG = {
    "dinov2_s": (load_dinov2_vits14, 384),
    "dinov2_l": (load_dinov2_vitl14, 1024),
    "retfound": (load_retfound_dinov2_meh, 1024),
}
LABELS = ("MS", "RP", "ODE")
CLASS_TO_LABEL = {"Macular Scar": "MS", "Retinitis Pigmentosa": "RP", "Disc Edema": "ODE"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=tuple(MODEL_CONFIG))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    with AUDIT.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1333 or len(set(row["record_id"] for row in rows)) != len(rows):
        raise RuntimeError("Frozen EDID query audit count or IDs changed")
    if any(row["preprocessing_status"] not in {"accept", "accept_with_warning"} for row in rows):
        raise RuntimeError("Frozen EDID query includes rejected preprocessing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, dim = MODEL_CONFIG[args.encoder]
    model = loader(ROOT, device=device)
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    embeddings = []
    batch = []

    def flush() -> None:
        if not batch:
            return
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            result = model(torch.stack(batch).to(device))
        if not torch.isfinite(result).all():
            raise RuntimeError("Nonfinite frozen encoder output")
        embeddings.extend(result.float().cpu().numpy())
        batch.clear()

    with zipfile.ZipFile(ARCHIVE) as archive:
        for number, row in enumerate(rows, 1):
            raw = archive.read(row["archive_member"])
            bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Previously accepted image failed decode: {row['record_id']}")
            result = pipeline.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image_id=row["record_id"])
            if result.status == ProcessingStatus.REJECT or result.processed_image is None:
                raise RuntimeError(f"Previously accepted image rejected: {row['record_id']}")
            if str(result.status) != row["preprocessing_status"]:
                raise RuntimeError(f"Preprocessing status drift: {row['record_id']}")
            batch.append(prepare_dinov2_tensor(result.processed_image))
            if len(batch) >= args.batch_size:
                flush()
            if number % 100 == 0 or number == len(rows):
                print(f"EDID {args.encoder}: {number}/{len(rows)}", flush=True)
    flush()
    features = np.asarray(embeddings, dtype=np.float32)
    if features.shape != (len(rows), dim):
        raise RuntimeError(f"Unexpected feature shape {features.shape}")
    labels = np.zeros((len(rows), len(LABELS)), dtype=np.uint8)
    for index, row in enumerate(rows):
        label = CLASS_TO_LABEL.get(row["class_folder"])
        if label is not None:
            labels[index, LABELS.index(label)] = 1
    out = ROOT / "data/features" / f"edid_exploratory_{args.encoder}_geometry448_v1.npz"
    np.savez_compressed(out, record_ids=np.asarray([row["record_id"] for row in rows]), embeddings=features, labels=labels, class_order=np.asarray(LABELS), split=np.asarray(["validation"]))
    summary = {"encoder": args.encoder, "records": len(rows), "embedding_shape": list(features.shape), "class_positive_counts": dict(zip(LABELS, labels.sum(axis=0).astype(int).tolist(), strict=True)), "healthy_controls": sum(row["class_folder"] == "Healthy" for row in rows), "feature_sha256": sha256_file(out), "preprocessing": pipeline.config.to_dict(), "query_audit": str(AUDIT.relative_to(ROOT)), "protected_test_images_opened": 0}
    summary_path = ROOT / "data/features" / f"edid_exploratory_{args.encoder}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
