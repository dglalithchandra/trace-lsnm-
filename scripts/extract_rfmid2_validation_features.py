"""Extract frozen encoder features from RFMiD 2.0 validation only."""

from __future__ import annotations

import argparse
import csv
import io
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

ARCHIVE = ROOT / "data/external/RFMiD2.0/RFMiD2_0.zip"
LABELS = ("CRS", "MS", "RT")
MODEL_CONFIG = {
    "dinov2_s": (load_dinov2_vits14, 384),
    "dinov2_l": (load_dinov2_vitl14, 1024),
    "retfound": (load_retfound_dinov2_meh, 1024),
}


def validation_records() -> list[tuple[str, np.ndarray, bytes]]:
    """Read the nested validation archive; never read Test_set.zip."""
    with zipfile.ZipFile(ARCHIVE) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("Validation_set.zip"))) as inner:
            csv_members = [name for name in inner.namelist() if name.lower().endswith(".csv")]
            if len(csv_members) != 1:
                raise RuntimeError(f"Expected one validation CSV, got {csv_members}")
            rows = list(csv.DictReader(io.StringIO(inner.read(csv_members[0]).decode("cp1252"))))
            if len(rows) != 177:
                raise RuntimeError(f"Validation row count changed: {len(rows)}")
            image_members = {
                Path(name).stem: name for name in inner.namelist() if name.lower().endswith(".jpg")
            }
            if len(image_members) != len(rows):
                raise RuntimeError("Validation image/label count mismatch")
            records = []
            for row in rows:
                image_id = row["ID"].strip()
                if image_id not in image_members:
                    raise RuntimeError(f"Missing validation image for ID {image_id}")
                label = np.asarray([int(row[name]) for name in LABELS], dtype=np.uint8)
                if not np.isin(label, [0, 1]).all():
                    raise RuntimeError(f"Nonbinary label for ID {image_id}")
                records.append((f"rfmid2:validation:{image_id}", label, inner.read(image_members[image_id])))
    if len({record_id for record_id, _, _ in records}) != len(records):
        raise RuntimeError("Duplicate validation IDs")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=tuple(MODEL_CONFIG))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, embedding_dim = MODEL_CONFIG[args.encoder]
    encoder = loader(ROOT, device=device)
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    records = validation_records()
    embeddings: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        inputs = torch.stack(batch).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = encoder(inputs)
        if not torch.isfinite(output).all():
            raise RuntimeError("Nonfinite encoder embedding")
        embeddings.extend(output.float().cpu().numpy())
        batch.clear()

    for number, (record_id, _, raw) in enumerate(records, start=1):
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot decode {record_id}")
        result = pipeline.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image_id=record_id)
        if result.status == ProcessingStatus.REJECT or result.processed_image is None:
            raise RuntimeError(f"Preprocessing rejected {record_id}: {result.status}")
        batch.append(prepare_dinov2_tensor(result.processed_image))
        if len(batch) >= args.batch_size:
            flush()
        if number % 25 == 0 or number == len(records):
            print(f"{args.encoder} validation {number}/{len(records)}", flush=True)
    flush()
    features = np.asarray(embeddings, dtype=np.float32)
    labels = np.stack([label for _, label, _ in records])
    if features.shape != (len(records), embedding_dim):
        raise RuntimeError(f"Unexpected embedding shape: {features.shape}")
    output_dir = ROOT / "data/features"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"rfmid2_validation_{args.encoder}_geometry448_v1.npz"
    np.savez_compressed(
        output,
        record_ids=np.asarray([record_id for record_id, _, _ in records]),
        embeddings=features,
        labels=labels,
        class_order=np.asarray(LABELS),
        split=np.asarray(["validation"]),
    )
    summary = {
        "encoder": args.encoder,
        "records": len(records),
        "embedding_shape": list(features.shape),
        "label_positive_counts": dict(zip(LABELS, labels.sum(axis=0).astype(int).tolist(), strict=True)),
        "feature_sha256": sha256_file(output),
        "preprocessing": pipeline.config.to_dict(),
        "source_archive": str(ARCHIVE.relative_to(ROOT)),
        "source_split": "Validation_set.zip",
        "protected_test_members_opened": 0,
    }
    (output_dir / f"rfmid2_validation_{args.encoder}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
