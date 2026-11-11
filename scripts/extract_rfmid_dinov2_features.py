"""Extract frozen DINOv2-S/14 embeddings for RFMiD train/validation only."""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.adapters import prepare_dinov2_tensor
from trace_preprocessing.loader import ManifestImageLoader
from trace_models.dinov2 import (
    DINOV2_SOURCE_COMMIT,
    DINOV2_VITS14_CHECKPOINT_SHA256,
    load_dinov2_vits14,
    sha256_file,
)


def official_class_order(workspace: Path) -> list[str]:
    archive_path = workspace / "A. RFMiD_All_Classes_Dataset.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name for name in archive.namelist() if name.endswith("RFMiD_Training_Labels.csv")
        ]
        if len(members) != 1:
            raise RuntimeError(f"Unexpected RFMiD training label files: {members}")
        text = archive.read(members[0]).decode("utf-8-sig")
    header = next(csv.reader(io.StringIO(text)))
    if header[:2] != ["ID", "Disease_Risk"] or len(header[2:]) != 45:
        raise RuntimeError(f"Unexpected RFMiD header: {header}")
    return header[2:]


def label_vector(raw_json: str, class_order: list[str]) -> np.ndarray:
    values = json.loads(raw_json)
    if set(values) != set(class_order):
        raise RuntimeError("RFMiD label keys do not match the official class order")
    return np.asarray([int(values[name]) for name in class_order], dtype=np.uint8)


def extract_split(
    *,
    split: str,
    loader: ManifestImageLoader,
    pipeline: TracePreprocessor,
    encoder: torch.nn.Module,
    class_order: list[str],
    device: torch.device,
    batch_size: int,
    embedding_dim: int,
    output_path: Path,
) -> dict[str, object]:
    record_ids = loader.record_ids(split=split, dataset="RFMiD")
    if not record_ids:
        raise RuntimeError(f"No RFMiD records for split: {split}")
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    completed_ids: list[str] = []
    batch_tensors: list[torch.Tensor] = []
    batch_ids: list[str] = []
    batch_labels: list[np.ndarray] = []

    def flush_batch() -> None:
        if not batch_tensors:
            return
        inputs = torch.stack(batch_tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = encoder(inputs)
        if not torch.isfinite(output).all():
            raise RuntimeError("Non-finite DINOv2 embedding")
        embeddings.extend(output.float().cpu().numpy())
        completed_ids.extend(batch_ids)
        labels.extend(batch_labels)
        batch_tensors.clear()
        batch_ids.clear()
        batch_labels.clear()

    for index, record_id in enumerate(record_ids, start=1):
        image, row = loader.load_image(record_id)
        if row["_effective_split"] != split:
            raise RuntimeError(f"Split mismatch for {record_id}")
        result = pipeline.process(image, image_id=record_id)
        if result.status == ProcessingStatus.REJECT or result.processed_image is None:
            raise RuntimeError(f"Geometry rejected RFMiD record: {record_id}")
        batch_tensors.append(prepare_dinov2_tensor(result.processed_image))
        batch_ids.append(record_id)
        batch_labels.append(label_vector(row["disease_labels_json"], class_order))
        if len(batch_tensors) >= batch_size:
            flush_batch()
        if index % 100 == 0 or index == len(record_ids):
            print(
                f"feature_progress split={split} {index}/{len(record_ids)}",
                flush=True,
            )
    flush_batch()

    embedding_array = np.asarray(embeddings, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.uint8)
    id_array = np.asarray(completed_ids, dtype=np.str_)
    if embedding_array.shape != (len(record_ids), embedding_dim):
        raise RuntimeError(f"Unexpected embedding shape: {embedding_array.shape}")
    if label_array.shape != (len(record_ids), 45):
        raise RuntimeError(f"Unexpected label shape: {label_array.shape}")
    if len(set(completed_ids)) != len(completed_ids):
        raise RuntimeError("Duplicate record IDs in extracted features")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        record_ids=id_array,
        embeddings=embedding_array,
        labels=label_array,
        class_order=np.asarray(class_order, dtype=np.str_),
        split=np.asarray([split], dtype=np.str_),
    )
    return {
        "split": split,
        "records": len(record_ids),
        "embedding_shape": list(embedding_array.shape),
        "label_shape": list(label_array.shape),
        "positive_counts": {
            name: int(label_array[:, index].sum())
            for index, name in enumerate(class_order)
        },
        "output": str(output_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    workspace = WORKSPACE
    output_dir = workspace / "data" / "features"
    class_order = official_class_order(workspace)
    label_spec = {
        "dataset": "RFMiD",
        "source": "official RFMiD CSV header",
        "output_type": "45 independent sigmoid logits",
        "class_order": class_order,
        "full_name_status": "abbreviations_preserved_pending_verified_dictionary",
    }
    labels_dir = workspace / "data" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "rfmid_class_order.json").write_text(
        json.dumps(label_spec, indent=2), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the full DINOv2 feature extraction run")
    encoder = load_dinov2_vits14(workspace, device=device)
    pipeline = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    )
    loader = ManifestImageLoader(workspace, allow_test=False)
    reports = []
    try:
        for split in ("train", "validation"):
            reports.append(
                extract_split(
                    split=split,
                    loader=loader,
                    pipeline=pipeline,
                    encoder=encoder,
                    class_order=class_order,
                    device=device,
                    batch_size=8,
                    embedding_dim=384,
                    output_path=output_dir / f"rfmid_{split}_dinov2_vits14_geometry_v1.npz",
                )
            )
    finally:
        loader.close()

    summary = {
        "status": "PASS",
        "model": "dinov2_vits14",
        "model_source_commit": DINOV2_SOURCE_COMMIT,
        "model_checkpoint_sha256": DINOV2_VITS14_CHECKPOINT_SHA256,
        "preprocessing": pipeline.config.to_dict(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "test_records_accessed": 0,
        "splits": reports,
    }
    (output_dir / "rfmid_dinov2_feature_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
