"""Extract frozen DINOv2-S features for audited Paraguay external development data."""

from __future__ import annotations

import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models.dinov2 import (
    DINOV2_SOURCE_COMMIT,
    DINOV2_VITS14_CHECKPOINT_SHA256,
    load_dinov2_vits14,
    sha256_file,
)
from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.adapters import prepare_dinov2_tensor


MANIFEST = WORKSPACE / "data" / "manifests" / "paraguay_dr_manifest.csv"
OUTPUT = WORKSPACE / "data" / "features"


def main() -> None:
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest["is_primary_analysis_record"]].copy()
    if len(manifest) != 1437 or set(manifest["analysis_split"]) != {
        "external_development_validation"
    }:
        raise RuntimeError("Unexpected Paraguay manifest selection")
    archive_values = manifest["archive_path"].unique().tolist()
    if len(archive_values) != 1:
        raise RuntimeError("Paraguay manifest must reference one archive")
    archive_path = WORKSPACE / archive_values[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Paraguay feature extraction")
    encoder = load_dinov2_vits14(WORKSPACE, device=device)
    pipeline = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    )

    features: list[np.ndarray] = []
    completed_ids: list[str] = []
    labels: list[tuple[int, int, int, int]] = []
    statuses: list[str] = []
    tensors: list[torch.Tensor] = []
    pending: list[tuple[str, tuple[int, int, int, int], str]] = []

    def flush() -> None:
        if not tensors:
            return
        inputs = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = encoder(inputs)
        if output.shape != (len(tensors), 384) or not torch.isfinite(output).all():
            raise RuntimeError("Invalid Paraguay DINOv2 output")
        features.extend(output.float().cpu().numpy())
        for record_id, record_labels, status in pending:
            completed_ids.append(record_id)
            labels.append(record_labels)
            statuses.append(status)
        tensors.clear()
        pending.clear()

    with zipfile.ZipFile(archive_path) as archive:
        for index, row in enumerate(manifest.itertuples(index=False), start=1):
            payload = archive.read(row.archive_member)
            with Image.open(io.BytesIO(payload)) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            result = pipeline.process(rgb, image_id=row.record_id)
            if result.status == ProcessingStatus.REJECT or result.processed_image is None:
                raise RuntimeError(f"Paraguay geometry rejected: {row.record_id}")
            tensors.append(prepare_dinov2_tensor(result.processed_image))
            pending.append(
                (
                    row.record_id,
                    (
                        int(row.dr_grade),
                        int(row.any_dr),
                        int(row.referable_dr),
                        int(row.proliferative_dr),
                    ),
                    result.status.value,
                )
            )
            if len(tensors) >= 8:
                flush()
            if index % 100 == 0 or index == len(manifest):
                print(
                    f"paraguay_feature_progress {index}/{len(manifest)}",
                    flush=True,
                )
    flush()

    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.uint8)
    if feature_array.shape != (len(manifest), 384):
        raise RuntimeError(f"Unexpected Paraguay feature shape: {feature_array.shape}")
    if label_array.shape != (len(manifest), 4):
        raise RuntimeError(f"Unexpected Paraguay label shape: {label_array.shape}")
    output_path = OUTPUT / "paraguay_dr_dinov2_vits14_geometry_v1.npz"
    np.savez_compressed(
        output_path,
        record_ids=np.asarray(completed_ids, dtype=np.str_),
        embeddings=feature_array,
        labels=label_array,
        label_order=np.asarray(
            ["dr_grade", "any_dr", "referable_dr", "proliferative_dr"],
            dtype=np.str_,
        ),
        preprocessing_status=np.asarray(statuses, dtype=np.str_),
        split=np.asarray(["external_development_validation"], dtype=np.str_),
    )
    summary = {
        "status": "PASS",
        "model": "dinov2_vits14",
        "model_source_commit": DINOV2_SOURCE_COMMIT,
        "model_checkpoint_sha256": DINOV2_VITS14_CHECKPOINT_SHA256,
        "preprocessing": pipeline.config.to_dict(),
        "records": len(manifest),
        "embedding_shape": list(feature_array.shape),
        "label_shape": list(label_array.shape),
        "preprocessing_status_counts": dict(Counter(statuses)),
        "output": str(output_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "output_sha256": sha256_file(output_path),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "dataset_role": "external development validation; no model fitting",
        "protected_test_records_accessed": 0,
    }
    (OUTPUT / "paraguay_dr_dinov2_feature_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
