"""Extract frozen DINOv2-S/14 features for PALM train/validation only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


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
from trace_preprocessing.loader import ManifestImageLoader


MANIFEST = WORKSPACE / "data" / "manifests" / "palm_manifest.csv"
OUTPUT = WORKSPACE / "data" / "features"
CAMERA_ORDER = ["ZEISS Visucam 500", "Canon CR-2"]


def split_ids(loader: ManifestImageLoader, split: str) -> list[str]:
    frame = loader.manifest
    primary = frame["is_primary_analysis_record"].astype(str).str.lower().eq("true")
    selected = frame[(frame["_effective_split"] == split) & primary]
    if selected.empty:
        raise RuntimeError(f"no_primary_palm_records_for_split:{split}")
    return selected["record_id"].tolist()


def extract_split(
    *,
    split: str,
    loader: ManifestImageLoader,
    pipeline: TracePreprocessor,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, object]:
    record_ids = split_ids(loader, split)
    features: list[np.ndarray] = []
    completed: list[str] = []
    cameras: list[int] = []
    pathology: list[int] = []
    patients: list[str] = []
    tensors: list[torch.Tensor] = []
    batch_ids: list[str] = []
    batch_cameras: list[int] = []
    batch_pathology: list[int] = []
    batch_patients: list[str] = []

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
        if output.ndim != 2 or output.shape[1] != 384:
            raise RuntimeError(f"unexpected_dinov2_output_shape:{tuple(output.shape)}")
        if not torch.isfinite(output).all():
            raise RuntimeError("non_finite_palm_embedding")
        features.extend(output.float().cpu().numpy())
        completed.extend(batch_ids)
        cameras.extend(batch_cameras)
        pathology.extend(batch_pathology)
        patients.extend(batch_patients)
        tensors.clear()
        batch_ids.clear()
        batch_cameras.clear()
        batch_pathology.clear()
        batch_patients.clear()

    for index, record_id in enumerate(record_ids, start=1):
        image, row = loader.load_image(record_id)
        if row["_effective_split"] != split or split == "test":
            raise RuntimeError(f"split_guard_failed:{record_id}")
        camera = row["camera_model"]
        if camera not in CAMERA_ORDER:
            raise RuntimeError(f"unexpected_palm_camera:{camera}")
        result = pipeline.process(image, image_id=record_id)
        if result.status == ProcessingStatus.REJECT or result.processed_image is None:
            raise RuntimeError(f"palm_geometry_rejected:{record_id}")
        tensors.append(prepare_dinov2_tensor(result.processed_image))
        batch_ids.append(record_id)
        batch_cameras.append(CAMERA_ORDER.index(camera))
        batch_pathology.append(int(row["pathologic_myopia"]))
        batch_patients.append(row.get("patient_id", ""))
        if len(tensors) >= batch_size:
            flush()
        if index % 50 == 0 or index == len(record_ids):
            print(f"palm_feature_progress split={split} {index}/{len(record_ids)}", flush=True)
    flush()

    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.shape != (len(record_ids), 384):
        raise RuntimeError(f"unexpected_saved_feature_shape:{feature_array.shape}")
    output_path = OUTPUT / f"palm_{split}_dinov2_vits14_geometry_v1.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        record_ids=np.asarray(completed, dtype=np.str_),
        patient_ids=np.asarray(patients, dtype=np.str_),
        embeddings=feature_array,
        camera_labels=np.asarray(cameras, dtype=np.uint8),
        pathology_labels=np.asarray(pathology, dtype=np.uint8),
        camera_order=np.asarray(CAMERA_ORDER, dtype=np.str_),
        split=np.asarray([split], dtype=np.str_),
    )
    return {
        "split": split,
        "records": len(record_ids),
        "embedding_shape": list(feature_array.shape),
        "camera_counts": {
            name: int(np.sum(np.asarray(cameras) == idx))
            for idx, name in enumerate(CAMERA_ORDER)
        },
        "pathologic_myopia_positive": int(np.sum(pathology)),
        "output": str(output_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA_is_required_for_full_PALM_feature_extraction")
    encoder = load_dinov2_vits14(WORKSPACE, device=device)
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    reports: list[dict[str, object]] = []
    with ManifestImageLoader(WORKSPACE, MANIFEST, allow_test=False) as loader:
        for split in ("train", "validation"):
            reports.append(
                extract_split(
                    split=split,
                    loader=loader,
                    pipeline=pipeline,
                    encoder=encoder,
                    device=device,
                )
            )
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
    path = OUTPUT / "palm_dinov2_feature_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
