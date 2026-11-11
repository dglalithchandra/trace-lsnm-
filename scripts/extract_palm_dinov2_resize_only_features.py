"""Extract PALM DINOv2-S features after direct square resize only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.extract_palm_dinov2_features import CAMERA_ORDER, MANIFEST, split_ids
from trace_models.dinov2 import (
    DINOV2_SOURCE_COMMIT,
    DINOV2_VITS14_CHECKPOINT_SHA256,
    load_dinov2_vits14,
    sha256_file,
)
from trace_preprocessing.adapters import prepare_dinov2_tensor
from trace_preprocessing.loader import ManifestImageLoader


OUTPUT = WORKSPACE / "data" / "features"


def extract_split(
    *,
    split: str,
    loader: ManifestImageLoader,
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
    batch_rows: list[tuple[str, int, int, str]] = []

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
            raise RuntimeError("invalid_resize_only_dinov2_output")
        features.extend(output.float().cpu().numpy())
        for record_id, camera, label, patient in batch_rows:
            completed.append(record_id)
            cameras.append(camera)
            pathology.append(label)
            patients.append(patient)
        tensors.clear()
        batch_rows.clear()

    for index, record_id in enumerate(record_ids, start=1):
        image, row = loader.load_image(record_id)
        if row["_effective_split"] != split or split == "test":
            raise RuntimeError(f"split_guard_failed:{record_id}")
        camera = row["camera_model"]
        if camera not in CAMERA_ORDER:
            raise RuntimeError(f"unexpected_palm_camera:{camera}")
        # This deliberately keeps borders, field placement and aspect-ratio
        # distortion. It is a weak resize-only control, not a recommended path.
        tensors.append(prepare_dinov2_tensor(image, size=448))
        batch_rows.append(
            (
                record_id,
                CAMERA_ORDER.index(camera),
                int(row["pathologic_myopia"]),
                row.get("patient_id", ""),
            )
        )
        if len(tensors) >= batch_size:
            flush()
        if index % 50 == 0 or index == len(record_ids):
            print(
                f"palm_resize_feature_progress split={split} {index}/{len(record_ids)}",
                flush=True,
            )
    flush()

    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.shape != (len(record_ids), 384):
        raise RuntimeError(f"unexpected_saved_feature_shape:{feature_array.shape}")
    output_path = OUTPUT / f"palm_{split}_dinov2_vits14_resize_only_v1.npz"
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
        "output": str(output_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA_is_required_for_full_PALM_feature_extraction")
    encoder = load_dinov2_vits14(WORKSPACE, device=device)
    reports: list[dict[str, object]] = []
    with ManifestImageLoader(WORKSPACE, MANIFEST, allow_test=False) as loader:
        for split in ("train", "validation"):
            reports.append(
                extract_split(
                    split=split,
                    loader=loader,
                    encoder=encoder,
                    device=device,
                )
            )
    summary = {
        "status": "PASS",
        "model": "dinov2_vits14",
        "model_source_commit": DINOV2_SOURCE_COMMIT,
        "model_checkpoint_sha256": DINOV2_VITS14_CHECKPOINT_SHA256,
        "preprocessing": "direct square resize to 448 only; no crop, centring or mask",
        "control_only": True,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "test_records_accessed": 0,
        "splits": reports,
    }
    (OUTPUT / "palm_dinov2_resize_only_feature_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
