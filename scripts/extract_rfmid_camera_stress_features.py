"""Extract paired camera-stressed DINOv2 embeddings for RFMiD train/validation."""

from __future__ import annotations

import json
import sys
from collections import Counter
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
from trace_preprocessing.measurements import measure_photometry, measure_quality
from trace_preprocessing.quality import compare_images, decide_quality
from trace_preprocessing.stress import apply_camera_stress, sample_stress_parameters


def extract_split(
    *,
    split: str,
    loader: ManifestImageLoader,
    encoder: torch.nn.Module,
    device: torch.device,
    pipeline: TracePreprocessor,
    output_path: Path,
) -> dict[str, object]:
    clean_path = (
        WORKSPACE
        / "data"
        / "features"
        / f"rfmid_{split}_dinov2_vits14_geometry_v1.npz"
    )
    with np.load(clean_path, allow_pickle=False) as payload:
        record_ids = payload["record_ids"].astype(str)
        clean_embeddings = payload["embeddings"].astype(np.float32)
        labels = payload["labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str)
        if str(payload["split"][0]) != split:
            raise RuntimeError(f"Clean feature split mismatch: {split}")
    loader_ids = loader.record_ids(split=split, dataset="RFMiD")
    if loader_ids != record_ids.tolist():
        raise RuntimeError(f"Loader/clean feature order mismatch: {split}")

    stressed_embeddings = np.zeros_like(clean_embeddings)
    fallback = np.zeros(len(record_ids), dtype=np.uint8)
    stress_parameters: list[str] = [""] * len(record_ids)
    safety_status: list[str] = [""] * len(record_ids)
    safety_reasons: list[str] = [""] * len(record_ids)
    mean_absolute_change = np.zeros(len(record_ids), dtype=np.float32)
    structural_similarity = np.zeros(len(record_ids), dtype=np.float32)
    pending_tensors: list[torch.Tensor] = []
    pending_indices: list[int] = []
    namespace = f"rfmid-{split}-camera-stress-v1"

    def flush_batch() -> None:
        if not pending_tensors:
            return
        inputs = torch.stack(pending_tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            output = encoder(inputs)
        output_array = output.float().cpu().numpy()
        if not np.isfinite(output_array).all():
            raise RuntimeError("Non-finite stressed DINOv2 embedding")
        for embedding, index in zip(output_array, pending_indices):
            stressed_embeddings[index] = embedding
        pending_tensors.clear()
        pending_indices.clear()

    for index, record_id in enumerate(record_ids):
        image, row = loader.load_image(record_id)
        if row["_effective_split"] != split:
            raise RuntimeError(f"Split mismatch for {record_id}")
        geometry = pipeline.process(image, image_id=record_id)
        if (
            geometry.status == ProcessingStatus.REJECT
            or geometry.geometry_only_image is None
            or geometry.retinal_mask is None
            or geometry.stats_before is None
            or geometry.quality_before is None
        ):
            raise RuntimeError(f"Geometry rejected stress record: {record_id}")
        parameters = sample_stress_parameters(
            record_id, seed_namespace=namespace
        )
        stressed = apply_camera_stress(
            geometry.geometry_only_image,
            geometry.retinal_mask,
            parameters,
            image_id=record_id,
            seed_namespace=namespace,
        )
        stats_after = measure_photometry(stressed, geometry.retinal_mask)
        quality_after = measure_quality(stressed, geometry.retinal_mask)
        comparison = compare_images(
            geometry.geometry_only_image,
            stressed,
            geometry.retinal_mask,
            geometry.stats_before,
            stats_after,
        )
        status, reasons, _ = decide_quality(
            comparison,
            geometry.quality_before,
            quality_after,
            pipeline.config.quality,
        )
        stress_parameters[index] = json.dumps(
            parameters.to_dict(), sort_keys=True, separators=(",", ":")
        )
        safety_status[index] = status.value
        safety_reasons[index] = "|".join(reasons)
        mean_absolute_change[index] = comparison.mean_absolute_change
        structural_similarity[index] = comparison.structural_similarity
        if status == ProcessingStatus.REJECT:
            fallback[index] = 1
            stressed_embeddings[index] = clean_embeddings[index]
        else:
            pending_tensors.append(prepare_dinov2_tensor(stressed))
            pending_indices.append(index)
            if len(pending_tensors) >= 8:
                flush_batch()
        completed = index + 1
        if completed % 100 == 0 or completed == len(record_ids):
            print(
                f"stress_feature_progress split={split} {completed}/{len(record_ids)}",
                flush=True,
            )
    flush_batch()
    if not np.isfinite(stressed_embeddings).all():
        raise RuntimeError("Non-finite values in stressed feature array")

    np.savez_compressed(
        output_path,
        record_ids=record_ids.astype(np.str_),
        clean_embeddings=clean_embeddings,
        stressed_embeddings=stressed_embeddings,
        labels=labels,
        class_order=class_order,
        fallback_to_clean=fallback,
        stress_parameters=np.asarray(stress_parameters, dtype=np.str_),
        safety_status=np.asarray(safety_status, dtype=np.str_),
        safety_reasons=np.asarray(safety_reasons, dtype=np.str_),
        mean_absolute_change=mean_absolute_change,
        structural_similarity=structural_similarity,
        split=np.asarray([split], dtype=np.str_),
        seed_namespace=np.asarray([namespace], dtype=np.str_),
    )
    styles = Counter()
    for payload in stress_parameters:
        values = json.loads(payload)
        if values["vignette_strength"] > 0:
            styles["illumination_falloff"] += 1
        elif values["blur_sigma"] > 0:
            styles["optics_sensor_compression"] += 1
        elif values["red_gain"] != 1.0 or values["blue_gain"] != 1.0:
            styles["colour_exposure"] += 1
        else:
            styles["tone_contrast"] += 1
    return {
        "split": split,
        "records": len(record_ids),
        "embedding_shape": list(stressed_embeddings.shape),
        "fallback_count": int(fallback.sum()),
        "fallback_fraction": float(fallback.mean()),
        "mean_absolute_change_mean": float(mean_absolute_change.mean()),
        "structural_similarity_p05": float(
            np.quantile(structural_similarity, 0.05)
        ),
        "style_counts": dict(styles),
        "output": str(output_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    output_dir = WORKSPACE / "data" / "features"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for full stress-feature extraction")
    encoder = load_dinov2_vits14(WORKSPACE, device=device)
    pipeline = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    )
    loader = ManifestImageLoader(WORKSPACE, allow_test=False)
    reports: list[dict[str, object]] = []
    try:
        for split in ("train", "validation"):
            reports.append(
                extract_split(
                    split=split,
                    loader=loader,
                    encoder=encoder,
                    device=device,
                    pipeline=pipeline,
                    output_path=(
                        output_dir
                        / f"rfmid_{split}_dinov2_camera_stress_v1.npz"
                    ),
                )
            )
    finally:
        loader.close()
    summary = {
        "status": "PASS",
        "purpose": "paired synthetic camera-shift consistency development",
        "synthetic_shift_disclaimer": (
            "bounded stress views are not verified simulations of specific cameras"
        ),
        "model": "frozen_dinov2_vits14",
        "model_source_commit": DINOV2_SOURCE_COMMIT,
        "model_checkpoint_sha256": DINOV2_VITS14_CHECKPOINT_SHA256,
        "splits": reports,
        "test_records_accessed": 0,
    }
    (output_dir / "rfmid_camera_stress_feature_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
