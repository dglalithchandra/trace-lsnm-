"""Extract isolated glare, resolution and field-of-view stress features.

Only RFMiD validation is used. Rejected artifacts are still embedded because
this is a failure-characterisation benchmark, not a deployment path.
"""

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
from trace_preprocessing.stress import (
    apply_artifact_stress,
    sample_artifact_stress_parameters,
)


STYLES = ("glare", "resolution_loss", "narrow_field_of_view")
SPLIT = "validation"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for artifact-stress extraction")
    clean_path = (
        WORKSPACE
        / "data/features/rfmid_validation_dinov2_vits14_geometry_v1.npz"
    )
    with np.load(clean_path, allow_pickle=False) as payload:
        record_ids = payload["record_ids"].astype(str)
        clean_embeddings = payload["embeddings"].astype(np.float32)
        labels = payload["labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str)

    encoder = load_dinov2_vits14(WORKSPACE, device=device)
    pipeline = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    )
    loader = ManifestImageLoader(WORKSPACE, allow_test=False)
    if loader.record_ids(split=SPLIT, dataset="RFMiD") != record_ids.tolist():
        loader.close()
        raise RuntimeError("artifact_stress_loader_order_mismatch")

    count = len(record_ids)
    embeddings = np.zeros((count, len(STYLES), clean_embeddings.shape[1]), dtype=np.float32)
    severities = np.empty((count, len(STYLES)), dtype="U8")
    parameters_json = np.empty((count, len(STYLES)), dtype="U256")
    safety_status = np.empty((count, len(STYLES)), dtype="U24")
    safety_reasons = np.empty((count, len(STYLES)), dtype="U512")
    structural_similarity = np.zeros((count, len(STYLES)), dtype=np.float32)
    mean_absolute_change = np.zeros((count, len(STYLES)), dtype=np.float32)
    pending_tensors: list[torch.Tensor] = []
    pending_locations: list[tuple[int, int]] = []
    namespace = "rfmid-validation-artifact-stress-v2"

    def flush() -> None:
        if not pending_tensors:
            return
        inputs = torch.stack(pending_tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            output = encoder(inputs).float().cpu().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("nonfinite_artifact_stress_embedding")
        for value, (record_index, style_index) in zip(output, pending_locations):
            embeddings[record_index, style_index] = value
        pending_tensors.clear()
        pending_locations.clear()

    try:
        for record_index, record_id in enumerate(record_ids):
            raw, row = loader.load_image(record_id)
            if row["_effective_split"] != SPLIT:
                raise RuntimeError(f"artifact_stress_split_mismatch:{record_id}")
            result = pipeline.process(raw, image_id=record_id)
            if (
                result.status == ProcessingStatus.REJECT
                or result.geometry_only_image is None
                or result.retinal_mask is None
            ):
                raise RuntimeError(f"geometry_rejected_artifact_source:{record_id}")
            clean_image = result.geometry_only_image
            mask = result.retinal_mask
            clean_stats = measure_photometry(clean_image, mask)
            clean_quality = measure_quality(clean_image, mask)
            for style_index, style in enumerate(STYLES):
                parameters = sample_artifact_stress_parameters(
                    record_id, style=style, seed_namespace=namespace
                )
                stressed = apply_artifact_stress(
                    clean_image,
                    mask,
                    parameters,
                    image_id=record_id,
                    seed_namespace=namespace,
                )
                stress_stats = measure_photometry(stressed, mask)
                stress_quality = measure_quality(stressed, mask)
                comparison = compare_images(
                    clean_image, stressed, mask, clean_stats, stress_stats
                )
                status, reasons, _ = decide_quality(
                    comparison,
                    clean_quality,
                    stress_quality,
                    pipeline.config.quality,
                )
                severities[record_index, style_index] = parameters.severity
                parameters_json[record_index, style_index] = json.dumps(
                    parameters.to_dict(), sort_keys=True, separators=(",", ":")
                )
                safety_status[record_index, style_index] = status.value
                safety_reasons[record_index, style_index] = "|".join(reasons)
                structural_similarity[record_index, style_index] = (
                    comparison.structural_similarity
                )
                mean_absolute_change[record_index, style_index] = (
                    comparison.mean_absolute_change
                )
                pending_tensors.append(prepare_dinov2_tensor(stressed))
                pending_locations.append((record_index, style_index))
                if len(pending_tensors) >= 8:
                    flush()
            completed = record_index + 1
            if completed % 100 == 0 or completed == count:
                print(
                    f"artifact_stress_progress {completed}/{count} "
                    f"views={completed * len(STYLES)}/{count * len(STYLES)}",
                    flush=True,
                )
        flush()
    finally:
        loader.close()

    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=2) == 0):
        raise RuntimeError("invalid_artifact_stress_embedding_array")
    output = (
        WORKSPACE
        / "data/features/rfmid_validation_dinov2_artifact_stress_v2.npz"
    )
    np.savez_compressed(
        output,
        record_ids=record_ids.astype(np.str_),
        clean_embeddings=clean_embeddings,
        artifact_embeddings=embeddings,
        labels=labels,
        class_order=class_order,
        styles=np.asarray(STYLES, dtype=np.str_),
        severities=severities,
        stress_parameters=parameters_json,
        safety_status=safety_status,
        safety_reasons=safety_reasons,
        structural_similarity=structural_similarity,
        mean_absolute_change=mean_absolute_change,
        split=np.asarray([SPLIT], dtype=np.str_),
        seed_namespace=np.asarray([namespace], dtype=np.str_),
    )

    reports = {}
    for style_index, style in enumerate(STYLES):
        reports[style] = {
            "severity_counts": dict(Counter(severities[:, style_index].tolist())),
            "safety_status_counts": dict(
                Counter(safety_status[:, style_index].tolist())
            ),
            "structural_similarity_p05": float(
                np.quantile(structural_similarity[:, style_index], 0.05)
            ),
            "mean_absolute_change_mean": float(
                mean_absolute_change[:, style_index].mean()
            ),
        }
    summary = {
        "status": "PASS",
        "purpose": "isolated missing-artifact failure characterisation",
        "records": count,
        "views": count * len(STYLES),
        "styles": reports,
        "model": "frozen_dinov2_vits14",
        "model_source_commit": DINOV2_SOURCE_COMMIT,
        "model_checkpoint_sha256": DINOV2_VITS14_CHECKPOINT_SHA256,
        "output": str(output.relative_to(WORKSPACE)).replace("\\", "/"),
        "output_sha256": sha256_file(output),
        "rejected_views_are_embedded_for_failure_analysis": True,
        "synthetic_shift_disclaimer": (
            "isolated artifacts are controlled stress tests, not verified camera simulations"
        ),
        "external_datasets_accessed": 0,
        "protected_test_records_accessed": 0,
    }
    summary_path = (
        WORKSPACE / "data/features/rfmid_artifact_stress_v2_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
