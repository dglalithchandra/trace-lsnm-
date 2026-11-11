"""Extract deployment-available QC features for clean and artifact views."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader
from trace_preprocessing.qc_features import extract_qc_features, qc_feature_names
from trace_preprocessing.stress import (
    apply_artifact_stress,
    sample_artifact_stress_parameters,
)


ARTIFACT_STYLES = ("glare", "resolution_loss", "narrow_field_of_view")
VIEW_NAMES = ("clean", *ARTIFACT_STYLES)


def extract_split(
    split: str, loader: ManifestImageLoader, pipeline: TracePreprocessor
) -> dict[str, object]:
    clean_path = (
        WORKSPACE
        / "data/features"
        / f"rfmid_{split}_dinov2_vits14_geometry_v1.npz"
    )
    with np.load(clean_path, allow_pickle=False) as payload:
        record_ids = payload["record_ids"].astype(str)
        labels = payload["labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str)
    if loader.record_ids(split=split, dataset="RFMiD") != record_ids.tolist():
        raise RuntimeError(f"qc_feature_loader_order_mismatch:{split}")

    feature_names = qc_feature_names()
    output = (
        WORKSPACE / "data/features" / f"rfmid_{split}_artifact_qc_features_v1.npz"
    )
    partial = output.with_suffix(".partial.npz")
    expected_shape = (len(record_ids), len(VIEW_NAMES), len(feature_names))
    if output.exists():
        with np.load(output, allow_pickle=False) as existing:
            if (
                existing["features"].shape != expected_shape
                or existing["record_ids"].astype(str).tolist() != record_ids.tolist()
            ):
                raise RuntimeError(f"existing_qc_feature_file_mismatch:{split}")
        return {
            "split": split,
            "records": len(record_ids),
            "views": int(len(record_ids) * len(VIEW_NAMES)),
            "feature_shape": list(expected_shape),
            "output": str(output.relative_to(WORKSPACE)).replace("\\", "/"),
            "resume_action": "reused_complete_split",
        }
    features = np.zeros(
        expected_shape, dtype=np.float32
    )
    severities = np.full((len(record_ids), len(VIEW_NAMES)), "clean", dtype="U8")
    parameters = np.full((len(record_ids), len(VIEW_NAMES)), "{}", dtype="U256")
    namespace = f"rfmid-{split}-artifact-stress-v2"
    start_index = 0
    if partial.exists():
        with np.load(partial, allow_pickle=False) as checkpoint:
            start_index = int(checkpoint["completed"][0])
            if (
                checkpoint["record_ids"].astype(str).tolist()
                != record_ids[:start_index].tolist()
                or checkpoint["features"].shape
                != (start_index, len(VIEW_NAMES), len(feature_names))
            ):
                raise RuntimeError(f"partial_qc_feature_file_mismatch:{split}")
            features[:start_index] = checkpoint["features"]
            severities[:start_index] = checkpoint["severities"].astype(str)
            parameters[:start_index] = checkpoint["stress_parameters"].astype(str)
        print(
            f"qc_feature_resume split={split} completed={start_index}", flush=True
        )

    def save_partial(completed: int) -> None:
        np.savez_compressed(
            partial,
            completed=np.asarray([completed], dtype=np.int32),
            record_ids=record_ids[:completed].astype(np.str_),
            features=features[:completed],
            severities=severities[:completed],
            stress_parameters=parameters[:completed],
            feature_names=np.asarray(feature_names, dtype=np.str_),
            view_names=np.asarray(VIEW_NAMES, dtype=np.str_),
            split=np.asarray([split], dtype=np.str_),
        )

    for index in range(start_index, len(record_ids)):
        record_id = record_ids[index]
        raw, row = loader.load_image(record_id)
        if row["_effective_split"] != split:
            raise RuntimeError(f"qc_feature_split_mismatch:{record_id}")
        result = pipeline.process(raw, image_id=record_id)
        if (
            result.status == ProcessingStatus.REJECT
            or result.geometry_only_image is None
            or result.retinal_mask is None
        ):
            raise RuntimeError(f"qc_feature_geometry_rejected:{record_id}")
        clean = result.geometry_only_image
        mask = result.retinal_mask
        features[index, 0] = extract_qc_features(clean, mask)
        for style_index, style in enumerate(ARTIFACT_STYLES, start=1):
            stress = sample_artifact_stress_parameters(
                record_id, style=style, seed_namespace=namespace
            )
            view = apply_artifact_stress(
                clean,
                mask,
                stress,
                image_id=record_id,
                seed_namespace=namespace,
            )
            features[index, style_index] = extract_qc_features(view, mask)
            severities[index, style_index] = stress.severity
            parameters[index, style_index] = json.dumps(
                stress.to_dict(), sort_keys=True, separators=(",", ":")
            )
        completed = index + 1
        if completed % 100 == 0 or completed == len(record_ids):
            save_partial(completed)
            print(
                f"qc_feature_progress split={split} {completed}/{len(record_ids)}",
                flush=True,
            )

    if not np.isfinite(features).all():
        raise RuntimeError(f"nonfinite_qc_features:{split}")
    np.savez_compressed(
        output,
        record_ids=record_ids.astype(np.str_),
        features=features,
        feature_names=np.asarray(feature_names, dtype=np.str_),
        view_names=np.asarray(VIEW_NAMES, dtype=np.str_),
        severities=severities,
        stress_parameters=parameters,
        labels=labels,
        class_order=class_order,
        split=np.asarray([split], dtype=np.str_),
        seed_namespace=np.asarray([namespace], dtype=np.str_),
    )
    partial.unlink(missing_ok=True)
    return {
        "split": split,
        "records": len(record_ids),
        "views": int(len(record_ids) * len(VIEW_NAMES)),
        "feature_shape": list(features.shape),
        "output": str(output.relative_to(WORKSPACE)).replace("\\", "/"),
        "resume_action": (
            f"resumed_from_{start_index}" if start_index else "started_from_zero"
        ),
    }


def main() -> None:
    pipeline = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    )
    loader = ManifestImageLoader(WORKSPACE, allow_test=False)
    reports = []
    try:
        for split in ("train", "validation"):
            reports.append(extract_split(split, loader, pipeline))
    finally:
        loader.close()
    summary = {
        "status": "PASS",
        "purpose": "train-only artifact QC development and frozen validation",
        "feature_names": list(qc_feature_names()),
        "view_names": list(VIEW_NAMES),
        "splits": reports,
        "external_datasets_accessed": 0,
        "protected_test_records_accessed": 0,
    }
    path = WORKSPACE / "data/features/rfmid_artifact_qc_feature_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
