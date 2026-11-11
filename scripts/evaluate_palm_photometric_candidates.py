"""Screen bounded PALM photometric harmonisation strengths on validation.

The reference is built from duplicate-controlled PALM train records only. The
frozen pathology head and a frozen train-fitted camera probe are guardrails.
No candidate is declared final automatically because a clinical non-inferiority
margin has not been established.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models.dinov2 import load_dinov2_vits14
from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.adapters import prepare_dinov2_tensor
from trace_preprocessing.loader import ManifestImageLoader
from trace_preprocessing.measurements import measure_photometry, measure_quality
from trace_preprocessing.photometric import apply_corrections, derive_corrections, scale_corrections
from trace_preprocessing.quality import compare_images, decide_quality
from trace_preprocessing.reference import build_reference_profile
from trace_preprocessing.types import PhotometricStats


MANIFEST = WORKSPACE / "data" / "manifests" / "palm_manifest.csv"
AUDIT = WORKSPACE / "outputs" / "palm_preprocessing_audit" / "palm_preprocessing_records.csv"
FEATURES = WORKSPACE / "data" / "features"
GUARDRAIL = WORKSPACE / "models" / "palm_pathology_guardrail_v1" / "palm_pathology_guardrail_v1.npz"
OUTPUT = WORKSPACE / "outputs" / "palm_photometric_candidates"
STRENGTHS = (0.25, 0.50, 1.00)


def load_features(split: str) -> dict[str, np.ndarray]:
    path = FEATURES / f"palm_{split}_dinov2_vits14_geometry_v1.npz"
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def joint_weights(camera: np.ndarray, pathology: np.ndarray) -> np.ndarray:
    groups = np.asarray([f"{a}:{b}" for a, b in zip(camera, pathology)])
    counts = Counter(groups.tolist())
    result = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    return result / result.mean()


def fixed_camera_probe(
    train_embeddings: np.ndarray,
    train_camera: np.ndarray,
    train_pathology: np.ndarray,
) -> Pipeline:
    probe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "probe",
                LogisticRegression(C=0.1, max_iter=5000, solver="liblinear", random_state=2026),
            ),
        ]
    )
    probe.fit(
        train_embeddings,
        train_camera,
        probe__sample_weight=joint_weights(train_camera, train_pathology),
    )
    return probe


def pathology_scores(embeddings: np.ndarray) -> tuple[np.ndarray, float]:
    with np.load(GUARDRAIL) as model:
        normalized = (
            embeddings - model["scaler_mean"].astype(np.float64)
        ) / model["scaler_scale"].astype(np.float64)
        logits = normalized @ model["coefficient"].astype(np.float64).T
        logits = logits[:, 0] + float(model["intercept"][0])
        threshold = float(model["threshold"][0])
    return 1.0 / (1.0 + np.exp(-logits)), threshold


def task_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    return {
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "sensitivity": float(np.mean(predicted[positive])),
        "specificity": float(np.mean(~predicted[negative])),
    }


def camera_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, scores >= 0.5)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def paired_bootstrap(
    labels: np.ndarray,
    cameras: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    threshold: float,
    repeats: int = 2000,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(2026)
    strata = cameras * 2 + labels
    groups = np.unique(strata)
    auprc_delta: list[float] = []
    sensitivity_delta: list[float] = []
    for _ in range(repeats):
        index = np.concatenate(
            [
                rng.choice(np.flatnonzero(strata == group), np.sum(strata == group), replace=True)
                for group in groups
            ]
        )
        y = labels[index]
        before = baseline[index]
        after = candidate[index]
        auprc_delta.append(
            average_precision_score(y, after) - average_precision_score(y, before)
        )
        positive = y == 1
        sensitivity_delta.append(
            float(np.mean(after[positive] >= threshold) - np.mean(before[positive] >= threshold))
        )
    return {
        "auprc_delta_95ci": [
            float(np.quantile(auprc_delta, 0.025)),
            float(np.quantile(auprc_delta, 0.975)),
        ],
        "sensitivity_delta_95ci": [
            float(np.quantile(sensitivity_delta, 0.025)),
            float(np.quantile(sensitivity_delta, 0.975)),
        ],
    }


def train_reference(audit: pd.DataFrame) -> Any:
    fields = list(PhotometricStats.__dataclass_fields__)
    train = audit[audit["effective_split"] == "train"]
    stats = [
        PhotometricStats(**{field: float(row[field]) for field in fields})
        for _, row in train.iterrows()
    ]
    return build_reference_profile(
        stats, ["train"] * len(stats), profile_name="palm_pooled_train_reference"
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT)
    reference = train_reference(audit)
    (OUTPUT / "palm_pooled_train_reference.json").write_text(
        json.dumps(reference.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    train = load_features("train")
    validation = load_features("validation")
    record_ids = validation["record_ids"].astype(str)
    baseline_embeddings = validation["embeddings"].astype(np.float32)
    labels = validation["pathology_labels"].astype(np.uint8)
    cameras = validation["camera_labels"].astype(np.uint8)
    baseline_pathology, threshold = pathology_scores(baseline_embeddings)
    camera_probe = fixed_camera_probe(
        train["embeddings"].astype(np.float64),
        train["camera_labels"].astype(np.uint8),
        train["pathology_labels"].astype(np.uint8),
    )
    baseline_camera = camera_probe.predict_proba(baseline_embeddings)[:, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA_is_required_for_PALM_candidate_evaluation")
    encoder = load_dinov2_vits14(WORKSPACE, device=device)
    config = PreprocessingConfig(apply_photometric_correction=False)
    pipeline = TracePreprocessor(config)
    candidates = {
        strength: np.zeros_like(baseline_embeddings) for strength in STRENGTHS
    }
    pending_tensors: list[torch.Tensor] = []
    pending_locations: list[tuple[float, int]] = []
    safety_rows: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending_tensors:
            return
        batch = torch.stack(pending_tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = encoder(batch).float().cpu().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("non_finite_candidate_embedding")
        for embedding, (strength, index) in zip(output, pending_locations):
            candidates[strength][index] = embedding
        pending_tensors.clear()
        pending_locations.clear()

    with ManifestImageLoader(WORKSPACE, MANIFEST, allow_test=False) as loader:
        for index, record_id in enumerate(record_ids):
            image, row = loader.load_image(record_id)
            if row["_effective_split"] != "validation":
                raise RuntimeError(f"validation_split_guard_failed:{record_id}")
            geometry = pipeline.process(image, image_id=record_id)
            if (
                geometry.status == ProcessingStatus.REJECT
                or geometry.geometry_only_image is None
                or geometry.retinal_mask is None
                or geometry.stats_before is None
                or geometry.quality_before is None
            ):
                raise RuntimeError(f"geometry_failed:{record_id}")
            full = derive_corrections(
                geometry.stats_before, reference, config.photometric_bounds
            )
            for strength in STRENGTHS:
                corrections = scale_corrections(full, strength)
                attempted = apply_corrections(
                    geometry.geometry_only_image, geometry.retinal_mask, corrections
                )
                stats_after = measure_photometry(attempted, geometry.retinal_mask)
                quality_after = measure_quality(attempted, geometry.retinal_mask)
                comparison = compare_images(
                    geometry.geometry_only_image,
                    attempted,
                    geometry.retinal_mask,
                    geometry.stats_before,
                    stats_after,
                )
                status, reasons, warnings = decide_quality(
                    comparison, geometry.quality_before, quality_after, config.quality
                )
                fallback = status == ProcessingStatus.REJECT
                safety_rows.append(
                    {
                        "record_id": record_id,
                        "camera_index": int(cameras[index]),
                        "pathologic_myopia": int(labels[index]),
                        "strength": strength,
                        "status": status.value,
                        "fallback_to_geometry": fallback,
                        "reasons": "|".join(reasons),
                        "warnings": "|".join(warnings),
                        **comparison.to_dict(),
                        **corrections.to_dict(),
                    }
                )
                if fallback:
                    candidates[strength][index] = baseline_embeddings[index]
                else:
                    pending_tensors.append(prepare_dinov2_tensor(attempted))
                    pending_locations.append((strength, index))
                    if len(pending_tensors) >= 8:
                        flush()
            completed = index + 1
            if completed % 50 == 0 or completed == len(record_ids):
                print(f"palm_candidate_progress={completed}/{len(record_ids)}", flush=True)
        flush()

    safety = pd.DataFrame(safety_rows)
    safety.to_csv(OUTPUT / "candidate_safety_records.csv", index=False)
    summaries: list[dict[str, Any]] = []
    for strength in STRENGTHS:
        embedding = candidates[strength]
        disease_scores, observed_threshold = pathology_scores(embedding)
        if abs(observed_threshold - threshold) > 1e-7:
            raise RuntimeError("guardrail_threshold_mismatch")
        camera_scores = camera_probe.predict_proba(embedding)[:, 1]
        selected_safety = safety[safety["strength"] == strength]
        disease = task_metrics(labels, disease_scores, threshold)
        baseline_disease = task_metrics(labels, baseline_pathology, threshold)
        camera = camera_metrics(cameras, camera_scores)
        baseline_domain = camera_metrics(cameras, baseline_camera)
        summary = {
            "strength": strength,
            "records": len(record_ids),
            "fallback_count": int(selected_safety["fallback_to_geometry"].sum()),
            "mean_absolute_image_change": float(selected_safety["mean_absolute_change"].mean()),
            "median_structural_similarity": float(selected_safety["structural_similarity"].median()),
            "pathology_metrics": disease,
            "pathology_delta": {
                key: float(disease[key] - baseline_disease[key]) for key in disease
            },
            "fixed_camera_probe": camera,
            "fixed_camera_probe_delta": {
                key: float(camera[key] - baseline_domain[key]) for key in camera
            },
            "paired_bootstrap": paired_bootstrap(
                labels, cameras, baseline_pathology, disease_scores, threshold
            ),
            "selection_status": "not_frozen_no_clinical_noninferiority_margin",
        }
        summaries.append(summary)
        np.savez_compressed(
            OUTPUT / f"palm_validation_strength_{strength:.2f}_embeddings.npz",
            record_ids=record_ids,
            embeddings=embedding,
            pathology_scores=disease_scores,
            camera_scores=camera_scores,
            strength=np.asarray([strength], dtype=np.float32),
        )

    final = {
        "status": "PASS",
        "reference": reference.to_dict(),
        "reference_source": "PALM duplicate-controlled train records only",
        "baseline": {
            "pathology_metrics": task_metrics(labels, baseline_pathology, threshold),
            "fixed_camera_probe": camera_metrics(cameras, baseline_camera),
        },
        "candidates": summaries,
        "test_records_accessed": 0,
        "decision": (
            "Screening only. A candidate may advance if it reduces camera signal and its "
            "paired disease-performance interval is acceptable under a predeclared margin."
        ),
    }
    (OUTPUT / "candidate_summary.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
