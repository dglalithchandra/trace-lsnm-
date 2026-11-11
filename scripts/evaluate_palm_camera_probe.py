"""Measure residual PALM camera signal without touching the protected test split.

The probe is trained on PALM train and evaluated once on PALM validation. Joint
camera/pathology sample weighting and disease-stratified metrics reduce (but do
not eliminate) the known camera/pathologic-myopia confounding.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parents[1]
FEATURES = WORKSPACE / "data" / "features"
AUDIT = WORKSPACE / "outputs" / "palm_preprocessing_audit" / "palm_preprocessing_records.csv"
OUTPUT = WORKSPACE / "outputs" / "palm_camera_probe"
CAMERA_ORDER = ["ZEISS Visucam 500", "Canon CR-2"]
GEOMETRY_FEATURES = [
    "fundus_detection_confidence",
    "source_retinal_coverage",
    "common_mask_retinal_loss_fraction",
    "normalized_detected_retina_fill_fraction",
]
PHOTOMETRIC_QUALITY_FEATURES = [
    "blur_laplacian_variance",
    "noise_mad",
    "glare_fraction",
    "compression_blockiness",
    "red_green_ratio",
    "blue_green_ratio",
    "saturation_median",
    "brightness_median",
    "contrast_p95_p05",
    "gamma_to_midgray",
    "illumination_nonuniformity",
]


def load_split(split: str) -> dict[str, np.ndarray]:
    path = FEATURES / f"palm_{split}_dinov2_vits14_geometry_v1.npz"
    with np.load(path) as data:
        result = {name: data[name].copy() for name in data.files}
    observed = result["camera_order"].astype(str).tolist()
    if observed != CAMERA_ORDER:
        raise RuntimeError(f"camera_order_mismatch:{observed}")
    return result


def joint_weights(camera: np.ndarray, pathology: np.ndarray) -> np.ndarray:
    groups = np.asarray([f"{a}:{b}" for a, b in zip(camera, pathology)])
    counts = Counter(groups.tolist())
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    return weights / weights.mean()


def build_model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "probe",
                LogisticRegression(
                    C=c_value,
                    class_weight=None,
                    max_iter=5000,
                    solver="liblinear",
                    random_state=2026,
                ),
            ),
        ]
    )


def select_c(
    features: np.ndarray,
    camera: np.ndarray,
    pathology: np.ndarray,
) -> tuple[float, dict[str, float]]:
    joint = camera * 2 + pathology
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    candidates = [0.001, 0.01, 0.1, 1.0, 10.0]
    results: dict[str, float] = {}
    for c_value in candidates:
        scores: list[float] = []
        for train_index, holdout_index in folds.split(features, joint):
            model = build_model(c_value)
            weights = joint_weights(camera[train_index], pathology[train_index])
            model.fit(
                features[train_index],
                camera[train_index],
                probe__sample_weight=weights,
            )
            prediction = model.predict(features[holdout_index])
            scores.append(
                balanced_accuracy_score(camera[holdout_index], prediction)
            )
        results[str(c_value)] = float(np.mean(scores))
    best = max(candidates, key=lambda value: (results[str(value)], -value))
    return best, results


def stratified_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    pathology: np.ndarray,
    *,
    repeats: int = 2000,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(2026)
    strata = labels * 2 + pathology
    unique = np.unique(strata)
    bacc_values: list[float] = []
    auc_values: list[float] = []
    for _ in range(repeats):
        sampled = np.concatenate(
            [
                rng.choice(np.flatnonzero(strata == group), size=np.sum(strata == group), replace=True)
                for group in unique
            ]
        )
        y = labels[sampled]
        s = scores[sampled]
        bacc_values.append(balanced_accuracy_score(y, s >= 0.5))
        auc_values.append(roc_auc_score(y, s))
    return {
        "balanced_accuracy_95ci": [
            float(np.quantile(bacc_values, 0.025)),
            float(np.quantile(bacc_values, 0.975)),
        ],
        "roc_auc_95ci": [
            float(np.quantile(auc_values, 0.025)),
            float(np.quantile(auc_values, 0.975)),
        ],
    }


def metric_block(
    labels: np.ndarray,
    scores: np.ndarray,
    pathology: np.ndarray,
) -> dict[str, Any]:
    prediction = (scores >= 0.5).astype(np.uint8)
    block: dict[str, Any] = {
        "records": int(len(labels)),
        "camera_counts": {
            CAMERA_ORDER[index]: int(np.sum(labels == index)) for index in (0, 1)
        },
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix_rows_true_ZEISS_Canon": confusion_matrix(
            labels, prediction, labels=[0, 1]
        ).tolist(),
    }
    block.update(stratified_bootstrap(labels, scores, pathology))
    by_pathology: dict[str, Any] = {}
    for value, name in ((0, "non_pathologic_myopia"), (1, "pathologic_myopia")):
        keep = pathology == value
        by_pathology[name] = {
            "records": int(keep.sum()),
            "camera_counts": {
                CAMERA_ORDER[index]: int(np.sum(labels[keep] == index))
                for index in (0, 1)
            },
            "balanced_accuracy": float(
                balanced_accuracy_score(labels[keep], prediction[keep])
            ),
            "roc_auc": float(roc_auc_score(labels[keep], scores[keep])),
        }
    block["by_pathology"] = by_pathology
    return block


def fit_probe(
    name: str,
    feature_names: list[str],
    train_features: np.ndarray,
    validation_features: np.ndarray,
    train_camera: np.ndarray,
    validation_camera: np.ndarray,
    train_pathology: np.ndarray,
    validation_pathology: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    c_value, cv_scores = select_c(train_features, train_camera, train_pathology)
    model = build_model(c_value)
    weights = joint_weights(train_camera, train_pathology)
    model.fit(train_features, train_camera, probe__sample_weight=weights)
    scores = model.predict_proba(validation_features)[:, 1]
    coefficient = model.named_steps["probe"].coef_[0]
    report = {
        "name": name,
        "feature_count": int(train_features.shape[1]),
        "selected_c": c_value,
        "train_only_cv_balanced_accuracy_by_c": cv_scores,
        "largest_standardized_coefficients": [
            {"feature": feature_names[index], "coefficient": float(coefficient[index])}
            for index in np.argsort(np.abs(coefficient))[::-1][:10]
        ],
        "validation": metric_block(
            validation_camera, scores, validation_pathology
        ),
    }
    return report, scores


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    validation = load_split("validation")
    train_ids = train["record_ids"].astype(str)
    validation_ids = validation["record_ids"].astype(str)
    train_camera = train["camera_labels"].astype(np.uint8)
    validation_camera = validation["camera_labels"].astype(np.uint8)
    train_pathology = train["pathology_labels"].astype(np.uint8)
    validation_pathology = validation["pathology_labels"].astype(np.uint8)

    audit = pd.read_csv(AUDIT).set_index("record_id")
    missing = (set(train_ids) | set(validation_ids)) - set(audit.index)
    if missing:
        raise RuntimeError(f"audit_features_missing_for_records:{len(missing)}")
    train_audit = audit.loc[train_ids]
    validation_audit = audit.loc[validation_ids]
    train_geometry = train_audit[GEOMETRY_FEATURES].to_numpy(dtype=np.float64)
    validation_geometry = validation_audit[GEOMETRY_FEATURES].to_numpy(dtype=np.float64)
    train_appearance = train_audit[PHOTOMETRIC_QUALITY_FEATURES].to_numpy(dtype=np.float64)
    validation_appearance = validation_audit[PHOTOMETRIC_QUALITY_FEATURES].to_numpy(dtype=np.float64)
    train_simple = np.concatenate([train_geometry, train_appearance], axis=1)
    validation_simple = np.concatenate([validation_geometry, validation_appearance], axis=1)
    if not all(
        np.isfinite(array).all()
        for array in (train_geometry, validation_geometry, train_appearance, validation_appearance)
    ):
        raise RuntimeError("non_finite_photometric_quality_probe_feature")

    disease_only_report, disease_scores = fit_probe(
        "pathology_label_only_confound_check",
        ["pathologic_myopia"],
        train_pathology[:, None].astype(np.float64),
        validation_pathology[:, None].astype(np.float64),
        train_camera,
        validation_camera,
        train_pathology,
        validation_pathology,
    )
    geometry_report, geometry_scores = fit_probe(
        "geometry_metadata_only",
        GEOMETRY_FEATURES,
        train_geometry,
        validation_geometry,
        train_camera,
        validation_camera,
        train_pathology,
        validation_pathology,
    )
    appearance_report, appearance_scores = fit_probe(
        "photometry_quality_metrics_only",
        PHOTOMETRIC_QUALITY_FEATURES,
        train_appearance,
        validation_appearance,
        train_camera,
        validation_camera,
        train_pathology,
        validation_pathology,
    )
    simple_report, simple_scores = fit_probe(
        "geometry_plus_photometry_quality_metrics",
        GEOMETRY_FEATURES + PHOTOMETRIC_QUALITY_FEATURES,
        train_simple,
        validation_simple,
        train_camera,
        validation_camera,
        train_pathology,
        validation_pathology,
    )
    embedding_report, embedding_scores = fit_probe(
        "frozen_dinov2_geometry_embeddings",
        [f"embedding_{index}" for index in range(train["embeddings"].shape[1])],
        train["embeddings"].astype(np.float64),
        validation["embeddings"].astype(np.float64),
        train_camera,
        validation_camera,
        train_pathology,
        validation_pathology,
    )

    prediction_frame = pd.DataFrame(
        {
            "record_id": validation_ids,
            "patient_id": validation["patient_ids"].astype(str),
            "true_camera_index": validation_camera,
            "true_camera": [CAMERA_ORDER[value] for value in validation_camera],
            "pathologic_myopia": validation_pathology,
            "disease_only_canon_probability": disease_scores,
            "geometry_canon_probability": geometry_scores,
            "appearance_canon_probability": appearance_scores,
            "simple_metrics_canon_probability": simple_scores,
            "dinov2_canon_probability": embedding_scores,
        }
    )
    prediction_frame.to_csv(OUTPUT / "palm_validation_camera_probe_predictions.csv", index=False)
    summary = {
        "status": "PASS",
        "camera_order": CAMERA_ORDER,
        "train_records": int(len(train_ids)),
        "validation_records": int(len(validation_ids)),
        "test_records_accessed": 0,
        "training_weighting": "inverse frequency of joint camera x pathology group",
        "model_selection": "five-fold train-only CV stratified by joint camera x pathology group",
        "probes": [
            disease_only_report,
            geometry_report,
            appearance_report,
            simple_report,
            embedding_report,
        ],
        "interpretation_limit": (
            "Camera and pathologic-myopia labels are associated in PALM. Joint weighting "
            "and within-pathology metrics reduce confounding but cannot prove causal camera invariance."
        ),
    }
    (OUTPUT / "palm_camera_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
