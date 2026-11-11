"""Train a frozen PALM pathologic-myopia guardrail on DINOv2 embeddings.

Hyperparameters and the operating threshold are selected from train-only folds.
The final model is evaluated once on PALM validation, overall and per camera.
Joint camera/pathology weighting discourages the head from exploiting the known
camera-label association. This is a development guardrail, not a clinical model.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


WORKSPACE = Path(__file__).resolve().parents[1]
FEATURES = WORKSPACE / "data" / "features"
OUTPUT = WORKSPACE / "models" / "palm_pathology_guardrail_v1"
CAMERA_ORDER = ["ZEISS Visucam 500", "Canon CR-2"]


def load_split(split: str) -> dict[str, np.ndarray]:
    path = FEATURES / f"palm_{split}_dinov2_vits14_geometry_v1.npz"
    with np.load(path) as data:
        return {name: data[name].copy() for name in data.files}


def joint_weights(camera: np.ndarray, pathology: np.ndarray) -> np.ndarray:
    groups = np.asarray([f"{a}:{b}" for a, b in zip(camera, pathology)])
    counts = Counter(groups.tolist())
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    return weights / weights.mean()


def model(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "head",
                LogisticRegression(
                    C=c_value,
                    max_iter=5000,
                    solver="liblinear",
                    random_state=2026,
                ),
            ),
        ]
    )


def select_model(
    features: np.ndarray,
    labels: np.ndarray,
    cameras: np.ndarray,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    joint = cameras * 2 + labels
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    candidates = [0.001, 0.01, 0.1, 1.0, 10.0]
    reports: dict[str, dict[str, float]] = {}
    oof_by_c: dict[float, np.ndarray] = {}
    for c_value in candidates:
        oof = np.full(len(labels), np.nan, dtype=np.float64)
        fold_ap: list[float] = []
        fold_auc: list[float] = []
        for train_index, holdout_index in folds.split(features, joint):
            fitted = model(c_value)
            weights = joint_weights(cameras[train_index], labels[train_index])
            fitted.fit(
                features[train_index],
                labels[train_index],
                head__sample_weight=weights,
            )
            scores = fitted.predict_proba(features[holdout_index])[:, 1]
            oof[holdout_index] = scores
            fold_ap.append(average_precision_score(labels[holdout_index], scores))
            fold_auc.append(roc_auc_score(labels[holdout_index], scores))
        if not np.isfinite(oof).all():
            raise RuntimeError("incomplete_train_oof_predictions")
        oof_by_c[c_value] = oof
        reports[str(c_value)] = {
            "mean_fold_auprc": float(np.mean(fold_ap)),
            "mean_fold_auroc": float(np.mean(fold_auc)),
            "pooled_oof_auprc": float(average_precision_score(labels, oof)),
            "pooled_oof_auroc": float(roc_auc_score(labels, oof)),
        }
    selected_c = max(
        candidates,
        key=lambda value: (reports[str(value)]["mean_fold_auprc"], -value),
    )
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, oof_by_c[selected_c]
    )
    finite = np.isfinite(thresholds)
    selected_threshold = float(
        thresholds[finite][
            np.argmax(true_positive_rate[finite] - false_positive_rate[finite])
        ]
    )
    return selected_c, selected_threshold, reports


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (scores >= threshold).astype(np.uint8)
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "records": int(len(labels)),
        "positive_records": int(labels.sum()),
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "confusion_matrix_rows_true_negative_positive": matrix.tolist(),
    }


def bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    cameras: np.ndarray,
    threshold: float,
    repeats: int = 2000,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(2026)
    strata = cameras * 2 + labels
    groups = np.unique(strata)
    values = {"auprc": [], "auroc": [], "balanced_accuracy": []}
    for _ in range(repeats):
        index = np.concatenate(
            [
                rng.choice(np.flatnonzero(strata == group), np.sum(strata == group), replace=True)
                for group in groups
            ]
        )
        block = metrics(labels[index], scores[index], threshold)
        for key in values:
            values[key].append(block[key])
    return {
        f"{key}_95ci": [
            float(np.quantile(series, 0.025)),
            float(np.quantile(series, 0.975)),
        ]
        for key, series in values.items()
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = load_split("train")
    validation = load_split("validation")
    train_features = train["embeddings"].astype(np.float64)
    validation_features = validation["embeddings"].astype(np.float64)
    train_labels = train["pathology_labels"].astype(np.uint8)
    validation_labels = validation["pathology_labels"].astype(np.uint8)
    train_cameras = train["camera_labels"].astype(np.uint8)
    validation_cameras = validation["camera_labels"].astype(np.uint8)

    c_value, threshold, cv_report = select_model(
        train_features, train_labels, train_cameras
    )
    fitted = model(c_value)
    weights = joint_weights(train_cameras, train_labels)
    fitted.fit(train_features, train_labels, head__sample_weight=weights)
    scores = fitted.predict_proba(validation_features)[:, 1]
    overall = metrics(validation_labels, scores, threshold)
    overall.update(
        bootstrap(validation_labels, scores, validation_cameras, threshold)
    )
    by_camera: dict[str, Any] = {}
    for camera_index, camera_name in enumerate(CAMERA_ORDER):
        keep = validation_cameras == camera_index
        by_camera[camera_name] = metrics(
            validation_labels[keep], scores[keep], threshold
        )

    scaler = fitted.named_steps["scale"]
    head = fitted.named_steps["head"]
    np.savez_compressed(
        OUTPUT / "palm_pathology_guardrail_v1.npz",
        scaler_mean=scaler.mean_.astype(np.float32),
        scaler_scale=scaler.scale_.astype(np.float32),
        coefficient=head.coef_.astype(np.float32),
        intercept=head.intercept_.astype(np.float32),
        threshold=np.asarray([threshold], dtype=np.float32),
        selected_c=np.asarray([c_value], dtype=np.float32),
        camera_order=np.asarray(CAMERA_ORDER, dtype=np.str_),
    )
    prediction_frame = pd.DataFrame(
        {
            "record_id": validation["record_ids"].astype(str),
            "patient_id": validation["patient_ids"].astype(str),
            "camera_model": [CAMERA_ORDER[value] for value in validation_cameras],
            "pathologic_myopia": validation_labels,
            "predicted_probability": scores,
            "predicted_label": (scores >= threshold).astype(np.uint8),
        }
    )
    prediction_frame.to_csv(OUTPUT / "validation_predictions.csv", index=False)
    summary = {
        "status": "PASS",
        "purpose": "development disease-preservation guardrail; not a clinical model",
        "model": "standardized linear head on frozen DINOv2-S/14 geometry embeddings",
        "training_weighting": "inverse frequency of joint camera x pathology group",
        "selected_c": c_value,
        "threshold_selected_on_train_oof": threshold,
        "train_only_cv": cv_report,
        "validation": {"overall": overall, "by_camera": by_camera},
        "test_records_accessed": 0,
        "important_limit": (
            "PALM contains only pathologic-myopia labels and two imbalanced cameras. "
            "This head is a filter guardrail, not evidence for rare-disease diagnosis."
        ),
    }
    (OUTPUT / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
