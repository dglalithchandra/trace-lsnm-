"""Frozen RFMiD-to-Paraguay external validation and trust diagnostics.

Nothing is fitted or selected on Paraguay. The RFMiD head, calibration,
decision threshold, conformal calibration set, and selective thresholds are
all fixed before external scores are evaluated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models.rfmid_dinov2 import (
    DECISION_TOLERANCE,
    load_rfmid_bundle,
    probabilities_from_embedding,
)
from trace_trust.metrics import (
    expected_calibration_error,
    finite_conformal_quantile,
    mondrian_prediction_sets,
    risk_coverage_auc,
    selective_metrics,
)


OUTPUT = WORKSPACE / "outputs" / "paraguay_external_shift_v1"
BOOTSTRAP_REPLICATES = 2_000
SEED = 20261111


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    labels = labels.astype(np.uint8)
    probabilities = probabilities.astype(np.float64)
    predictions = (probabilities >= threshold - DECISION_TOLERANCE).astype(np.uint8)
    positive = labels == 1
    negative = ~positive
    sensitivity = float(predictions[positive].mean())
    specificity = float((predictions[negative] == 0).mean())
    return {
        "records": int(len(labels)),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "prevalence": float(labels.mean()),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10_bin": expected_calibration_error(labels, probabilities, bins=10),
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "predicted_positive_rate": float(predictions.mean()),
    }


def stratified_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    blocks = []
    for value in (0, 1):
        indices = np.flatnonzero(labels == value)
        blocks.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(blocks)


def percentile_interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def bootstrap_classification(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(SEED)
    tracked = {
        name: []
        for name in (
            "auroc",
            "auprc",
            "brier",
            "ece_10_bin",
            "sensitivity",
            "specificity",
            "balanced_accuracy",
        )
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        query = stratified_indices(labels, rng)
        block = classification_metrics(labels[query], probabilities[query], threshold)
        for name in tracked:
            tracked[name].append(float(block[name]))
    return {name: percentile_interval(values) for name, values in tracked.items()}


def cosine_knn_distances(
    train: np.ndarray, *queries: np.ndarray
) -> tuple[np.ndarray, ...]:
    train_norm = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    model = NearestNeighbors(n_neighbors=6, metric="cosine", n_jobs=-1).fit(train_norm)
    outputs = []
    for query in queries:
        query_norm = query / np.maximum(
            np.linalg.norm(query, axis=1, keepdims=True), 1e-12
        )
        distances, _ = model.kneighbors(query_norm)
        outputs.append(distances[:, 0])
    return tuple(outputs)


def mahalanobis_distances(
    train: np.ndarray, *queries: np.ndarray
) -> tuple[np.ndarray, ...]:
    estimator = LedoitWolf().fit(train)
    return tuple(estimator.mahalanobis(query) for query in queries)


def shift_metrics(source_scores: np.ndarray, external_scores: np.ndarray) -> dict[str, float]:
    labels = np.concatenate(
        [np.zeros(len(source_scores), dtype=np.uint8), np.ones(len(external_scores), dtype=np.uint8)]
    )
    scores = np.concatenate([source_scores, external_scores])
    threshold95 = float(np.quantile(source_scores, 0.95))
    return {
        "auroc_external_vs_source": float(roc_auc_score(labels, scores)),
        "source_p95_threshold": threshold95,
        "source_flag_rate": float((source_scores > threshold95).mean()),
        "external_flag_rate": float((external_scores > threshold95).mean()),
        "source_median": float(np.median(source_scores)),
        "external_median": float(np.median(external_scores)),
    }


def bootstrap_shift_auc(source: np.ndarray, external: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 1)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        source_query = rng.choice(len(source), size=len(source), replace=True)
        external_query = rng.choice(len(external), size=len(external), replace=True)
        labels = np.concatenate(
            [np.zeros(len(source_query)), np.ones(len(external_query))]
        )
        scores = np.concatenate([source[source_query], external[external_query]])
        values.append(float(roc_auc_score(labels, scores)))
    return percentile_interval(values)


def conformal_diagnostic(
    calibration_labels: np.ndarray,
    calibration_probabilities: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    alpha: float = 0.10,
) -> dict[str, float | int]:
    q0 = finite_conformal_quantile(
        calibration_probabilities[calibration_labels == 0], alpha
    )
    q1 = finite_conformal_quantile(
        1.0 - calibration_probabilities[calibration_labels == 1], alpha
    )
    sets = mondrian_prediction_sets(probabilities, q0=q0, q1=q1)
    covered = sets[np.arange(len(labels)), labels]
    sizes = sets.sum(axis=1)
    negative = labels == 0
    positive = labels == 1
    return {
        "nominal_coverage": 1.0 - alpha,
        "q0_from_rfmid_validation": q0,
        "q1_from_rfmid_validation": q1,
        "coverage": float(covered.mean()),
        "negative_coverage": float(covered[negative].mean()),
        "positive_coverage": float(covered[positive].mean()),
        "mean_set_size": float(sizes.mean()),
        "empty_set_rate": float((sizes == 0).mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "both_classes_rate": float((sizes == 2).mean()),
    }


def selective_diagnostic(
    source_labels: np.ndarray,
    source_probabilities: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    source_knn: np.ndarray,
    external_knn: np.ndarray,
    source_mahal: np.ndarray,
    external_mahal: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], dict[str, float]]:
    source_confidence = np.abs(source_probabilities - threshold)
    external_confidence = np.abs(probabilities - threshold)
    predictions = (probabilities >= threshold - DECISION_TOLERANCE).astype(np.uint8)
    rows: list[dict[str, float | int | str]] = []
    for target_coverage in (0.90, 0.80, 0.70):
        confidence_threshold = float(np.quantile(source_confidence, 1.0 - target_coverage))
        knn_threshold = float(np.quantile(source_knn, target_coverage))
        mahal_threshold = float(np.quantile(source_mahal, target_coverage))
        selections = {
            "confidence": external_confidence >= confidence_threshold,
            "knn_ood": external_knn <= knn_threshold,
            "mahalanobis_ood": external_mahal <= mahal_threshold,
            "confidence_and_knn": (external_confidence >= confidence_threshold)
            & (external_knn <= knn_threshold),
        }
        for method, accepted in selections.items():
            row: dict[str, float | int | str] = {
                "method": method,
                "source_target_coverage": target_coverage,
            }
            row.update(selective_metrics(labels, predictions, accepted))
            rows.append(row)
    aurc = {
        "confidence": risk_coverage_auc(labels, predictions, external_confidence),
        "knn_reliability": risk_coverage_auc(labels, predictions, -external_knn),
        "mahalanobis_reliability": risk_coverage_auc(labels, predictions, -external_mahal),
    }
    return rows, aurc


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_path = WORKSPACE / "data/features/rfmid_train_dinov2_vits14_geometry_v1.npz"
    validation_path = WORKSPACE / "data/features/rfmid_validation_dinov2_vits14_geometry_v1.npz"
    external_path = WORKSPACE / "data/features/paraguay_dr_dinov2_vits14_geometry_v1.npz"
    checkpoint_path = WORKSPACE / "models/rfmid_dinov2_linear_v1/rfmid_dinov2_linear_v1.pt"

    train = load_npz(train_path)
    validation = load_npz(validation_path)
    external = load_npz(external_path)
    class_order = validation["class_order"].astype(str).tolist()
    label_order = external["label_order"].astype(str).tolist()
    dr_index = class_order.index("DR")
    any_dr_index = label_order.index("any_dr")

    model, bundle = load_rfmid_bundle(checkpoint_path)
    if bundle["class_order"] != class_order:
        raise RuntimeError("rfmid_class_order_mismatch")
    threshold = float(bundle["thresholds"][dr_index])

    def score(embeddings: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(embeddings.astype(np.float32))
        return probabilities_from_embedding(model, tensor, bundle)[:, dr_index].numpy()

    validation_probabilities = score(validation["embeddings"])
    external_probabilities = score(external["embeddings"])
    validation_labels = validation["labels"][:, dr_index].astype(np.uint8)
    external_labels = external["labels"][:, any_dr_index].astype(np.uint8)

    primary = classification_metrics(external_labels, external_probabilities, threshold)
    primary_ci = bootstrap_classification(external_labels, external_probabilities, threshold)
    secondary_targets = {}
    for target_name in ("referable_dr", "proliferative_dr"):
        target_labels = external["labels"][:, label_order.index(target_name)].astype(
            np.uint8
        )
        secondary_targets[target_name] = {
            "metrics": classification_metrics(
                target_labels, external_probabilities, threshold
            ),
            "bootstrap_ci_95": bootstrap_classification(
                target_labels, external_probabilities, threshold
            ),
            "interpretation": (
                "Exploratory only: the frozen RFMiD DR output was not trained or "
                "selected specifically for this target definition."
            ),
        }

    source_knn, external_knn = cosine_knn_distances(
        train["embeddings"], validation["embeddings"], external["embeddings"]
    )
    source_mahal, external_mahal = mahalanobis_distances(
        train["embeddings"], validation["embeddings"], external["embeddings"]
    )
    shift = {
        "knn_cosine": shift_metrics(source_knn, external_knn),
        "mahalanobis_ledoit_wolf": shift_metrics(source_mahal, external_mahal),
    }
    shift["knn_cosine"]["auroc_ci_95"] = bootstrap_shift_auc(source_knn, external_knn)
    shift["mahalanobis_ledoit_wolf"]["auroc_ci_95"] = bootstrap_shift_auc(
        source_mahal, external_mahal
    )

    conformal = conformal_diagnostic(
        validation_labels,
        validation_probabilities,
        external_labels,
        external_probabilities,
    )
    selective_rows, aurc = selective_diagnostic(
        validation_labels,
        validation_probabilities,
        external_labels,
        external_probabilities,
        threshold,
        source_knn,
        external_knn,
        source_mahal,
        external_mahal,
    )

    grades = external["labels"][:, label_order.index("dr_grade")].astype(int)
    grade_rows = []
    for grade in sorted(np.unique(grades)):
        selected = grades == grade
        grade_rows.append(
            {
                "grade": int(grade),
                "records": int(selected.sum()),
                "mean_probability": float(external_probabilities[selected].mean()),
                "median_probability": float(np.median(external_probabilities[selected])),
                "predicted_positive_rate": float(
                    (external_probabilities[selected] >= threshold - DECISION_TOLERANCE).mean()
                ),
            }
        )

    result = {
        "status": "PASS",
        "protocol": "frozen external development validation; no Paraguay fitting or selection",
        "primary_target": "any diabetic retinopathy (grades 1-6 vs grade 0)",
        "primary_metrics": primary,
        "primary_bootstrap_ci_95": primary_ci,
        "secondary_exploratory_targets": secondary_targets,
        "rfmid_validation_reference": classification_metrics(
            validation_labels, validation_probabilities, threshold
        ),
        "shift_detection": shift,
        "conformal_diagnostic": conformal,
        "selective_aurc": aurc,
        "preprocessing_status_counts": {
            status: int((external["preprocessing_status"].astype(str) == status).sum())
            for status in np.unique(external["preprocessing_status"].astype(str))
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path.relative_to(WORKSPACE)),
            "checkpoint_sha256": sha256(checkpoint_path),
            "external_features": str(external_path.relative_to(WORKSPACE)),
            "external_features_sha256": sha256(external_path),
        },
        "protected_test_records_accessed": 0,
        "limitations": [
            "Paraguay is one external hospital and one ZEISS Visucam 500 camera.",
            "Patient identifiers are unavailable, so patient-clustered confidence intervals cannot be computed.",
            "Any-DR labels are not identical to RFMiD's original multi-disease annotation process.",
            "Conformal coverage is diagnostic under external shift, not a guarantee.",
            "This is research evaluation, not clinical validation.",
        ],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_csv(OUTPUT / "grade_stratified_scores.csv", grade_rows)
    write_csv(OUTPUT / "selective_metrics.csv", selective_rows)
    np.savez_compressed(
        OUTPUT / "scores.npz",
        record_ids=external["record_ids"],
        labels=external_labels,
        probabilities=external_probabilities,
        grades=grades,
        knn_ood=external_knn,
        mahalanobis_ood=external_mahal,
        threshold=np.asarray([threshold]),
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
