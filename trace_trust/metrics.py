"""Small, dependency-light trust-layer metrics with explicit edge handling."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    labels = np.asarray(labels, dtype=np.uint8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("ece_requires_aligned_1d_arrays")
    if bins < 2 or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("invalid_ece_input")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        if not selected.any():
            continue
        ece += selected.mean() * abs(
            probabilities[selected].mean() - labels[selected].mean()
        )
    return float(ece)


def finite_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("conformal_scores_must_be_nonempty_1d")
    if not 0 < alpha < 1:
        raise ValueError("alpha_must_be_between_zero_and_one")
    rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(scores))
    return float(np.sort(scores)[rank - 1])


def mondrian_prediction_sets(
    probabilities: np.ndarray, *, q0: float, q1: float
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("invalid_probabilities")
    include_zero = probabilities <= q0
    include_one = (1.0 - probabilities) <= q1
    return np.column_stack([include_zero, include_one])


def selective_metrics(
    labels: np.ndarray, predictions: np.ndarray, accepted: np.ndarray
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=np.uint8)
    accepted = np.asarray(accepted, dtype=bool)
    if labels.shape != predictions.shape or labels.shape != accepted.shape:
        raise ValueError("selective_arrays_must_align")
    kept_labels = labels[accepted]
    kept_predictions = predictions[accepted]
    if len(kept_labels) == 0:
        return {
            "accepted": 0,
            "coverage": 0.0,
            "selective_risk": float("nan"),
            "balanced_accuracy": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
        }
    positive = kept_labels == 1
    negative = kept_labels == 0
    sensitivity = (
        float((kept_predictions[positive] == 1).mean()) if positive.any() else float("nan")
    )
    specificity = (
        float((kept_predictions[negative] == 0).mean()) if negative.any() else float("nan")
    )
    balanced = (
        float((sensitivity + specificity) / 2)
        if np.isfinite(sensitivity) and np.isfinite(specificity)
        else float("nan")
    )
    return {
        "accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "selective_risk": float((kept_predictions != kept_labels).mean()),
        "balanced_accuracy": balanced,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def risk_coverage_auc(
    labels: np.ndarray, predictions: np.ndarray, confidence: np.ndarray
) -> float:
    labels = np.asarray(labels, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=np.uint8)
    confidence = np.asarray(confidence, dtype=np.float64)
    if labels.shape != predictions.shape or labels.shape != confidence.shape:
        raise ValueError("aurc_arrays_must_align")
    order = np.argsort(-confidence, kind="stable")
    errors = (predictions[order] != labels[order]).astype(np.float64)
    cumulative_risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return float(np.trapezoid(cumulative_risk, coverage))
