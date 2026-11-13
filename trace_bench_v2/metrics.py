"""Binary ranking, probability and selective metrics with explicit failures."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


def _undefined(reason: str, labels: np.ndarray) -> dict[str, Any]:
    return {
        "value": None,
        "reason": reason,
        "n_samples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
    }


def _defined(value: float, labels: np.ndarray) -> dict[str, Any]:
    result = _undefined("defined", labels)
    result["value"] = float(value)
    result["reason"] = None
    return result


def _validate_binary(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1 or labels_array.shape != scores_array.shape:
        raise ValueError("labels_and_scores_must_be_aligned_1d_arrays")
    if not np.isin(labels_array, [0, 1]).all():
        raise ValueError("labels_must_be_binary")
    if not np.isfinite(scores_array).all():
        raise ValueError("scores_must_be_finite")
    return labels_array.astype(np.uint8), scores_array


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Average precision using score groups, so ties do not depend on row order."""

    y, values = _validate_binary(labels, scores)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-values, kind="stable")
    sorted_y = y[order]
    sorted_scores = values[order]
    group_ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(y)]
    cumulative_positive = np.cumsum(sorted_y)
    previous_recall = 0.0
    ap = 0.0
    for end in group_ends:
        true_positive = int(cumulative_positive[end - 1])
        recall = true_positive / positives
        precision = true_positive / int(end)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return float(ap)


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    y, values = _validate_binary(labels, scores)
    positive = values[y == 1]
    negative = values[y == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def expected_calibration_error(
    labels: Sequence[int], probabilities: Sequence[float], *, bins: int = 10
) -> float:
    """Equal-width ECE over [0,1], weighted by bin sample count."""

    y, probabilities_array = _validate_binary(labels, probabilities)
    if bins < 2 or np.any((probabilities_array < 0) | (probabilities_array > 1)):
        raise ValueError("ece_requires_probabilities_and_at_least_two_bins")
    assignments = np.minimum((probabilities_array * bins).astype(int), bins - 1)
    value = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            value += selected.mean() * abs(
                probabilities_array[selected].mean() - y[selected].mean()
            )
    return float(value)


def _risk_coverage_auc(labels: np.ndarray, predictions: np.ndarray, confidence: np.ndarray) -> float:
    order = np.argsort(-confidence, kind="stable")
    errors = (predictions[order] != labels[order]).astype(np.float64)
    risk = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    return float(np.trapezoid(risk, coverage))


def evaluate_binary(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    score_type: str,
    calibration_status: str,
    threshold: float | None = None,
    threshold_source_partition: str | None = None,
    ece_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate one label and preserve every undefined-metric reason."""

    y, values = _validate_binary(labels, scores)
    if score_type not in {"ranking_score", "probability"}:
        raise ValueError("score_type_must_be_ranking_score_or_probability")
    if score_type == "ranking_score" and calibration_status != "not_probability":
        raise ValueError("ranking_score_calibration_status_must_be_not_probability")
    if score_type == "probability" and calibration_status == "not_probability":
        raise ValueError("probability_calibration_status_is_required")
    if score_type == "probability" and np.any((values < 0) | (values > 1)):
        raise ValueError("probability_scores_must_be_within_zero_and_one")

    prevalence = float(y.mean()) if len(y) else float("nan")
    one_class = y.sum() == 0 or y.sum() == len(y)
    ranking_reason = "requires_both_evaluation_classes"
    auc_value = auroc(y, values)
    ap_value = average_precision(y, values)
    metrics: dict[str, dict[str, Any]] = {
        "auroc": _undefined(ranking_reason, y) if auc_value is None else _defined(auc_value, y),
        "average_precision": (
            _undefined("requires_positive_evaluation_examples", y)
            if ap_value is None
            else _defined(ap_value, y)
        ),
        "prevalence": _defined(prevalence, y) if len(y) else _undefined("empty_evaluation", y),
        "ap_over_prevalence": (
            _undefined("requires_positive_evaluation_examples", y)
            if ap_value is None or prevalence == 0
            else _defined(ap_value / prevalence, y)
        ),
    }

    if score_type == "probability":
        metrics["brier_score"] = _defined(float(np.mean((values - y) ** 2)), y)
        metrics["ece_equal_width"] = _defined(
            expected_calibration_error(y, values, bins=ece_bins), y
        )
    else:
        metrics["brier_score"] = _undefined("requires_genuine_probability_inputs", y)
        metrics["ece_equal_width"] = _undefined("requires_genuine_probability_inputs", y)

    if threshold is None:
        for name in ("sensitivity", "specificity", "risk_coverage_auc"):
            metrics[name] = _undefined("requires_separately_fitted_threshold", y)
    elif not threshold_source_partition or threshold_source_partition == "evaluation":
        raise ValueError("threshold_must_come_from_a_separate_permitted_partition")
    else:
        predictions = (values >= float(threshold)).astype(np.uint8)
        positive = y == 1
        negative = y == 0
        metrics["sensitivity"] = (
            _defined(float(predictions[positive].mean()), y)
            if positive.any()
            else _undefined("no_positive_evaluation_examples", y)
        )
        metrics["specificity"] = (
            _defined(float((predictions[negative] == 0).mean()), y)
            if negative.any()
            else _undefined("no_negative_evaluation_examples", y)
        )
        confidence = np.abs(values - float(threshold))
        metrics["risk_coverage_auc"] = (
            _defined(_risk_coverage_auc(y, predictions, confidence), y)
            if not one_class
            else _undefined("requires_both_evaluation_classes", y)
        )

    return {
        "score_type": score_type,
        "calibration_status": calibration_status,
        "ece_definition": f"equal_width_{ece_bins}_bins_weighted_absolute_gap",
        "threshold": threshold,
        "threshold_source_partition": threshold_source_partition,
        "metrics": metrics,
    }


def macro_metrics(
    results: Sequence[Mapping[str, Any]],
    *,
    episode_plan_sha256: str | None = None,
    seeds: Sequence[int] = (),
) -> dict[str, Any]:
    """Average defined per-label/episode metrics and retain exclusions."""

    names = sorted(
        set().union(*(result.get("metrics", {}).keys() for result in results))
    )
    output: dict[str, Any] = {}
    for name in names:
        values: list[float] = []
        reasons: Counter[str] = Counter()
        for result in results:
            metric = result.get("metrics", {}).get(name)
            if metric is None:
                reasons["metric_not_reported"] += 1
            elif metric["value"] is None:
                reasons[str(metric["reason"])] += 1
            else:
                values.append(float(metric["value"]))
        output[name] = {
            "value": float(np.mean(values)) if values else None,
            "defined_count": len(values),
            "total_count": len(results),
            "undefined_reasons": dict(sorted(reasons.items())),
        }
    output["_provenance"] = {
        "episode_plan_sha256": episode_plan_sha256,
        "seeds": [int(seed) for seed in seeds],
    }
    return output
