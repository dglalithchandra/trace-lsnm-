"""Separated calibration/evaluation and label-budget analysis for TRACE."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import evaluate_binary


class CalibrationSeparationError(ValueError):
    """Raised when calibration and evaluation records are not separated."""


@dataclass(frozen=True)
class CalibrationRecord:
    record_id: str
    partition: str
    target: int
    score: float
    group_id: str | None = None

    def sampling_unit(self) -> str:
        return f"group:{self.group_id}" if self.group_id else f"record:{self.record_id}"


def _seed(global_seed: int, budget: int, repeat: int) -> int:
    token = f"{global_seed}|{budget}|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2**32)


def _validate(records: Sequence[CalibrationRecord]) -> tuple[list[CalibrationRecord], list[CalibrationRecord]]:
    if not records:
        raise CalibrationSeparationError("records_must_be_nonempty")
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise CalibrationSeparationError("record_ids_must_be_unique_across_partitions")
    if any(record.partition not in {"calibration", "evaluation"} for record in records):
        raise CalibrationSeparationError("only_calibration_and_evaluation_partitions_are_permitted")
    if any(record.target not in (0, 1) for record in records):
        raise CalibrationSeparationError("targets_must_be_binary")
    if any(not math.isfinite(float(record.score)) for record in records):
        raise CalibrationSeparationError("scores_must_be_finite")
    calibration = [record for record in records if record.partition == "calibration"]
    evaluation = [record for record in records if record.partition == "evaluation"]
    if not calibration or not evaluation:
        raise CalibrationSeparationError("both_partitions_are_required")
    calibration_groups = {record.group_id for record in calibration if record.group_id}
    evaluation_groups = {record.group_id for record in evaluation if record.group_id}
    overlap = sorted(calibration_groups & evaluation_groups)
    if overlap:
        raise CalibrationSeparationError(
            f"known_group_overlap_between_partitions:{overlap[:3]}"
        )
    return calibration, evaluation


def _unique_units(records: Sequence[CalibrationRecord]) -> list[CalibrationRecord]:
    units: dict[str, CalibrationRecord] = {}
    for record in sorted(records, key=lambda item: item.record_id):
        units.setdefault(record.sampling_unit(), record)
    return list(units.values())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_platt(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    if int(targets.sum()) < 2 or int((1 - targets).sum()) < 2:
        raise ValueError("platt_fit_requires_at_least_two_examples_per_class")
    design = np.column_stack([scores, np.ones(len(scores))])
    coefficients = np.zeros(2, dtype=np.float64)
    regularisation = np.diag([1e-4, 1e-6])
    for _ in range(100):
        probabilities = _sigmoid(design @ coefficients)
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-8)
        gradient = design.T @ (probabilities - targets) + regularisation @ coefficients
        hessian = (design.T * weights) @ design + regularisation
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if np.linalg.norm(step) < 1e-9:
            break
    slope, intercept = map(float, coefficients)
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("platt_fit_is_non_finite")
    if slope <= 0:
        raise ValueError("platt_fit_non_positive_slope_would_reverse_ranking")
    return slope, intercept


def _balanced_threshold(scores: np.ndarray, targets: np.ndarray) -> float:
    if targets.min() == targets.max():
        raise ValueError("threshold_fit_requires_both_classes")
    unique = np.unique(scores)
    candidates = np.r_[
        np.nextafter(unique[0], -np.inf),
        (unique[:-1] + unique[1:]) / 2.0,
        np.nextafter(unique[-1], np.inf),
    ]
    best: tuple[float, float] | None = None
    for threshold in candidates:
        predictions = scores >= threshold
        sensitivity = float(predictions[targets == 1].mean())
        specificity = float((~predictions[targets == 0]).mean())
        balanced = (sensitivity + specificity) / 2.0
        candidate = (balanced, -float(threshold))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return -best[1]


def _select_metrics(result: dict[str, object], names: Sequence[str]) -> dict[str, object]:
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    return {name: metrics[name] for name in names}


def run_label_budget_analysis(
    records: Sequence[CalibrationRecord],
    *,
    budgets: Sequence[int] = (0, 5, 10, 20, 50),
    repeats: int = 20,
    global_seed: int = 20261112,
    frozen_source_threshold: float = 0.5,
    source_score_type: str = "probability",
    source_calibration_status: str = "frozen_source_probability",
) -> dict[str, object]:
    """Run repeated target-label budgets without fitting on evaluation labels."""

    calibration, evaluation = _validate(tuple(records))
    if repeats < 1 or any(int(budget) < 0 for budget in budgets):
        raise ValueError("budgets_must_be_nonnegative_and_repeats_positive")
    if len(set(int(budget) for budget in budgets)) != len(tuple(budgets)):
        raise ValueError("budgets_must_be_unique")
    calibration_units = _unique_units(calibration)
    evaluation_labels = np.asarray([record.target for record in evaluation], dtype=np.uint8)
    evaluation_scores = np.asarray([record.score for record in evaluation], dtype=np.float64)
    ranking = evaluate_binary(
        evaluation_labels,
        evaluation_scores,
        score_type=source_score_type,
        calibration_status=source_calibration_status,
    )
    ranking_metrics = _select_metrics(
        ranking, ("auroc", "average_precision", "prevalence", "ap_over_prevalence")
    )

    rows: list[dict[str, object]] = []
    for raw_budget in budgets:
        budget = int(raw_budget)
        for repeat in range(repeats):
            draw_seed = _seed(global_seed, budget, repeat)
            common = {
                "budget": budget,
                "repeat": repeat,
                "seed": draw_seed,
                "evaluation_count": len(evaluation),
                "evaluation_positive": int(evaluation_labels.sum()),
                "evaluation_negative": int(len(evaluation_labels) - evaluation_labels.sum()),
                "ranking": ranking_metrics,
            }
            if budget == 0:
                frozen = evaluate_binary(
                    evaluation_labels,
                    evaluation_scores,
                    score_type=source_score_type,
                    calibration_status=source_calibration_status,
                    threshold=frozen_source_threshold,
                    threshold_source_partition="frozen_source_rule",
                )
                rows.append(
                    {
                        **common,
                        "status": "feasible_no_target_fit",
                        "calibration_count": 0,
                        "calibration_positive": 0,
                        "calibration_negative": 0,
                        "fit": {
                            "calibration_method": "none_frozen_source_rule",
                            "threshold_method": "none_frozen_source_rule",
                            "threshold": frozen_source_threshold,
                        },
                        "threshold": _select_metrics(frozen, ("sensitivity", "specificity")),
                        "calibration": _select_metrics(
                            frozen, ("brier_score", "ece_equal_width")
                        ),
                        "selective": _select_metrics(frozen, ("risk_coverage_auc",)),
                    }
                )
                continue
            if len(calibration_units) < budget:
                rows.append(
                    {
                        **common,
                        "status": "infeasible",
                        "reason": "insufficient_distinct_calibration_units",
                        "calibration_count": len(calibration_units),
                        "calibration_positive": None,
                        "calibration_negative": None,
                    }
                )
                continue

            rng = np.random.default_rng(draw_seed)
            selected_indices = np.sort(
                rng.choice(len(calibration_units), size=budget, replace=False)
            )
            selected = [calibration_units[index] for index in selected_indices]
            calibration_labels = np.asarray(
                [record.target for record in selected], dtype=np.uint8
            )
            calibration_scores = np.asarray(
                [record.score for record in selected], dtype=np.float64
            )
            class_counts = {
                "calibration_count": budget,
                "calibration_positive": int(calibration_labels.sum()),
                "calibration_negative": int(budget - calibration_labels.sum()),
            }
            try:
                threshold = _balanced_threshold(calibration_scores, calibration_labels)
                slope, intercept = _fit_platt(calibration_scores, calibration_labels)
            except (ValueError, np.linalg.LinAlgError) as error:
                rows.append(
                    {
                        **common,
                        **class_counts,
                        "status": "infeasible",
                        "reason": str(error),
                    }
                )
                continue

            threshold_result = evaluate_binary(
                evaluation_labels,
                evaluation_scores,
                score_type=source_score_type,
                calibration_status=source_calibration_status,
                threshold=threshold,
                threshold_source_partition="calibration",
            )
            calibrated_probabilities = _sigmoid(slope * evaluation_scores + intercept)
            calibrated_threshold = float(_sigmoid(np.asarray([slope * threshold + intercept]))[0])
            calibrated_result = evaluate_binary(
                evaluation_labels,
                calibrated_probabilities,
                score_type="probability",
                calibration_status="calibrated_platt",
                threshold=calibrated_threshold,
                threshold_source_partition="calibration",
            )
            rows.append(
                {
                    **common,
                    **class_counts,
                    "status": "feasible",
                    "fit": {
                        "calibration_method": "platt_fixed_l2_newton",
                        "calibration_slope": slope,
                        "calibration_intercept": intercept,
                        "threshold_method": "maximum_balanced_accuracy_on_calibration",
                        "threshold": threshold,
                        "calibrated_threshold": calibrated_threshold,
                    },
                    "threshold": _select_metrics(
                        threshold_result, ("sensitivity", "specificity")
                    ),
                    "calibration": _select_metrics(
                        calibrated_result, ("brier_score", "ece_equal_width")
                    ),
                    "selective": _select_metrics(
                        calibrated_result, ("risk_coverage_auc",)
                    ),
                }
            )

    return {
        "version": "trace-recalibration-dev-v1",
        "budgets": [int(budget) for budget in budgets],
        "repeats": repeats,
        "global_seed": global_seed,
        "calibration_pool_count": len(calibration),
        "calibration_distinct_unit_count": len(calibration_units),
        "evaluation_count": len(evaluation),
        "group_separation": (
            "checked_for_all_supplied_group_ids"
            if any(record.group_id for record in records)
            else "not_checkable_group_ids_absent"
        ),
        "evaluation_labels_used_for_fitting": 0,
        "rows": rows,
    }
