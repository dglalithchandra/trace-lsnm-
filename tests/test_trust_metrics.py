from __future__ import annotations

import numpy as np

from trace_trust import (
    expected_calibration_error,
    finite_conformal_quantile,
    mondrian_prediction_sets,
    risk_coverage_auc,
    selective_metrics,
)


def test_ece_is_zero_for_matching_bin_frequencies() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.uint8)
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(labels, probabilities, bins=2) == 0.0


def test_finite_conformal_quantile_uses_higher_finite_sample_rank() -> None:
    scores = np.arange(1, 10, dtype=float) / 10
    assert finite_conformal_quantile(scores, alpha=0.2) == 0.8


def test_mondrian_sets_can_abstain_or_be_ambiguous() -> None:
    sets = mondrian_prediction_sets(np.array([0.1, 0.5, 0.9]), q0=0.2, q1=0.2)
    assert sets.tolist() == [[True, False], [False, False], [False, True]]


def test_selective_metrics_report_coverage_and_class_metrics() -> None:
    result = selective_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 0]),
        np.array([True, False, True, False]),
    )
    assert result["coverage"] == 0.5
    assert result["selective_risk"] == 0.0
    assert result["balanced_accuracy"] == 1.0


def test_risk_coverage_rewards_correct_high_confidence_ordering() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 0])
    good = risk_coverage_auc(labels, predictions, np.array([0.9, 0.1, 0.8, 0.2]))
    bad = risk_coverage_auc(labels, predictions, np.array([0.1, 0.9, 0.2, 0.8]))
    assert good < bad
