from __future__ import annotations

import numpy as np

from scripts.evaluate_trace_a1_rfmid2_k3 import paired_query_interval


def test_identical_predictions_have_zero_paired_interval() -> None:
    truth = np.array([1, 1, 0, 0], dtype=np.uint8)
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
    low, high = paired_query_interval(truth, scores, scores, seed=7)
    assert low == 0.0
    assert high == 0.0
