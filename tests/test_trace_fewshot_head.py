from __future__ import annotations

import numpy as np
import pytest

from trace_models import TraceFewShotHead


def test_fewshot_head_ranks_positive_like_query_higher() -> None:
    positive = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    negative = np.array([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = TraceFewShotHead().fit(positive, negative).score(query)
    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_fewshot_head_rejects_embedding_width_mismatch() -> None:
    head = TraceFewShotHead().fit(np.ones((1, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="embedding width"):
        head.score(np.ones((1, 2)))


def test_fewshot_head_calls_scores_margins_not_probabilities() -> None:
    head = TraceFewShotHead().fit(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    assert "not calibrated probabilities" in head.describe()["score_interpretation"]
