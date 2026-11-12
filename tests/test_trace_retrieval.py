from __future__ import annotations

import numpy as np
import pytest

from trace_models import TraceRetrievalIndex


def test_retrieval_returns_nearest_with_provenance() -> None:
    index = TraceRetrievalIndex(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [{"id": "horizontal"}, {"id": "vertical"}],
    )
    result = index.search(np.array([[0.9, 0.1]], dtype=np.float32), top_k=2)[0]
    assert [item["id"] for item in result] == ["horizontal", "vertical"]
    assert [item["rank"] for item in result] == [1, 2]
    assert result[0]["cosine_similarity"] > result[1]["cosine_similarity"]


def test_retrieval_rejects_unaligned_metadata() -> None:
    with pytest.raises(ValueError, match="one-to-one"):
        TraceRetrievalIndex(np.ones((2, 3)), [{"id": "only-one"}])


def test_retrieval_rejects_top_k_beyond_index() -> None:
    index = TraceRetrievalIndex(np.ones((1, 3)), [{"id": "one"}])
    with pytest.raises(ValueError, match="top_k"):
        index.search(np.ones((1, 3)), top_k=2)
