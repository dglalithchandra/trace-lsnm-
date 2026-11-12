"""Small provenance-preserving cosine retrieval index for TRACE embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from trace_protocols import l2_normalize


class TraceRetrievalIndex:
    def __init__(self, embeddings: np.ndarray, metadata: Sequence[dict[str, Any]]) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or not len(matrix):
            raise ValueError("retrieval embeddings must be a non-empty 2D matrix")
        if len(metadata) != len(matrix):
            raise ValueError("retrieval metadata must align one-to-one with embeddings")
        if not np.isfinite(matrix).all():
            raise ValueError("retrieval embeddings contain non-finite values")
        self.embeddings = l2_normalize(matrix)
        self.metadata = [dict(item) for item in metadata]

    def search(self, queries: np.ndarray, *, top_k: int = 5) -> list[list[dict[str, Any]]]:
        matrix = np.asarray(queries, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query embeddings must be a 2D matrix with matching width")
        if not np.isfinite(matrix).all():
            raise ValueError("query embeddings contain non-finite values")
        if not 1 <= top_k <= len(self.embeddings):
            raise ValueError("top_k must be between 1 and the index size")
        similarities = l2_normalize(matrix) @ self.embeddings.T
        outputs: list[list[dict[str, Any]]] = []
        for row in similarities:
            order = np.argsort(-row, kind="stable")[:top_k]
            outputs.append(
                [
                    {
                        "rank": rank,
                        "cosine_similarity": float(row[index]),
                        **self.metadata[int(index)],
                    }
                    for rank, index in enumerate(order, 1)
                ]
            )
        return outputs
