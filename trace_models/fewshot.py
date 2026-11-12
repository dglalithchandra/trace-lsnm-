"""Locked few-shot ranking head used by TRACE development evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from trace_protocols import l2_normalize


class TraceFewShotHead:
    """Balanced C=1 logistic ranking head for one binary retinal target."""

    def __init__(self, *, random_state: int = 20261111) -> None:
        self.random_state = int(random_state)
        self.model: LogisticRegression | None = None
        self.embedding_width: int | None = None
        self.positive_support = 0
        self.negative_support = 0

    @staticmethod
    def _matrix(name: str, values: np.ndarray, width: int | None = None) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or not len(matrix):
            raise ValueError(f"{name} must be a non-empty 2D matrix")
        if width is not None and matrix.shape[1] != width:
            raise ValueError(f"{name} embedding width differs: {matrix.shape[1]} != {width}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} contains non-finite values")
        return matrix

    def fit(self, positive: np.ndarray, negative: np.ndarray) -> "TraceFewShotHead":
        positive = self._matrix("positive", positive)
        negative = self._matrix("negative", negative, positive.shape[1])
        support = l2_normalize(np.concatenate((positive, negative), axis=0))
        target = np.concatenate(
            (np.ones(len(positive), dtype=np.uint8), np.zeros(len(negative), dtype=np.uint8))
        )
        self.model = LogisticRegression(
            C=1.0,
            l1_ratio=0.0,
            class_weight="balanced",
            solver="liblinear",
            dual=len(support) < support.shape[1],
            max_iter=500,
            random_state=self.random_state,
        )
        self.model.fit(support, target)
        self.embedding_width = int(support.shape[1])
        self.positive_support = len(positive)
        self.negative_support = len(negative)
        return self

    def score(self, query: np.ndarray) -> np.ndarray:
        if self.model is None or self.embedding_width is None:
            raise RuntimeError("Fit the TRACE few-shot head before scoring")
        query = self._matrix("query", query, self.embedding_width)
        return self.model.decision_function(l2_normalize(query)).astype(np.float32)

    def describe(self) -> dict[str, object]:
        if self.model is None:
            raise RuntimeError("Fit the TRACE few-shot head before describing it")
        return {
            "head": "balanced_l2_logistic",
            "c": 1.0,
            "positive_support": self.positive_support,
            "negative_support": self.negative_support,
            "embedding_width": self.embedding_width,
            "score_interpretation": (
                "Signed ranking margin. Larger values indicate stronger similarity to the "
                "labelled positive support under this fitted head; values are not calibrated probabilities."
            ),
            "research_only": True,
        }
