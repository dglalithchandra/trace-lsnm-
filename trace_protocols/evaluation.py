"""Small, auditable evaluation primitives used by TRACE protocols."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def prototype_margin_scores(
    support_features: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    query_features: np.ndarray,
) -> np.ndarray:
    """Cosine margin to positive versus negative support prototypes."""

    if len(positive_indices) == 0 or len(negative_indices) == 0:
        raise ValueError("Both positive and negative support are required")
    support = l2_normalize(support_features)
    query = l2_normalize(query_features)
    positive = support[positive_indices].mean(axis=0, keepdims=True)
    negative = support[negative_indices].mean(axis=0, keepdims=True)
    positive = l2_normalize(positive)[0]
    negative = l2_normalize(negative)[0]
    return query @ positive - query @ negative


def knn_class_margin_scores(
    support_features: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    query_features: np.ndarray,
    neighbors_per_class: int,
) -> np.ndarray:
    """Mean top-m positive similarity minus mean top-m negative similarity."""

    if neighbors_per_class < 1:
        raise ValueError("neighbors_per_class must be positive")
    if len(positive_indices) < neighbors_per_class:
        raise ValueError("Not enough positive support for requested neighbors")
    if len(negative_indices) < neighbors_per_class:
        raise ValueError("Not enough negative support for requested neighbors")
    support = l2_normalize(support_features)
    query = l2_normalize(query_features)
    positive_similarity = query @ support[positive_indices].T
    negative_similarity = query @ support[negative_indices].T
    positive_top = np.partition(
        positive_similarity,
        kth=positive_similarity.shape[1] - neighbors_per_class,
        axis=1,
    )[:, -neighbors_per_class:]
    negative_top = np.partition(
        negative_similarity,
        kth=negative_similarity.shape[1] - neighbors_per_class,
        axis=1,
    )[:, -neighbors_per_class:]
    return positive_top.mean(axis=1) - negative_top.mean(axis=1)


def logistic_fewshot_scores(
    support_features: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    query_features: np.ndarray,
    c_value: float,
    random_state: int,
) -> np.ndarray:
    """Balanced L2 logistic probe fitted only on the episode support set."""

    if c_value <= 0:
        raise ValueError("c_value must be positive")
    support = l2_normalize(support_features)
    query = l2_normalize(query_features)
    selected = np.concatenate([positive_indices, negative_indices])
    targets = np.concatenate(
        [
            np.ones(len(positive_indices), dtype=np.uint8),
            np.zeros(len(negative_indices), dtype=np.uint8),
        ]
    )
    dual = len(selected) < support.shape[1]
    model = LogisticRegression(
        C=float(c_value),
        l1_ratio=0.0,
        class_weight="balanced",
        solver="liblinear",
        dual=dual,
        max_iter=500,
        random_state=int(random_state),
    )
    model.fit(support[selected], targets)
    return model.decision_function(query).astype(np.float32)

