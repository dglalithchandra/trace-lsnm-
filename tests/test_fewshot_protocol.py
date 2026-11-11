from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trace_protocols.fewshot import (
    FeatureSplit,
    FewShotProtocolConfig,
    build_episode_plan,
    classify_labels,
    validate_episode_plan,
)
from trace_protocols.evaluation import (
    knn_class_margin_scores,
    logistic_fewshot_scores,
    prototype_margin_scores,
)


WORKSPACE = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = WORKSPACE / "data" / "protocols" / "trace_fewshot_v1"


def _synthetic_split(split: str, labels: np.ndarray) -> FeatureSplit:
    return FeatureSplit(
        split=split,
        embeddings=np.arange(len(labels) * 4, dtype=np.float32).reshape(len(labels), 4),
        labels=labels.astype(np.uint8),
        record_ids=np.array([f"{split}:{i}" for i in range(len(labels))]),
        class_order=("BASE", "NOVEL"),
    )


def test_count_only_label_roles_and_episode_plan_are_deterministic() -> None:
    config = FewShotProtocolConfig(
        seed=7,
        k_values=(1, 3),
        episode_replicates=4,
        minimum_query_positives=2,
        base_minimum_train_positives=6,
    )
    train_labels = np.zeros((10, 2), dtype=np.uint8)
    train_labels[:7, 0] = 1
    train_labels[:4, 1] = 1
    validation_labels = np.zeros((6, 2), dtype=np.uint8)
    validation_labels[:3, :] = 1
    train = _synthetic_split("train", train_labels)
    validation = _synthetic_split("validation", validation_labels)

    table = classify_labels(
        train.labels, validation.labels, train.class_order, config
    )
    assert table.set_index("label")["role"].to_dict() == {
        "BASE": "base",
        "NOVEL": "novel",
    }
    first = build_episode_plan(train, table, config)
    second = build_episode_plan(train, table, config)
    assert first.equals(second)
    report = validate_episode_plan(train, validation, table, first, config)
    assert report["status"] == "PASS"
    assert report["episode_rows"] == 8
    assert report["test_records_accessed"] == 0

    base = build_episode_plan(train, table, config, roles=("base",))
    base_report = validate_episode_plan(
        train, validation, table, base, config, roles=("base",)
    )
    assert base_report["episode_rows"] == 8


def test_frozen_protocol_does_not_access_test_and_has_paired_grid() -> None:
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["checks"]["status"] == "PASS"
    assert protocol["checks"]["test_records_accessed"] == 0
    assert protocol["checks"]["base_labels"] == 10
    assert protocol["checks"]["novel_labels"] == 11
    assert protocol["checks"]["episode_rows"] == 1650
    assert protocol["config"]["k_values"] == [1, 3, 5]


def test_prototype_margin_prefers_matching_query() -> None:
    support = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = prototype_margin_scores(
        support,
        positive_indices=np.array([0]),
        negative_indices=np.array([1]),
        query_features=query,
    )
    assert scores[0] > 0
    assert scores[1] < 0

    knn_scores = knn_class_margin_scores(
        support,
        positive_indices=np.array([0]),
        negative_indices=np.array([1]),
        query_features=query,
        neighbors_per_class=1,
    )
    assert knn_scores[0] > 0
    assert knn_scores[1] < 0

    logistic_scores = logistic_fewshot_scores(
        support,
        positive_indices=np.array([0]),
        negative_indices=np.array([1]),
        query_features=query,
        c_value=0.1,
        random_state=7,
    )
    assert logistic_scores[0] > logistic_scores[1]

