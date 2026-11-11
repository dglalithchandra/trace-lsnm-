"""Deterministic base/novel few-shot protocol construction.

This module deliberately loads only precomputed training and validation feature
files. The protected RFMiD test feature file is neither required nor permitted.
Dataset frequency is used only as an operational support rule; it must not be
interpreted as the real-world rarity of a disease.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSplit:
    split: str
    embeddings: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    class_order: tuple[str, ...]


@dataclass(frozen=True)
class FewShotProtocolConfig:
    version: str = "trace-fewshot-v1"
    seed: int = 20261111
    k_values: tuple[int, ...] = (1, 3, 5)
    episode_replicates: int = 50
    minimum_query_positives: int = 5
    base_minimum_train_positives: int = 50
    strict_negative_ratio: int = 1

    @property
    def maximum_k(self) -> int:
        return max(self.k_values)


def load_feature_split(path: Path, expected_split: str) -> FeatureSplit:
    """Load a train/validation feature file and reject malformed inputs."""

    if expected_split not in {"train", "validation"}:
        raise ValueError("Only train and validation feature files are permitted")
    with np.load(path, allow_pickle=False) as payload:
        required = {"split", "embeddings", "labels", "record_ids", "class_order"}
        missing = required - set(payload.files)
        if missing:
            raise RuntimeError(f"Missing feature arrays: {sorted(missing)}")
        actual_split = str(payload["split"][0])
        if actual_split != expected_split:
            raise RuntimeError(
                f"Feature split mismatch: {actual_split!r} != {expected_split!r}"
            )
        embeddings = payload["embeddings"].astype(np.float32)
        labels = payload["labels"].astype(np.uint8)
        record_ids = payload["record_ids"].astype(str)
        class_order = tuple(payload["class_order"].astype(str).tolist())

    if embeddings.ndim != 2 or labels.ndim != 2:
        raise RuntimeError("Embeddings and labels must both be matrices")
    if embeddings.shape[0] != labels.shape[0] or len(record_ids) != len(labels):
        raise RuntimeError("Feature, label and record counts do not match")
    if labels.shape[1] != len(class_order):
        raise RuntimeError("Label width does not match class order")
    if len(set(record_ids.tolist())) != len(record_ids):
        raise RuntimeError(f"Duplicate record IDs in {expected_split} features")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Feature file contains non-finite embeddings")
    if not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Labels must be binary")
    return FeatureSplit(
        split=actual_split,
        embeddings=embeddings,
        labels=labels,
        record_ids=record_ids,
        class_order=class_order,
    )


def classify_labels(
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    class_order: tuple[str, ...],
    config: FewShotProtocolConfig,
) -> pd.DataFrame:
    """Assign labels using count-only rules fixed before model comparison."""

    if train_labels.shape[1] != len(class_order):
        raise ValueError("Training labels and class order do not match")
    if validation_labels.shape[1] != len(class_order):
        raise ValueError("Validation labels and class order do not match")

    rows: list[dict[str, object]] = []
    for index, label in enumerate(class_order):
        train_positive = int(train_labels[:, index].sum())
        query_positive = int(validation_labels[:, index].sum())
        total_positive = train_positive + query_positive
        eligible_k = [
            k
            for k in config.k_values
            if train_positive >= k
            and query_positive >= config.minimum_query_positives
            and total_positive >= k + config.minimum_query_positives
        ]

        if train_positive == 0:
            role = "unsupported_no_train_positives"
        elif query_positive == 0:
            role = "unevaluable_no_query_positives"
        elif query_positive < config.minimum_query_positives:
            role = "descriptive_only_low_query_support"
        elif train_positive < config.maximum_k:
            role = "lower_k_only"
        elif train_positive >= config.base_minimum_train_positives:
            role = "base"
        else:
            role = "novel"

        rows.append(
            {
                "class_index": index,
                "label": label,
                "train_positive": train_positive,
                "validation_positive": query_positive,
                "total_development_positive": total_positive,
                "role": role,
                "eligible_k_json": json.dumps(eligible_k),
                "eligible_for_primary_k_grid": eligible_k
                == list(config.k_values),
                "selection_basis": (
                    "count_only_not_model_performance_or_clinical_prevalence"
                ),
            }
        )
    return pd.DataFrame(rows)


def _episode_seed(
    global_seed: int, label: str, k: int, replicate: int
) -> int:
    token = f"{global_seed}|{label}|{k}|{replicate}".encode("utf-8")
    digest = hashlib.sha256(token).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def build_episode_plan(
    train: FeatureSplit,
    label_table: pd.DataFrame,
    config: FewShotProtocolConfig,
    roles: tuple[str, ...] = ("novel",),
) -> pd.DataFrame:
    """Create paired support draws shared by every future model."""

    rows: list[dict[str, object]] = []
    selected = label_table[label_table["role"].isin(roles)]
    for label_row in selected.itertuples(index=False):
        class_index = int(label_row.class_index)
        positive_indices = np.flatnonzero(train.labels[:, class_index] == 1)
        negative_indices = np.flatnonzero(train.labels[:, class_index] == 0)
        eligible_k = set(json.loads(label_row.eligible_k_json))
        for k in config.k_values:
            if k not in eligible_k:
                continue
            strict_negative_count = k * config.strict_negative_ratio
            if len(negative_indices) < strict_negative_count:
                raise RuntimeError(f"Not enough negatives for {label_row.label}")
            for replicate in range(config.episode_replicates):
                seed = _episode_seed(config.seed, label_row.label, k, replicate)
                rng = np.random.default_rng(seed)
                positive_choice = np.sort(
                    rng.choice(positive_indices, size=k, replace=False)
                )
                negative_choice = np.sort(
                    rng.choice(
                        negative_indices, size=strict_negative_count, replace=False
                    )
                )
                rows.append(
                    {
                        "episode_id": (
                            f"{config.version}:{label_row.label}:k{k}:r{replicate:03d}"
                        ),
                        "label": label_row.label,
                        "class_index": class_index,
                        "k": k,
                        "replicate": replicate,
                        "seed": seed,
                        "support_positive_record_ids_json": json.dumps(
                            train.record_ids[positive_choice].tolist()
                        ),
                        "strict_support_negative_record_ids_json": json.dumps(
                            train.record_ids[negative_choice].tolist()
                        ),
                        "abundant_negative_pool_size": int(len(negative_indices)),
                        "query_split": "validation",
                        "query_policy": "all_validation_records_fixed_across_methods",
                    }
                )
    return pd.DataFrame(rows)


def validate_episode_plan(
    train: FeatureSplit,
    validation: FeatureSplit,
    label_table: pd.DataFrame,
    episodes: pd.DataFrame,
    config: FewShotProtocolConfig,
    roles: tuple[str, ...] = ("novel",),
) -> dict[str, object]:
    """Fail closed on split leakage, label errors or non-paired episode grids."""

    if train.class_order != validation.class_order:
        raise AssertionError("Train and validation class orders differ")
    overlap = set(train.record_ids.tolist()) & set(validation.record_ids.tolist())
    if overlap:
        raise AssertionError(f"Train/validation record overlap: {len(overlap)}")
    if episodes["episode_id"].duplicated().any():
        raise AssertionError("Episode IDs are not unique")

    train_index = {record_id: i for i, record_id in enumerate(train.record_ids)}
    expected_rows = 0
    selected_rows = label_table[label_table["role"].isin(roles)]
    for label_row in selected_rows.itertuples(index=False):
        eligible_k = set(json.loads(label_row.eligible_k_json))
        expected_rows += config.episode_replicates * sum(
            k in eligible_k for k in config.k_values
        )
    if len(episodes) != expected_rows:
        raise AssertionError(f"Unexpected episode count: {len(episodes)} != {expected_rows}")

    for row in episodes.itertuples(index=False):
        positives = json.loads(row.support_positive_record_ids_json)
        negatives = json.loads(row.strict_support_negative_record_ids_json)
        if len(positives) != row.k or len(set(positives)) != row.k:
            raise AssertionError(f"Invalid positive support in {row.episode_id}")
        expected_negatives = row.k * config.strict_negative_ratio
        if len(negatives) != expected_negatives or len(set(negatives)) != expected_negatives:
            raise AssertionError(f"Invalid negative support in {row.episode_id}")
        if set(positives) & set(negatives):
            raise AssertionError(f"Positive/negative overlap in {row.episode_id}")
        indices_positive = [train_index[item] for item in positives]
        indices_negative = [train_index[item] for item in negatives]
        if not np.all(train.labels[indices_positive, row.class_index] == 1):
            raise AssertionError(f"Mislabeled positive support in {row.episode_id}")
        if not np.all(train.labels[indices_negative, row.class_index] == 0):
            raise AssertionError(f"Mislabeled negative support in {row.episode_id}")

    counts = (
        episodes.groupby(["label", "k"], as_index=False)
        .size()
        .rename(columns={"size": "replicates"})
    )
    if not (counts["replicates"] == config.episode_replicates).all():
        raise AssertionError("Episode grid is incomplete")
    return {
        "status": "PASS",
        "train_records": len(train.record_ids),
        "validation_query_records": len(validation.record_ids),
        "train_validation_record_overlap": 0,
        "base_labels": int(label_table["role"].eq("base").sum()),
        "novel_labels": int(label_table["role"].eq("novel").sum()),
        "episode_rows": len(episodes),
        "episode_replicates": config.episode_replicates,
        "k_values": list(config.k_values),
        "test_records_accessed": 0,
        "limitation": (
            "RFMiD has no patient identifiers; duplicate-group separation does not "
            "establish patient-independent evaluation. Dataset support counts are "
            "not estimates of clinical rarity."
        ),
    }

