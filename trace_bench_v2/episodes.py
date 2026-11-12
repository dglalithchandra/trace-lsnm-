"""Deterministic TRACE-Bench v2 support/query episode construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


class EpisodeConfigurationError(ValueError):
    """Raised when known record relationships violate episode separation."""


@dataclass(frozen=True)
class CandidateRecord:
    record_id: str
    split: str
    labels: Mapping[str, int | None]
    patient_id: str | None = None
    group_id: str | None = None

    def sampling_unit(self) -> str:
        if self.patient_id:
            return f"patient:{self.patient_id}"
        if self.group_id:
            return f"group:{self.group_id}"
        return f"record:{self.record_id}"

    def known_separation_keys(self) -> set[str]:
        keys: set[str] = set()
        if self.patient_id:
            keys.add(f"patient:{self.patient_id}")
        if self.group_id:
            keys.add(f"group:{self.group_id}")
        return keys


@dataclass(frozen=True)
class Episode:
    episode_id: str
    label_id: str
    k: int
    replicate: int
    seed: int
    positive_policy: str
    negative_policy: str
    support_positive_record_ids: tuple[str, ...]
    support_negative_record_ids: tuple[str, ...]
    query_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class UnsupportedSetting:
    label_id: str
    k: int
    reason: str
    available_positive_units: int
    available_negative_units: int
    query_positives: int
    query_negatives: int


@dataclass(frozen=True)
class EpisodePlan:
    version: str
    global_seed: int
    k_values: tuple[int, ...]
    replicates: int
    episodes: tuple[Episode, ...]
    unsupported: tuple[UnsupportedSetting, ...]
    patient_independence: str
    query_resampling_unit: str


def _stable_seed(global_seed: int, label_id: str, k: int, replicate: int) -> int:
    value = f"{global_seed}|{label_id}|{k}|{replicate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % (2**32)


def _unique_by_sampling_unit(records: Sequence[CandidateRecord]) -> list[CandidateRecord]:
    selected: dict[str, CandidateRecord] = {}
    for record in sorted(records, key=lambda item: item.record_id):
        selected.setdefault(record.sampling_unit(), record)
    return list(selected.values())


def _validate_records(records: Sequence[CandidateRecord]) -> None:
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise EpisodeConfigurationError("record IDs must be unique")
    allowed_splits = {"development_train", "evaluation"}
    invalid = sorted({record.split for record in records} - allowed_splits)
    if invalid:
        raise EpisodeConfigurationError(f"unsupported development splits: {invalid}")


def _validate_known_separation(
    support: Sequence[CandidateRecord], query: Sequence[CandidateRecord]
) -> None:
    support_keys = set().union(*(record.known_separation_keys() for record in support))
    query_keys = set().union(*(record.known_separation_keys() for record in query))
    overlap = sorted(support_keys & query_keys)
    if overlap:
        raise EpisodeConfigurationError(
            f"known patient/group overlap between support and query: {overlap[:3]}"
        )


def build_episode_plan(
    records: Sequence[CandidateRecord],
    label_ids: Sequence[str],
    *,
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
    replicates: int = 50,
    global_seed: int = 20261112,
) -> EpisodePlan:
    """Build a fixed balanced-support grid and retain infeasible settings.

    One record per known patient/group sampling unit is eligible for a support
    draw.  Every episode uses all fixed evaluation records as queries.  K=10 and
    K=20 therefore appear only when both support classes have enough units.
    """

    if replicates < 1:
        raise EpisodeConfigurationError("replicates must be positive")
    if not label_ids or len(set(label_ids)) != len(label_ids):
        raise EpisodeConfigurationError("label_ids must be non-empty and unique")
    k_grid = tuple(int(k) for k in k_values)
    if any(k < 1 for k in k_grid) or len(set(k_grid)) != len(k_grid):
        raise EpisodeConfigurationError("k_values must be unique positive integers")

    records = tuple(records)
    _validate_records(records)
    support = tuple(record for record in records if record.split == "development_train")
    query = tuple(record for record in records if record.split == "evaluation")
    if not support or not query:
        raise EpisodeConfigurationError("support and evaluation records are both required")
    if set(record.record_id for record in support) & set(record.record_id for record in query):
        raise EpisodeConfigurationError("support and query record IDs must be disjoint")
    _validate_known_separation(support, query)

    episodes: list[Episode] = []
    unsupported: list[UnsupportedSetting] = []
    query_ids = tuple(record.record_id for record in sorted(query, key=lambda item: item.record_id))
    for label_id in label_ids:
        positive = _unique_by_sampling_unit(
            [record for record in support if record.labels.get(label_id) == 1]
        )
        negative = _unique_by_sampling_unit(
            [record for record in support if record.labels.get(label_id) == 0]
        )
        query_positive = sum(record.labels.get(label_id) == 1 for record in query)
        query_negative = sum(record.labels.get(label_id) == 0 for record in query)
        for k in k_grid:
            reason = None
            if query_positive == 0 or query_negative == 0:
                reason = "evaluation_requires_both_classes"
            elif len(positive) < k:
                reason = "insufficient_positive_support_units"
            elif len(negative) < k:
                reason = "insufficient_negative_support_units"
            if reason:
                unsupported.append(
                    UnsupportedSetting(
                        label_id=label_id,
                        k=k,
                        reason=reason,
                        available_positive_units=len(positive),
                        available_negative_units=len(negative),
                        query_positives=query_positive,
                        query_negatives=query_negative,
                    )
                )
                continue
            for replicate in range(replicates):
                seed = _stable_seed(global_seed, label_id, k, replicate)
                rng = np.random.default_rng(seed)
                positive_indices = np.sort(rng.choice(len(positive), size=k, replace=False))
                negative_indices = np.sort(rng.choice(len(negative), size=k, replace=False))
                episodes.append(
                    Episode(
                        episode_id=(
                            f"trace-bench-v2:{label_id}:k{k}:r{replicate:03d}:s{seed}"
                        ),
                        label_id=label_id,
                        k=k,
                        replicate=replicate,
                        seed=seed,
                        positive_policy="k_positive_distinct_known_units",
                        negative_policy="k_negative_distinct_known_units",
                        support_positive_record_ids=tuple(
                            positive[index].record_id for index in positive_indices
                        ),
                        support_negative_record_ids=tuple(
                            negative[index].record_id for index in negative_indices
                        ),
                        query_record_ids=query_ids,
                    )
                )

    all_have_patients = all(record.patient_id for record in records)
    any_known_groups = any(record.patient_id or record.group_id for record in records)
    return EpisodePlan(
        version="trace-bench-v2",
        global_seed=global_seed,
        k_values=k_grid,
        replicates=replicates,
        episodes=tuple(episodes),
        unsupported=tuple(unsupported),
        patient_independence=(
            "verified_from_supplied_patient_ids"
            if all_have_patients
            else "not_claimed_patient_ids_incomplete"
        ),
        query_resampling_unit=("known_patient_or_group" if any_known_groups else "image_record"),
    )


def episode_plan_hash(plan: EpisodePlan) -> str:
    """Return the SHA-256 of the canonical plan payload."""

    payload = json.dumps(asdict(plan), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
