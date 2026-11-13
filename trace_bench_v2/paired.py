"""Paired hierarchical uncertainty for TRACE-Bench v2 model contrasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import average_precision, auroc


@dataclass(frozen=True)
class PairedObservation:
    label_id: str
    episode_id: str
    query_record_id: str
    target: int
    score_a: float
    score_b: float
    query_group_id: str | None = None


def _metric_value(name: str, labels: list[int], scores: list[float]) -> float | None:
    if name == "average_precision":
        return average_precision(labels, scores)
    if name == "auroc":
        return auroc(labels, scores)
    raise ValueError("paired_metric_must_be_average_precision_or_auroc")


def _validate_grid(observations: Sequence[PairedObservation]) -> None:
    if not observations:
        raise ValueError("paired_observations_must_be_nonempty")
    keys = [
        (item.label_id, item.episode_id, item.query_record_id)
        for item in observations
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("paired_observation_keys_must_be_unique")
    if any(item.target not in (0, 1) for item in observations):
        raise ValueError("paired_targets_must_be_binary")
    values = np.array(
        [[item.score_a, item.score_b] for item in observations], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("paired_scores_must_be_finite")

    by_label_episode: dict[tuple[str, str], set[str]] = {}
    for item in observations:
        by_label_episode.setdefault((item.label_id, item.episode_id), set()).add(
            item.query_record_id
        )
    query_sets = list(by_label_episode.values())
    if any(query_set != query_sets[0] for query_set in query_sets[1:]):
        raise ValueError("all_models_labels_and_support_draws_must_share_query_records")


def _query_unit(item: PairedObservation, use_groups: bool) -> str:
    return item.query_group_id if use_groups else item.query_record_id


def _aggregate_difference(
    observations: Sequence[PairedObservation],
    metric: str,
    label_draw: Sequence[str],
    episode_draws: dict[str, Sequence[str]],
    query_draw: Sequence[str],
    *,
    use_groups: bool,
) -> float | None:
    differences: list[float] = []
    for label_id in label_draw:
        for episode_id in episode_draws[label_id]:
            subset = [
                item
                for item in observations
                if item.label_id == label_id and item.episode_id == episode_id
            ]
            by_unit: dict[str, list[PairedObservation]] = {}
            for item in subset:
                by_unit.setdefault(_query_unit(item, use_groups), []).append(item)
            sampled = [item for unit in query_draw for item in by_unit.get(unit, [])]
            labels = [item.target for item in sampled]
            model_a = _metric_value(metric, labels, [item.score_a for item in sampled])
            model_b = _metric_value(metric, labels, [item.score_b for item in sampled])
            if model_a is not None and model_b is not None:
                differences.append(model_a - model_b)
    return float(np.mean(differences)) if differences else None


def paired_bootstrap_difference(
    observations: Sequence[PairedObservation],
    *,
    metric: str = "average_precision",
    resamples: int = 2000,
    seed: int = 20261112,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Resample labels, support draws and shared query units as paired data."""

    observations = tuple(observations)
    _validate_grid(observations)
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("invalid_resample_or_confidence_setting")
    labels = sorted({item.label_id for item in observations})
    episodes = {
        label_id: sorted(
            {item.episode_id for item in observations if item.label_id == label_id}
        )
        for label_id in labels
    }
    all_have_groups = all(item.query_group_id for item in observations)
    query_units = sorted(
        {_query_unit(item, all_have_groups) for item in observations}
    )
    point = _aggregate_difference(
        observations,
        metric,
        labels,
        episodes,
        query_units,
        use_groups=all_have_groups,
    )
    if point is None:
        raise ValueError("paired_metric_is_undefined_for_all_cells")

    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(resamples):
        label_draw = rng.choice(labels, size=len(labels), replace=True).tolist()
        episode_draws = {
            label_id: rng.choice(
                episodes[label_id], size=len(episodes[label_id]), replace=True
            ).tolist()
            for label_id in set(label_draw)
        }
        query_draw = rng.choice(
            query_units, size=len(query_units), replace=True
        ).tolist()
        difference = _aggregate_difference(
            observations,
            metric,
            label_draw,
            episode_draws,
            query_draw,
            use_groups=all_have_groups,
        )
        if difference is not None:
            draws.append(difference)
    if not draws:
        raise ValueError("all_paired_bootstrap_draws_were_undefined")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(np.asarray(draws), [alpha, 1.0 - alpha])
    return {
        "metric": metric,
        "difference": "model_a_minus_model_b",
        "point_estimate": point,
        "confidence": confidence,
        "interval": [float(lower), float(upper)],
        "interval_excludes_zero": bool(lower > 0 or upper < 0),
        "resamples_requested": resamples,
        "resamples_defined": len(draws),
        "seed": seed,
        "label_count": len(labels),
        "support_draws_per_label": {
            label_id: len(episode_ids) for label_id, episode_ids in episodes.items()
        },
        "query_resampling_unit": "known_group" if all_have_groups else "image_record",
        "limitation": (
            None
            if all_have_groups
            else "Patient/group IDs are incomplete; uncertainty uses image-level query resampling."
        ),
        "interpretation_guard": "An interval containing zero is inconclusive and is not an equivalence claim.",
    }
