from __future__ import annotations

from collections import Counter

import pytest

from scripts.train_trace_source_balance_v1 import sample_batch_indices


def rows(counts: dict[str, int]) -> list[dict[str, str]]:
    return [
        {"record_id": f"{source}:{index}", "source_id": source}
        for source, count in counts.items()
        for index in range(count)
    ]


def test_natural_sampling_is_deterministic_and_unique_within_batch() -> None:
    pool = rows({"large": 20, "small": 2})
    first = sample_batch_indices(pool, step=7, batch_size=4, sampling="natural", seed=11)
    second = sample_batch_indices(pool, step=7, batch_size=4, sampling="natural", seed=11)
    assert first == second
    assert len(first) == len(set(first)) == 4


def test_source_balanced_sampling_equalises_slots_over_steps() -> None:
    pool = rows({"large": 100, "medium": 10, "small": 2})
    draws: Counter[str] = Counter()
    for step in range(30):
        indices = sample_batch_indices(
            pool, step=step, batch_size=4, sampling="source_balanced", seed=17
        )
        draws.update(pool[index]["source_id"] for index in indices)
    assert draws == {"large": 40, "medium": 40, "small": 40}


def test_source_balanced_sampling_does_not_duplicate_when_avoidable() -> None:
    pool = rows({"a": 4, "b": 4})
    indices = sample_batch_indices(
        pool, step=0, batch_size=4, sampling="source_balanced", seed=3
    )
    assert len(indices) == len(set(indices)) == 4


def test_unknown_sampling_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown sampling mode"):
        sample_batch_indices(rows({"a": 2}), step=0, batch_size=2, sampling="bad", seed=1)
