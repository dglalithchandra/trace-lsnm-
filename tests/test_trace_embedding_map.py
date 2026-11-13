from __future__ import annotations

import numpy as np

from scripts.generate_trace_a1_embedding_map import novel_group


def test_novel_group_marks_no_low_support_label() -> None:
    assert novel_group(np.array([0, 0]), ["A", "B"], ["A", "B"]) == "No eligible low-support label"


def test_novel_group_returns_single_label() -> None:
    assert novel_group(np.array([0, 1]), ["A", "B"], ["A", "B"]) == "B"


def test_novel_group_marks_multilabel_records() -> None:
    assert novel_group(np.array([1, 1]), ["A", "B"], ["A", "B"]) == "Multiple low-support labels"
