from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_trace_fewshot_images import collect_images, validate_role_separation


def make(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_collect_images_is_recursive_and_filters_extensions(tmp_path: Path) -> None:
    first = make(tmp_path / "a.jpg", b"a")
    second = make(tmp_path / "nested" / "b.PNG", b"b")
    make(tmp_path / "ignore.txt", b"c")
    assert collect_images(tmp_path) == [first, second]


def test_exact_duplicate_across_roles_is_rejected(tmp_path: Path) -> None:
    positive = make(tmp_path / "p.jpg", b"same")
    query = make(tmp_path / "q.jpg", b"same")
    with pytest.raises(RuntimeError, match="Exact duplicate"):
        validate_role_separation({"positive_support": [positive], "query": [query]})


def test_distinct_roles_return_a_hash_per_image(tmp_path: Path) -> None:
    positive = make(tmp_path / "p.jpg", b"positive")
    query = make(tmp_path / "q.jpg", b"query")
    hashes = validate_role_separation({"positive_support": [positive], "query": [query]})
    assert len(hashes) == 2
    assert all(len(value) == 64 for value in hashes.values())
