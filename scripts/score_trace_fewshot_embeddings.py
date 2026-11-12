"""Fit the locked TRACE few-shot head and rank query embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models import TraceFewShotHead


def load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive", type=Path, required=True, help="Positive support .npy")
    parser.add_argument("--negative", type=Path, required=True, help="Negative support .npy")
    parser.add_argument("--query", type=Path, required=True, help="Query embedding .npy")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV")
    parser.add_argument("--query-ids", type=Path, help="Optional text file with one ID per row")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".csv":
        raise ValueError("--output must end in .csv")

    queries = load_matrix(args.query)
    if queries.ndim == 1:
        queries = queries[None, :]
    head = TraceFewShotHead().fit(load_matrix(args.positive), load_matrix(args.negative))
    scores = head.score(queries)
    ids = [str(index) for index in range(len(scores))]
    if args.query_ids:
        ids = [line.strip() for line in args.query_ids.read_text(encoding="utf-8").splitlines()]
        if len(ids) != len(scores) or any(not item for item in ids):
            raise ValueError("--query-ids must contain one non-empty ID per query")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("query_id", "ranking_margin"))
        writer.writerows(zip(ids, scores.tolist(), strict=True))
    print(json.dumps({"output": str(args.output.resolve()), "queries": len(scores), **head.describe()}, indent=2))


if __name__ == "__main__":
    main()
