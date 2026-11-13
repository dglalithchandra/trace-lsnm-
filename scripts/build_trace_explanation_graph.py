#!/usr/bin/env python3
"""Validate the curated explanation-only graph and pinned source availability."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_explanations import load_graph


DEFAULT_GRAPH = ROOT / "data" / "knowledge_sources" / "curated" / "trace_explanation_graph_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_graph(args.graph)
    sources = []
    for source in graph["sources"]:
        path = ROOT / source["relative_path"]
        actual_sha = None
        if path.is_file():
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append(
            {
                "source_id": source["source_id"],
                "path": source["relative_path"],
                "present": path.is_file(),
                "expected_sha256": source["expected_sha256"],
                "actual_sha256": actual_sha,
                "hash_matches": actual_sha == source["expected_sha256"] if actual_sha else None,
            }
        )
    report = {
        "status": "PASS_GRAPH_SCHEMA",
        "graph_version": graph["graph_version"],
        "mapping_count": len(graph["mappings"]),
        "knowledge_edge_count": len(graph["knowledge_edges"]),
        "curation_queue_count": len(graph["curation_queue"]),
        "sources": sources,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
