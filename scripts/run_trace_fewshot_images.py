"""Run the research-only TRACE A1 few-shot workflow on three image folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models import TraceA1Encoder, TraceFewShotHead


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    paths = sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise RuntimeError(f"No supported images found under {folder}")
    return paths


def validate_role_separation(roles: dict[str, list[Path]]) -> dict[str, str]:
    """Reject exact duplicates within or across support/query roles."""

    by_hash: dict[str, list[tuple[str, Path]]] = {}
    hashes: dict[str, str] = {}
    for role, paths in roles.items():
        for path in paths:
            digest = sha256(path)
            hashes[str(path.resolve())] = digest
            by_hash.setdefault(digest, []).append((role, path))
    duplicates = [items for items in by_hash.values() if len(items) > 1]
    if duplicates:
        details = [
            [{"role": role, "path": str(path)} for role, path in group]
            for group in duplicates
        ]
        raise RuntimeError(f"Exact duplicate image detected in few-shot roles: {details}")
    return hashes


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-dir", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    roles = {
        "positive_support": collect_images(args.positive_dir),
        "negative_support": collect_images(args.negative_dir),
        "query": collect_images(args.query_dir),
    }
    image_hashes = validate_role_separation(roles)
    encoder = TraceA1Encoder(WORKSPACE, checkpoint=args.checkpoint)

    audit_rows: list[dict[str, Any]] = []

    def embed(role: str, path: Path) -> np.ndarray | None:
        result = encoder.embed_image(load_rgb(path), image_id=str(path.resolve()))
        audit_rows.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "image_sha256": image_hashes[str(path.resolve())],
                "status": result["status"],
                "preprocessing_status": result["preprocessing_status"],
                "reasons": result["reasons"],
                "warnings": result["warnings"],
            }
        )
        return result["embedding"]

    support: dict[str, list[np.ndarray]] = {"positive_support": [], "negative_support": []}
    for role in support:
        for path in roles[role]:
            value = embed(role, path)
            if value is None:
                raise RuntimeError(f"Support image rejected by preprocessing: {path}")
            support[role].append(value)

    head = TraceFewShotHead().fit(
        np.stack(support["positive_support"]), np.stack(support["negative_support"])
    )
    query_paths: list[Path] = []
    query_embeddings: list[np.ndarray] = []
    rejected: list[Path] = []
    for path in roles["query"]:
        value = embed("query", path)
        if value is None:
            rejected.append(path)
        else:
            query_paths.append(path)
            query_embeddings.append(value)
    scores = head.score(np.stack(query_embeddings)) if query_embeddings else np.empty(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "query_scores.csv"
    score_by_path = dict(zip(query_paths, scores.tolist(), strict=True))
    with score_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("query_path", "status", "ranking_margin"))
        for path in roles["query"]:
            writer.writerow(
                (str(path.resolve()), "scored" if path in score_by_path else "rejected", score_by_path.get(path, ""))
            )

    summary = {
        "status": "RESEARCH_ONLY_COMPLETE",
        "model_id": f"trace_a1_seed_{encoder.config['seed']}",
        "checkpoint_sha256": encoder.checkpoint_sha256,
        "support_head": head.describe(),
        "role_counts": {role: len(paths) for role, paths in roles.items()},
        "query_scored": len(query_paths),
        "query_rejected": len(rejected),
        "scores_file": str(score_path.resolve()),
        "image_audit": audit_rows,
        "interpretation": (
            "Ranking margins are conditional on these support images and are not calibrated "
            "probabilities, diagnostic certainty or clinical decisions."
        ),
        "research_only": True,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "image_audit"}, indent=2))


if __name__ == "__main__":
    main()
