"""Propose visually similar EDID query pairs for retinal-detail review.

This only reads EDID's original archive and RFMiD v1 training images. Hashes
and thumbnails are candidate generators, never duplicate decisions.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EDID_ARCHIVE = ROOT / "data/external/EDID/Original_Dataset.zip"
EDID_MANIFEST = ROOT / "data/manifests/edid_exploratory_manifest.csv"
RFMID_ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
RFMID_MANIFEST = ROOT / "data/manifests/trace_analysis_manifest.csv"
OUTPUT = ROOT / "data/manifests/edid_near_duplicate_candidates.csv"
SUMMARY = ROOT / "data/manifests/edid_near_duplicate_screen_summary.json"
QUERY_CLASSES = {"Macular Scar", "Retinitis Pigmentosa", "Disc Edema", "Healthy"}


def signature(raw: bytes) -> tuple[np.ndarray, np.uint64, np.ndarray]:
    with Image.open(io.BytesIO(raw)) as image:
        gray = image.convert("L")
        small = np.asarray(gray.resize((17, 16), Image.Resampling.LANCZOS))
        dhash = np.packbits((small[:, :-1] > small[:, 1:]).ravel()).view(np.uint64)
        square = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        dct = cv2.dct(square)[:8, :8].ravel()
        phash = np.packbits(dct > np.median(dct[1:])).view(np.uint64)[0]
        thumbnail = np.asarray(gray.resize((96, 72), Image.Resampling.BILINEAR), dtype=np.float32)
    thumbnail = thumbnail.ravel()
    thumbnail = (thumbnail - thumbnail.mean()) / max(float(thumbnail.std()), 1e-6)
    return dhash, phash, thumbnail.astype(np.float32)


def load_sources() -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, tuple[np.ndarray, np.uint64, np.ndarray]]]:
    with EDID_MANIFEST.open(newline="", encoding="utf-8") as stream:
        edid = [row for row in csv.DictReader(stream) if row["analysis_eligible"] == "True"]
    with RFMID_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rfmid = [
            row for row in csv.DictReader(stream)
            if row["dataset"] == "RFMiD" and row["final_split"] == "train"
        ]
    if len(edid) != 4370 or len(rfmid) != 1917:
        raise RuntimeError(f"Source count changed: EDID={len(edid)}, RFMiD={len(rfmid)}")
    signatures = {}
    with zipfile.ZipFile(EDID_ARCHIVE) as archive:
        for index, row in enumerate(edid, 1):
            signatures[f"edid:{row['archive_member']}"] = signature(archive.read(row["archive_member"]))
            if index % 1000 == 0:
                print(f"EDID signatures {index}/{len(edid)}", flush=True)
    with zipfile.ZipFile(RFMID_ARCHIVE) as archive:
        for index, row in enumerate(rfmid, 1):
            signatures[row["record_id"]] = signature(archive.read(row["archive_member"]))
            if index % 500 == 0:
                print(f"RFMiD training signatures {index}/{len(rfmid)}", flush=True)
    return edid, rfmid, signatures


def propose(
    queries: list[dict[str, str]],
    corpus: list[tuple[str, str]],
    signatures: dict[str, tuple[np.ndarray, np.uint64, np.ndarray]],
    source: str,
) -> list[dict[str, object]]:
    corpus_ids = [identifier for identifier, _ in corpus]
    corpus_classes = [category for _, category in corpus]
    dhashes = np.stack([signatures[identifier][0] for identifier in corpus_ids])
    phashes = np.asarray([signatures[identifier][1] for identifier in corpus_ids], dtype=np.uint64)
    thumbnails = np.stack([signatures[identifier][2] for identifier in corpus_ids])
    rows: list[dict[str, object]] = []
    for number, query in enumerate(queries, 1):
        query_id = f"edid:{query['archive_member']}"
        qd, qp, qt = signatures[query_id]
        d_distance = np.bitwise_count(np.bitwise_xor(dhashes, qd)).sum(axis=1)
        p_distance = np.bitwise_count(np.bitwise_xor(phashes, qp))
        nearest = set(np.argsort(d_distance)[:8].tolist()) | set(np.argsort(p_distance)[:8].tolist())
        for index in nearest:
            target_id = corpus_ids[index]
            if query_id == target_id:
                continue
            # An already excluded byte-identical group cannot become a query;
            # only identical-query and same-hash corpus copies are skipped here.
            if source == "edid" and query["sha256"] == corpus_classes[index].split("::", 1)[0]:
                continue
            dh = int(d_distance[index])
            ph = int(p_distance[index])
            if dh > 24 and ph > 8:
                continue
            correlation = float(np.dot(qt, thumbnails[index]) / len(qt))
            if correlation < 0.88 and not (dh <= 8 or ph <= 3):
                continue
            rows.append({
                "query_member": query["archive_member"],
                "query_class": query["class_folder"],
                "candidate_source": source,
                "candidate_id": target_id,
                "candidate_class": corpus_classes[index].split("::", 1)[-1],
                "dhash_hamming_256": dh,
                "phash_hamming_64": ph,
                "thumbnail_correlation": round(correlation, 6),
                "needs_detail_review": dh <= 12 or ph <= 5 or correlation >= 0.97,
            })
        if number % 300 == 0:
            print(f"{source} candidates queried {number}/{len(queries)}", flush=True)
    return rows


def main() -> None:
    edid, rfmid, signatures = load_sources()
    queries = [row for row in edid if row["class_folder"] in QUERY_CLASSES]
    if len(queries) != 1411:
        raise RuntimeError(f"Query candidate count changed: {len(queries)}")
    edid_corpus = [
        (f"edid:{row['archive_member']}", f"{row['sha256']}::{row['class_folder']}")
        for row in edid
    ]
    rfmid_corpus = [(row["record_id"], "RFMiD training") for row in rfmid]
    rows = propose(queries, edid_corpus, signatures, "edid")
    rows.extend(propose(queries, rfmid_corpus, signatures, "rfmid_train"))
    rows.sort(key=lambda row: (row["candidate_source"], row["query_member"], row["dhash_hamming_256"], row["phash_hamming_64"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["query_member", "query_class", "candidate_source", "candidate_id", "candidate_class", "dhash_hamming_256", "phash_hamming_64", "thumbnail_correlation", "needs_detail_review"]
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "CANDIDATE_SCREEN_ONLY",
        "edid_corpus": len(edid),
        "rfmid_training_corpus": len(rfmid),
        "edid_query_candidates": len(queries),
        "proposed_pairs": len(rows),
        "pairs_by_source": {name: sum(row["candidate_source"] == name for row in rows) for name in ("edid", "rfmid_train")},
        "detail_review_flagged": sum(bool(row["needs_detail_review"]) for row in rows),
        "protected_test_images_opened": 0,
        "warning": "Coarse photographic similarity does not establish duplicate retinal identity.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
