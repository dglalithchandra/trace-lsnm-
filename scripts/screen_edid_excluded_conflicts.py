"""Check eligible EDID queries against images excluded for exact class conflicts."""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.screen_edid_near_duplicates import QUERY_CLASSES, propose, signature

MANIFEST = ROOT / "data/manifests/edid_exploratory_manifest.csv"
ARCHIVE = ROOT / "data/external/EDID/Original_Dataset.zip"
OUTPUT = ROOT / "data/manifests/edid_excluded_conflict_near_candidates.csv"
SUMMARY = ROOT / "data/manifests/edid_excluded_conflict_near_summary.json"


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    queries = [row for row in rows if row["analysis_eligible"] == "True" and row["class_folder"] in QUERY_CLASSES]
    excluded_by_hash = {}
    for row in rows:
        if row["status"] == "cross_class_exact_copy":
            excluded_by_hash.setdefault(row["sha256"], row)
    excluded = list(excluded_by_hash.values())
    if len(queries) != 1411 or len(excluded) != 464:
        raise RuntimeError("Query or excluded-group count changed")
    signatures = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        for number, row in enumerate(queries + excluded, 1):
            member = row["archive_member"]
            signatures[f"edid:{member}"] = signature(archive.read(member))
            if number % 500 == 0:
                print(f"Signatures {number}/{len(queries) + len(excluded)}", flush=True)
    corpus = [
        (f"edid:{row['archive_member']}", f"{row['sha256']}::EXCLUDED_EXACT_CROSS_CLASS")
        for row in excluded
    ]
    candidates = propose(queries, corpus, signatures, "edid")
    fields = ["query_member", "query_class", "candidate_source", "candidate_id", "candidate_class", "dhash_hamming_256", "phash_hamming_64", "thumbnail_correlation", "needs_detail_review"]
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    summary = {
        "status": "CANDIDATE_SCREEN_ONLY",
        "queries": len(queries),
        "excluded_exact_cross_class_groups": len(excluded),
        "proposed_pairs": len(candidates),
        "protected_test_images_opened": 0,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
