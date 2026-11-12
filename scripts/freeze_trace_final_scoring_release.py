"""Freeze a v1 scoring release from completed human image-identity decisions.

Never run this script to guess a decision or repair a source label mapping.
It does not open any test image or compute a model score.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_trace_final_v1_test_features import DECISIONS, LOCK, OUTPUT, PROTOCOL, RELEASE, REQUIRED_REVIEW_PAIRS, required_code_paths
from scripts.freeze_trace_final_plan import sha256

QUEUE = PROTOCOL / "AUDIT_REVIEW_QUEUE.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-after-human-review", action="store_true")
    args = parser.parse_args()
    if not args.freeze_after_human_review:
        raise RuntimeError("A completed human image-identity review is required")
    if DECISIONS.exists() or RELEASE.exists() or (OUTPUT.exists() and any(OUTPUT.iterdir())):
        raise RuntimeError("Decisions, release or final output already exists; refusing to overwrite")
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    pairs = {(row["query_id"], row["reference_id"]) for row in rows}
    if len(rows) != len(REQUIRED_REVIEW_PAIRS) or pairs != REQUIRED_REVIEW_PAIRS:
        raise RuntimeError("Human review queue is missing or has duplicate pairs")
    if any(row["human_decision"] not in {"duplicate", "distinct"} or not row["reviewer"].strip() for row in rows):
        raise RuntimeError("Every flagged pair needs a human decision and reviewer; leave the gate closed if uncertain")
    decisions = [{"query_id": row["query_id"], "reference_id": row["reference_id"], "decision": row["human_decision"], "reviewer": row["reviewer"].strip()} for row in rows]
    excluded = sorted({row["query_id"] for row in decisions if row["decision"] == "duplicate"})
    if len(excluded) >= 644:
        raise RuntimeError("No test query would remain")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for relative, expected in lock["inputs_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Locked input changed: {relative}")
    code_hashes = {relative: sha256(ROOT / relative) for relative in required_code_paths()}
    with DECISIONS.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("query_id", "reference_id", "decision", "reviewer"))
        writer.writeheader()
        writer.writerows(decisions)
    release = {"status": "PASSED", "v1_audit_decision": "ALL_AMBIGUITIES_ADJUDICATED", "plan_lock_sha256": sha256(LOCK), "audit_review_queue_sha256": sha256(QUEUE), "audit_decisions_sha256": sha256(DECISIONS), "code_sha256": code_hashes, "excluded_v1_test_record_ids": excluded, "unresolved_v1_candidate_pairs": [], "note": "Human image-identity decisions recorded before final-test model scoring. This does not establish patient independence or clinical validity."}
    RELEASE.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SCORING_RELEASE_FROZEN", "reviewed_pairs": len(rows), "excluded_test_queries": len(excluded), "release_path": str(RELEASE)}, indent=2))


if __name__ == "__main__":
    main()
