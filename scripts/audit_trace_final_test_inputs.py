"""Pre-scoring test schema and duplicate-candidate audit; no model inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_trace_final_plan import sha256
from scripts.screen_rfmid2_development_overlap import signature

LOCK = ROOT / "data/protocols/trace_final_v1/plan_lock.json"
MANIFEST = ROOT / "data/manifests/trace_analysis_manifest.csv"
V1_ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
V2_ARCHIVE = ROOT / "data/external/RFMiD2.0/RFMiD2_0.zip"
OUTPUT = ROOT / "outputs/trace_final_v1_audit"


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def top_candidates(query: dict, references: list[dict], pool: str, limit: int = 5) -> list[dict]:
    closest = sorted(references, key=lambda item: hamming(query["dhash"], item["dhash"]))[:limit]
    return [
        {"query_id": query["id"], "query_pool": query["pool"], "reference_id": item["id"], "reference_pool": item["pool"], "comparison_pool": pool,
         "dhash_hamming_256": hamming(query["dhash"], item["dhash"]), "exact_sha256_match": query["sha256"] == item["sha256"],
         "detail_review_priority": "exact" if query["sha256"] == item["sha256"] else ("high" if hamming(query["dhash"], item["dhash"]) <= 20 else "nearest")}
        for item in closest
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-test-for-audit", action="store_true", help="Required explicit flag; this reads test images and labels only for the locked audit")
    args = parser.parse_args()
    if not args.open_test_for_audit:
        raise RuntimeError("Pass --open-test-for-audit only after the analysis plan is committed")
    if OUTPUT.exists():
        raise RuntimeError(f"Audit output exists; do not overwrite an earlier test read: {OUTPUT}")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    for path in (MANIFEST, V1_ARCHIVE, V2_ARCHIVE):
        relative = path.relative_to(ROOT).as_posix()
        if sha256(path) != lock["inputs_sha256"][relative]:
            raise RuntimeError(f"Locked input changed: {relative}")
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        records = [row for row in csv.DictReader(stream) if row["dataset"] == "RFMiD"]
    by_split = Counter(row["final_split"] for row in records)
    if by_split != {"train": 1917, "validation": 639, "test": 644}:
        raise RuntimeError(f"RFMiD split changed: {by_split}")
    if len({row["record_id"] for row in records}) != len(records):
        raise RuntimeError("Duplicate RFMiD record IDs")
    split_groups = defaultdict(set)
    for row in records:
        split_groups[row["split_group_id"]].add(row["final_split"])
    if any(len(splits) != 1 for splits in split_groups.values()):
        raise RuntimeError("Known duplicate group crosses final splits")
    v1 = []
    with zipfile.ZipFile(V1_ARCHIVE) as archive:
        for i, row in enumerate(records, 1):
            raw = archive.read(row["archive_member"])
            computed_code, _ = signature(raw)
            recorded_code = row["perceptual_hash_dhash256"]
            if recorded_code and (len(recorded_code) != 64 or int(recorded_code, 16) != computed_code):
                raise RuntimeError(f"Recorded v1 dHash differs for {row['record_id']}")
            v1.append({"id": row["record_id"], "pool": f"v1_{row['final_split']}", "sha256": hashlib.sha256(raw).hexdigest(), "dhash": computed_code, "member": row["archive_member"]})
            if i % 500 == 0:
                print(f"v1 audit images {i}/{len(records)}", flush=True)
    v2 = []
    v2_schema = {}
    with zipfile.ZipFile(V2_ARCHIVE) as outer:
        for split, member in (("train", "Training_set.zip"), ("validation", "Validation_set.zip"), ("test", "Test_set.zip")):
            with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
                csv_members = [name for name in inner.namelist() if name.lower().endswith(".csv")]
                jpg_members = [name for name in inner.namelist() if name.lower().endswith(".jpg")]
                if len(csv_members) != 1 or not jpg_members:
                    raise RuntimeError(f"Unexpected v2 {split} archive schema")
                text = inner.read(csv_members[0]).decode("cp1252")
                rows = list(csv.DictReader(io.StringIO(text)))
                image_by_id = {Path(name).stem: name for name in jpg_members}
                if len(image_by_id) != len(jpg_members) or len(rows) != len(jpg_members):
                    raise RuntimeError(f"v2 {split} image/label count mismatch")
                header = list(rows[0])
                if split == "test" and any(label not in header for label in ("CRS", "MS", "RT")):
                    raise RuntimeError("Mapped v2 test labels missing")
                counts = {}
                if split == "test":
                    for label in ("CRS", "MS", "RT"):
                        values = [row[label].strip() for row in rows]
                        if any(value not in {"0", "1"} for value in values):
                            raise RuntimeError(f"Nonbinary v2 test label: {label}")
                        counts[label] = sum(value == "1" for value in values)
                v2_schema[split] = {"images": len(jpg_members), "label_rows": len(rows), "headers": header, "mapped_test_positive_counts": counts}
                seen_ids = set()
                for j, row in enumerate(rows, 1):
                    image_id = row["ID"].strip()
                    if image_id in seen_ids or image_id not in image_by_id:
                        raise RuntimeError(f"v2 {split} ID mismatch: {image_id}")
                    seen_ids.add(image_id)
                    name = image_by_id[image_id]
                    raw = inner.read(name)
                    code, _ = signature(raw)
                    v2.append({"id": f"v2_{split}:{image_id}", "pool": f"v2_{split}", "sha256": hashlib.sha256(raw).hexdigest(), "dhash": code, "member": name})
                    if j % 100 == 0:
                        print(f"v2 {split} audit images {j}/{len(rows)}", flush=True)
    v1_dev = [item for item in v1 if item["pool"] != "v1_test"]
    v1_test = [item for item in v1 if item["pool"] == "v1_test"]
    v2_dev = [item for item in v2 if item["pool"] != "v2_test"]
    v2_test = [item for item in v2 if item["pool"] == "v2_test"]
    candidates = []
    for item in v1_test:
        candidates.extend(top_candidates(item, v1_dev, "v1_test_vs_v1_development"))
    for item in v2_test:
        candidates.extend(top_candidates(item, v1, "v2_test_vs_v1_all"))
        candidates.extend(top_candidates(item, v2_dev, "v2_test_vs_v2_development"))
    exact_index = defaultdict(list)
    for item in v1 + v2:
        exact_index[item["sha256"]].append(item)
    exact_groups = [[{"id": item["id"], "pool": item["pool"]} for item in group] for group in exact_index.values() if len(group) > 1 and any(item["pool"].endswith("test") for item in group)]
    OUTPUT.mkdir(parents=True)
    fields = list(candidates[0])
    with (OUTPUT / "coarse_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)
    (OUTPUT / "exact_overlap_groups.json").write_text(json.dumps(exact_groups, indent=2), encoding="utf-8")
    summary = {"status": "TEST_INPUT_AUDIT_COARSE_STAGE_ONLY_NO_MODEL_SCORES", "plan_sha256": lock["inputs_sha256"]["data/protocols/trace_final_v1/ANALYSIS_PLAN.md"], "v1_split_counts": dict(by_split), "v2_schema": v2_schema, "test_images_read": {"v1": len(v1_test), "v2": len(v2_test)}, "exact_overlap_groups_involving_test": len(exact_groups), "coarse_candidate_pairs": len(candidates), "high_priority_pairs": sum(row["detail_review_priority"] in {"exact", "high"} for row in candidates), "next_gate": "SIFT local-detail review and visual adjudication before any model scoring"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "v2_schema"}, indent=2))


if __name__ == "__main__":
    main()
