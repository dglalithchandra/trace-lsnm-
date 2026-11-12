"""Screen pre-scoring v1 test/development candidate pairs using retinal detail."""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.confirm_rfmid2_development_overlap import features

AUDIT = ROOT / "outputs/trace_final_v1_audit"
MANIFEST = ROOT / "data/manifests/trace_analysis_manifest.csv"
ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
OUT = AUDIT / "v1_local_detail_candidates.csv"


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Existing local-detail result must not be overwritten: {OUT}")
    summary = json.loads((AUDIT / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "TEST_INPUT_AUDIT_COARSE_STAGE_ONLY_NO_MODEL_SCORES":
        raise RuntimeError("Coarse audit is not complete")
    with (AUDIT / "coarse_candidates.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["comparison_pool"] == "v1_test_vs_v1_development"]
    if len(rows) != 3220:
        raise RuntimeError(f"Unexpected v1 candidate count: {len(rows)}")
    by_query: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_query[row["query_id"]].append(row)
    selected_by_query = {}
    for query_id, candidates in by_query.items():
        candidates.sort(key=lambda row: int(row["dhash_hamming_256"]))
        selected_by_query[query_id] = [row for i, row in enumerate(candidates) if i == 0 or int(row["dhash_hamming_256"]) <= 20]
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        members = {row["record_id"]: row["archive_member"] for row in csv.DictReader(stream) if row["dataset"] == "RFMiD"}
    if len(by_query) != 644 or sum(map(len, selected_by_query.values())) < 644:
        raise RuntimeError("Not every test query is represented")
    detector = cv2.SIFT_create(nfeatures=1200)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    output_rows = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        @lru_cache(maxsize=128)
        def reference_features(record_id: str) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
            return features(archive.read(members[record_id]), detector)

        for index, (query_id, candidates) in enumerate(by_query.items(), 1):
            q_points, q_desc = features(archive.read(members[query_id]), detector)
            for row in selected_by_query[query_id]:
                r_points, r_desc = reference_features(row["reference_id"])
                good = []
                if q_desc is not None and r_desc is not None:
                    good = [a for a, b in matcher.knnMatch(q_desc, r_desc, k=2) if a.distance < 0.7 * b.distance]
                inliers = 0
                if len(good) >= 8:
                    a = np.float32([q_points[match.queryIdx].pt for match in good])
                    b = np.float32([r_points[match.trainIdx].pt for match in good])
                    _, mask = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
                    inliers = int(mask.sum()) if mask is not None else 0
                output_rows.append({**row, "sift_ratio_matches": len(good), "ransac_inliers": inliers, "inlier_fraction": inliers / max(len(good), 1), "local_geometry_alarm": inliers >= 20 and inliers / max(len(good), 1) >= 0.35})
            if index % 50 == 0 or index == len(by_query):
                print(f"v1 test local-detail queries {index}/{len(by_query)} pairs {len(output_rows)}", flush=True)
    fields = list(output_rows[0])
    with OUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    result = {"status": "LOCAL_DETAIL_SCREEN_COMPLETE_VISUAL_REVIEW_PENDING", "query_images": len(by_query), "screened_pairs": len(output_rows), "local_geometry_alarms": sum(row["local_geometry_alarm"] for row in output_rows), "distinct_alarm_queries": len({row["query_id"] for row in output_rows if row["local_geometry_alarm"]}), "model_scores_computed": 0, "warning": "SIFT/RANSAC alarm is not proof of identical retina; visual review required before scoring"}
    (AUDIT / "v1_local_detail_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
