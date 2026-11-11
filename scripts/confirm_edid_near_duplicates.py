"""Use local vessel-detail geometry to refine EDID overlap candidates."""

from __future__ import annotations

import csv
import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "data/manifests/edid_near_duplicate_candidates.csv"
EDID_ARCHIVE = ROOT / "data/external/EDID/Original_Dataset.zip"
RFMID_ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
RFMID_MANIFEST = ROOT / "data/manifests/trace_analysis_manifest.csv"
OUTPUT = ROOT / "data/manifests/edid_near_duplicate_detail.csv"
SUMMARY = ROOT / "data/manifests/edid_near_duplicate_detail_summary.json"


def selected_rows(screen: Path, excluded_conflicts: bool = False) -> list[dict[str, str]]:
    with screen.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    edid = [
        row for row in rows
        if row["candidate_source"] == "edid"
        and (
            (int(row["dhash_hamming_256"]) <= 20 and float(row["thumbnail_correlation"]) >= 0.99)
            or (int(row["dhash_hamming_256"]) <= 30 and float(row["thumbnail_correlation"]) >= 0.995)
            or int(row["dhash_hamming_256"]) <= 8
        )
    ]
    rfmid = [] if excluded_conflicts else [row for row in rows if row["candidate_source"] == "rfmid_train"]
    rfmid_sorted_d = sorted(rfmid, key=lambda row: int(row["dhash_hamming_256"]))[:40]
    rfmid_sorted_c = sorted(rfmid, key=lambda row: -float(row["thumbnail_correlation"]))[:40]
    unique = {}
    for row in edid + rfmid_sorted_d + rfmid_sorted_c:
        query = row["query_member"]
        candidate = row["candidate_id"]
        key = (
            "edid", *sorted((query, candidate.removeprefix("edid:")))
        ) if row["candidate_source"] == "edid" else ("rfmid_train", query, candidate)
        unique.setdefault(key, row)
    return list(unique.values())


def features(raw: bytes, detector: cv2.SIFT) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Candidate image decode failed")
    height, width = image.shape
    scale = min(1.0, 850 / max(height, width))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return detector.detectAndCompute(image, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excluded-conflicts", action="store_true")
    args = parser.parse_args()
    screen = ROOT / "data/manifests/edid_excluded_conflict_near_candidates.csv" if args.excluded_conflicts else SCREEN
    output = ROOT / "data/manifests/edid_excluded_conflict_near_detail.csv" if args.excluded_conflicts else OUTPUT
    summary_path = ROOT / "data/manifests/edid_excluded_conflict_near_detail_summary.json" if args.excluded_conflicts else SUMMARY
    candidates = selected_rows(screen, args.excluded_conflicts)
    with RFMID_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rfmid_members = {
            row["record_id"]: row["archive_member"]
            for row in csv.DictReader(stream)
            if row["dataset"] == "RFMiD" and row["final_split"] == "train"
        }
    detector = cv2.SIFT_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    edid_cache: dict[str, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
    rfmid_cache: dict[str, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
    results = []
    with zipfile.ZipFile(EDID_ARCHIVE) as edid_zip, zipfile.ZipFile(RFMID_ARCHIVE) as rfmid_zip:
        for number, row in enumerate(candidates, 1):
            query_member = row["query_member"]
            if query_member not in edid_cache:
                edid_cache[query_member] = features(edid_zip.read(query_member), detector)
            if row["candidate_source"] == "edid":
                other_member = row["candidate_id"].removeprefix("edid:")
                if other_member not in edid_cache:
                    edid_cache[other_member] = features(edid_zip.read(other_member), detector)
                right_points, right_desc = edid_cache[other_member]
            else:
                other_id = row["candidate_id"]
                if other_id not in rfmid_cache:
                    rfmid_cache[other_id] = features(rfmid_zip.read(rfmid_members[other_id]), detector)
                right_points, right_desc = rfmid_cache[other_id]
            left_points, left_desc = edid_cache[query_member]
            good = []
            if left_desc is not None and right_desc is not None:
                for match_pair in matcher.knnMatch(left_desc, right_desc, k=2):
                    if len(match_pair) == 2 and match_pair[0].distance < 0.7 * match_pair[1].distance:
                        good.append(match_pair[0])
            inliers = 0
            if len(good) >= 8:
                left = np.float32([left_points[match.queryIdx].pt for match in good])
                right = np.float32([right_points[match.trainIdx].pt for match in good])
                _, mask = cv2.findHomography(left, right, cv2.RANSAC, 5.0)
                inliers = int(mask.sum()) if mask is not None else 0
            pass_screen = inliers >= 20 and inliers / max(len(good), 1) >= 0.35
            results.append({
                **row,
                "sift_ratio_matches": len(good),
                "ransac_inliers": inliers,
                "ransac_inlier_fraction": round(inliers / max(len(good), 1), 5),
                "passes_local_geometry_screen": pass_screen,
            })
            if number % 100 == 0:
                print(f"Detail-checked {number}/{len(candidates)} pairs", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0]) if results else []
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: -int(row["ransac_inliers"])))
    summary = {
        "status": "LOCAL_GEOMETRY_SCREEN_ONLY",
        "pairs_checked": len(results),
        "by_source": {source: sum(row["candidate_source"] == source for row in results) for source in ("edid", "rfmid_train")},
        "passes_local_geometry_screen": sum(bool(row["passes_local_geometry_screen"]) for row in results),
        "passes_by_source": {source: sum(row["candidate_source"] == source and row["passes_local_geometry_screen"] for row in results) for source in ("edid", "rfmid_train")},
        "protected_test_images_opened": 0,
        "warning": "Local geometry alone does not prove that two photographs show the same retina; visual review follows.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
