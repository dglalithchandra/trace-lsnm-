"""Check coarse RFMiD v1/v2 overlap candidates using local retinal detail."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT / "data" / "manifests" / "rfmid2_development_overlap_screen.csv"
MANIFEST = ROOT / "data" / "manifests" / "trace_analysis_manifest.csv"
V1_ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
V2_ARCHIVE = ROOT / "data" / "external" / "RFMiD2.0" / "RFMiD2_0.zip"
OUTPUT = ROOT / "data" / "manifests" / "rfmid2_development_detail_matches.csv"


def features(image_bytes: bytes, detector: cv2.SIFT) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not decode candidate photograph")
    height, width = image.shape
    scale = min(1.0, 900 / max(height, width))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return detector.detectAndCompute(image, None)


def main() -> None:
    with SCREEN.open(newline="", encoding="utf-8") as stream:
        candidates = [
            row for row in csv.DictReader(stream)
            if int(row["dhash_hamming_256"]) <= 20
            or float(row["thumbnail_correlation"]) >= 0.98
        ]
    with MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        v1_members = {
            row["record_id"]: row["archive_member"]
            for row in csv.DictReader(stream)
            if row["dataset"] == "RFMiD" and row["final_split"] in {"train", "validation"}
        }
    detector = cv2.SIFT_create(nfeatures=1200)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    v1_cache: dict[str, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
    v2_cache: dict[tuple[str, str], tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
    result: list[dict[str, object]] = []
    with zipfile.ZipFile(V1_ARCHIVE) as v1_zip, zipfile.ZipFile(V2_ARCHIVE) as outer:
        inner = {
            split: zipfile.ZipFile(io.BytesIO(outer.read(member)))
            for split, member in (("train", "Training_set.zip"), ("validation", "Validation_set.zip"))
        }
        for index, row in enumerate(candidates, 1):
            v1_id = row["v1_record_id"]
            v2_key = (row["v2_split"], row["v2_member"])
            if v1_id not in v1_cache:
                v1_cache[v1_id] = features(v1_zip.read(v1_members[v1_id]), detector)
            if v2_key not in v2_cache:
                v2_cache[v2_key] = features(inner[v2_key[0]].read(v2_key[1]), detector)
            left_points, left_desc = v1_cache[v1_id]
            right_points, right_desc = v2_cache[v2_key]
            good = []
            if left_desc is not None and right_desc is not None:
                good = [a for a, b in matcher.knnMatch(left_desc, right_desc, k=2) if a.distance < 0.7 * b.distance]
            inliers = 0
            if len(good) >= 8:
                a = np.float32([left_points[match.queryIdx].pt for match in good])
                b = np.float32([right_points[match.trainIdx].pt for match in good])
                _, mask = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
                inliers = int(mask.sum()) if mask is not None else 0
            result.append({**row, "sift_ratio_matches": len(good), "ransac_inliers": inliers,
                           "candidate_same_retina": inliers >= 20 and inliers / max(len(good), 1) >= 0.35})
            if index % 100 == 0:
                print(f"detail_pairs={index}", flush=True)
        for archive in inner.values():
            archive.close()
    fields = list(result[0]) if result else ["v2_split", "v2_member", "v1_record_id", "v1_split", "dhash_hamming_256", "thumbnail_correlation", "strong_candidate", "sift_ratio_matches", "ransac_inliers", "candidate_same_retina"]
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)
    print(json.dumps({"detail_pairs": len(result), "candidate_same_retina_pairs": sum(row["candidate_same_retina"] for row in result), "distinct_v2_candidates": len({(row["v2_split"], row["v2_member"]) for row in result if row["candidate_same_retina"]}), "protected_test_images_accessed": 0, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
