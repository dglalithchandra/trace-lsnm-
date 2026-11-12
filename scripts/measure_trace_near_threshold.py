"""Sensitivity screen for v1 test pairs just below the main SIFT alarm gate."""

from __future__ import annotations

import csv
import json
import zipfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/trace_final_v1_audit"


def main() -> None:
    with (AUDIT / "v1_local_detail_candidates.csv").open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if 12 <= int(row["ransac_inliers"]) < 20 and float(row["inlier_fraction"]) >= 0.7]
    with (ROOT / "data/manifests/trace_analysis_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        members = {row["record_id"]: row["archive_member"] for row in csv.DictReader(stream) if row["dataset"] == "RFMiD"}
    detector = cv2.SIFT_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    results = []
    with zipfile.ZipFile(ROOT / "A. RFMiD_All_Classes_Dataset.zip") as archive:
        @lru_cache(maxsize=256)
        def load(record_id: str) -> np.ndarray:
            image = cv2.imdecode(np.frombuffer(archive.read(members[record_id]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Could not decode {record_id}")
            scale = 900 / max(image.shape)
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        for index, row in enumerate(rows, 1):
            query = load(row["query_id"])
            reference = load(row["reference_id"])
            q_points, q_desc = detector.detectAndCompute(query, None)
            r_points, r_desc = detector.detectAndCompute(reference, None)
            good = [a for a, b in matcher.knnMatch(r_desc, q_desc, k=2) if a.distance < 0.7 * b.distance]
            if len(good) < 8:
                continue
            a = np.float32([r_points[match.queryIdx].pt for match in good])
            b = np.float32([q_points[match.trainIdx].pt for match in good])
            homography, mask = cv2.findHomography(a, b, cv2.RANSAC, 4.0)
            if homography is None:
                continue
            warped = cv2.warpPerspective(reference, homography, (query.shape[1], query.shape[0]))
            overlap = cv2.warpPerspective(np.ones(reference.shape, dtype=np.uint8), homography, (query.shape[1], query.shape[0])) > 0
            valid = overlap & (query > 20) & (warped > 20)
            if valid.sum() < 20000:
                continue
            q_high = query.astype(np.float32) - cv2.GaussianBlur(query.astype(np.float32), (0, 0), 9)
            w_high = warped.astype(np.float32) - cv2.GaussianBlur(warped.astype(np.float32), (0, 0), 9)
            results.append({"query_id": row["query_id"], "reference_id": row["reference_id"], "original_inliers": int(row["ransac_inliers"]), "aligned_inliers": int(mask.sum()), "aligned_grayscale_correlation": float(np.corrcoef(query[valid], warped[valid])[0, 1]), "aligned_highpass_correlation": float(np.corrcoef(q_high[valid], w_high[valid])[0, 1])})
            if index % 20 == 0:
                print(f"near-threshold pairs {index}/{len(rows)}", flush=True)
    (AUDIT / "near_threshold_alignment.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"screened_pairs": len(rows), "aligned_pairs": len(results), "highpass_above_0_5": sum(row["aligned_highpass_correlation"] > 0.5 for row in results), "max_highpass": max(row["aligned_highpass_correlation"] for row in results)}, indent=2))


if __name__ == "__main__":
    main()
