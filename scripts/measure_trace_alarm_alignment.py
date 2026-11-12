"""Quantify aligned retinal detail for six pre-scoring visual-review pairs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/trace_final_v1_audit"
IMAGES = ROOT / "tmp/trace_final_v1_alarm_images"


def load(record_id: str) -> np.ndarray:
    path = IMAGES / f"{record_id.replace(':', '_')}.png"
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    scale = 900 / max(image.shape)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main() -> None:
    with (AUDIT / "v1_local_detail_candidates.csv").open(newline="", encoding="utf-8") as stream:
        alarms = [row for row in csv.DictReader(stream) if row["local_geometry_alarm"] == "True"]
    detector = cv2.SIFT_create(nfeatures=1500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    results = []
    for row in alarms:
        query = load(row["query_id"])
        reference = load(row["reference_id"])
        q_points, q_desc = detector.detectAndCompute(query, None)
        r_points, r_desc = detector.detectAndCompute(reference, None)
        good = [a for a, b in matcher.knnMatch(r_desc, q_desc, k=2) if a.distance < 0.7 * b.distance]
        a = np.float32([r_points[match.queryIdx].pt for match in good])
        b = np.float32([q_points[match.trainIdx].pt for match in good])
        homography, mask = cv2.findHomography(a, b, cv2.RANSAC, 4.0)
        if homography is None:
            raise RuntimeError(f"No transform for {row['query_id']}")
        warped = cv2.warpPerspective(reference, homography, (query.shape[1], query.shape[0]))
        overlap = cv2.warpPerspective(np.ones(reference.shape, dtype=np.uint8), homography, (query.shape[1], query.shape[0])) > 0
        valid = overlap & (query > 20) & (warped > 20)
        q_high = query.astype(np.float32) - cv2.GaussianBlur(query.astype(np.float32), (0, 0), 9)
        w_high = warped.astype(np.float32) - cv2.GaussianBlur(warped.astype(np.float32), (0, 0), 9)
        high_corr = float(np.corrcoef(q_high[valid], w_high[valid])[0, 1])
        raw_corr = float(np.corrcoef(query[valid], warped[valid])[0, 1])
        results.append({"query_id": row["query_id"], "reference_id": row["reference_id"], "sift_matches": len(good), "ransac_inliers": int(mask.sum()), "valid_overlap_pixels": int(valid.sum()), "aligned_grayscale_correlation": raw_corr, "aligned_highpass_correlation": high_corr})
    (AUDIT / "alarm_alignment.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
