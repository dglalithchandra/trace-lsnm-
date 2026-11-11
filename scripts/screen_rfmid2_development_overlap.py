"""Screen development images for cross-release duplicate photographs.

Protected test members are never opened. A low-resolution perceptual screen
produces candidates for subsequent fine-detail confirmation.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
V1_ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
V1_MANIFEST = ROOT / "data" / "manifests" / "trace_analysis_manifest.csv"
V2_ARCHIVE = ROOT / "data" / "external" / "RFMiD2.0" / "RFMiD2_0.zip"
OUTPUT = ROOT / "data" / "manifests" / "rfmid2_development_overlap_screen.csv"


def signature(image_bytes: bytes) -> tuple[int, np.ndarray]:
    with Image.open(io.BytesIO(image_bytes)) as image:
        gray = image.convert("L")
        narrow = np.asarray(gray.resize((17, 16), Image.Resampling.LANCZOS))
        bits = narrow[:, :-1] > narrow[:, 1:]
        code = int.from_bytes(np.packbits(bits.ravel()).tobytes(), "big")
        thumb = np.asarray(gray.resize((128, 96), Image.Resampling.BILINEAR), dtype=np.uint8)
    return code, thumb


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float32).ravel()
    y = right.astype(np.float32).ravel()
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    with V1_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        records = [
            row for row in csv.DictReader(stream)
            if row["dataset"] == "RFMiD" and row["final_split"] in {"train", "validation"}
        ]
    if len(records) != 2556:
        raise ValueError(f"Unexpected RFMiD development split size: {len(records)}")
    v1: list[tuple[str, str, int, np.ndarray]] = []
    with zipfile.ZipFile(V1_ARCHIVE) as archive:
        for index, record in enumerate(records, 1):
            code, thumb = signature(archive.read(record["archive_member"]))
            v1.append((record["record_id"], record["final_split"], code, thumb))
            if index % 500 == 0:
                print(f"v1_development_images={index}", flush=True)

    candidates: list[dict[str, object]] = []
    with zipfile.ZipFile(V2_ARCHIVE) as outer:
        for split, filename in (("train", "Training_set.zip"), ("validation", "Validation_set.zip")):
            with zipfile.ZipFile(io.BytesIO(outer.read(filename))) as inner:
                names = [name for name in inner.namelist() if name.lower().endswith(".jpg")]
                for index, name in enumerate(names, 1):
                    code, thumb = signature(inner.read(name))
                    nearest = sorted(v1, key=lambda item: (code ^ item[2]).bit_count())[:5]
                    for v1_id, v1_split, v1_code, v1_thumb in nearest:
                        distance = (code ^ v1_code).bit_count()
                        corr = correlation(thumb, v1_thumb)
                        if distance <= 64 or corr >= 0.75:
                            candidates.append({
                                "v2_split": split,
                                "v2_member": name,
                                "v1_record_id": v1_id,
                                "v1_split": v1_split,
                                "dhash_hamming_256": distance,
                                "thumbnail_correlation": round(corr, 6),
                                "strong_candidate": distance <= 25 or corr >= 0.97,
                            })
                    if index % 100 == 0:
                        print(f"v2_{split}_images={index}", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["v2_split", "v2_member", "v1_record_id", "v1_split", "dhash_hamming_256", "thumbnail_correlation", "strong_candidate"]
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(candidates, key=lambda row: (row["dhash_hamming_256"], -row["thumbnail_correlation"])))
    print(json.dumps({
        "v1_development_images": len(v1),
        "v2_development_candidates": len(candidates),
        "strong_candidates": sum(bool(row["strong_candidate"]) for row in candidates),
        "output": str(OUTPUT),
        "protected_test_images_accessed": 0,
        "interpretation": "screening candidates only; fine retinal detail must confirm overlap",
    }, indent=2))


if __name__ == "__main__":
    main()
