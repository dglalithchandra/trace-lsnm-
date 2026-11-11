"""Render local contact sheets for visual review of EDID geometry-screen pairs."""

from __future__ import annotations

import csv
import argparse
import io
import math
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DETAIL = ROOT / "data/manifests/edid_near_duplicate_detail.csv"
ARCHIVE = ROOT / "data/external/EDID/Original_Dataset.zip"
OUTPUT = ROOT / "tmp/edid_overlap_review"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excluded-conflicts", action="store_true")
    args = parser.parse_args()
    detail = ROOT / "data/manifests/edid_excluded_conflict_near_detail.csv" if args.excluded_conflicts else DETAIL
    output = ROOT / "tmp/edid_excluded_conflict_overlap_review" if args.excluded_conflicts else OUTPUT
    with detail.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row["candidate_source"] == "edid" and row["passes_local_geometry_screen"] == "True"
        ]
    rows.sort(key=lambda row: (row["query_class"] == row["candidate_class"], int(row["ransac_inliers"])))
    output.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height = 600, 245
    with zipfile.ZipFile(ARCHIVE) as archive:
        for page in range(math.ceil(len(rows) / 8)):
            sheet = Image.new("RGB", (tile_width * 2, tile_height * 4), "white")
            draw = ImageDraw.Draw(sheet)
            for slot, row in enumerate(rows[page * 8:(page + 1) * 8]):
                col, line = slot % 2, slot // 2
                left, top = col * tile_width, line * tile_height
                member_a = row["query_member"]
                member_b = row["candidate_id"].removeprefix("edid:")
                for side, member in enumerate((member_a, member_b)):
                    with Image.open(io.BytesIO(archive.read(member))) as image:
                        thumbnail = ImageOps.contain(image.convert("RGB"), (290, 190))
                    sheet.paste(thumbnail, (left + side * 300 + 5, top + 30))
                label = (
                    f"{PurePosixPath(member_a).stem} | {PurePosixPath(member_b).stem} "
                    f"| {row['query_class']} / {row['candidate_class']} "
                    f"| inliers {row['ransac_inliers']}"
                )
                draw.text((left + 5, top + 5), label[:90], fill="black")
                draw.text((left + 5, top + 222), f"dHash {row['dhash_hamming_256']}  corr {row['thumbnail_correlation']}", fill="black")
            path = output / f"review_{page + 1:02d}.jpg"
            sheet.save(path, quality=90)
            print(path)


if __name__ == "__main__":
    main()
