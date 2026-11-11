"""Extract selected development photographs for visual duplicate review."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tmp" / "rfmid2_overlap_review"


def main() -> None:
    with (ROOT / "data/manifests/rfmid2_development_detail_matches.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        matches = [row for row in csv.DictReader(stream) if row["passes_local_geometry_screen"] == "True"]
    with (ROOT / "data/manifests/trace_analysis_manifest.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        v1 = {row["record_id"]: row for row in csv.DictReader(stream) if row["dataset"] == "RFMiD"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT / "A. RFMiD_All_Classes_Dataset.zip") as source, zipfile.ZipFile(
        ROOT / "data/external/RFMiD2.0/RFMiD2_0.zip"
    ) as outer, zipfile.ZipFile(io.BytesIO(outer.read("Validation_set.zip"))) as validation:
        for row in matches:
            v1_id = row["v1_record_id"]
            if v1[v1_id]["final_split"] not in {"train", "validation"}:
                raise RuntimeError("Protected v1 image requested")
            v2_name = row["v2_member"]
            stem = f"v2_{Path(v2_name).stem}__v1_{v1_id.replace(':', '_')}"
            (OUTPUT / f"{stem}_v2.jpg").write_bytes(validation.read(v2_name))
            (OUTPUT / f"{stem}_v1.png").write_bytes(source.read(v1[v1_id]["archive_member"]))
    print(f"Extracted {len(matches)} development pairs to {OUTPUT}")


if __name__ == "__main__":
    main()
