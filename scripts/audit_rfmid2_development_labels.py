"""Summarize RFMiD 2.0 development labels without opening its test archive."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "external" / "RFMiD2.0" / "RFMiD2_0.zip"


def summarize_split(outer: zipfile.ZipFile, member: str) -> dict[str, object]:
    with zipfile.ZipFile(io.BytesIO(outer.read(member))) as inner:
        csv_members = [name for name in inner.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"Expected one label CSV in {member}")
        text = inner.read(csv_members[0]).decode("cp1252")
        rows = list(csv.DictReader(io.StringIO(text)))
        images = [name for name in inner.namelist() if name.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(rows) != len(images):
            raise ValueError(f"Image/label count mismatch in {member}")
        columns = list(rows[0])
        counts = {
            column: sum(str(row[column]).strip() == "1" for row in rows)
            for column in columns
            if column not in {"ID", "Disease_Risk"}
        }
        return {
            "records": len(rows),
            "image_count": len(images),
            "columns": columns,
            "positive_counts": counts,
            "labels_with_at_least_10_positives": [
                label for label, count in counts.items() if count >= 10
            ],
        }


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as outer:
        result = {
            "training": summarize_split(outer, "Training_set.zip"),
            "validation": summarize_split(outer, "Validation_set.zip"),
            "test_archive_accessed": False,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
