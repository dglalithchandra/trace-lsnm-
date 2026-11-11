"""Build a development-only RFMiD v1/v2 code mapping and eligibility audit."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/RFMiD2.0/RFMiD2_0.zip"
OUTPUT = ROOT / "data/protocols/trace_bench_v2"
PAPER_V1 = "https://www.mdpi.com/2306-5729/6/2/14"
PAPER_V2 = "https://www.mdpi.com/2306-5729/8/2/29"
SOURCE_DEFINITION_ALIGNED = {"MS", "RT", "CRS"}
PAPER_CSV_CONFLICT = {"EDN"}


def read_labels(outer: zipfile.ZipFile, archive_name: str) -> tuple[list[str], list[dict[str, str]]]:
    with zipfile.ZipFile(io.BytesIO(outer.read(archive_name))) as inner:
        member = next(name for name in inner.namelist() if name.lower().endswith(".csv"))
        rows = list(csv.DictReader(io.StringIO(inner.read(member).decode("cp1252"))))
        return list(rows[0]), rows


def main() -> None:
    v1_order = json.loads((ROOT / "data/labels/rfmid_class_order.json").read_text(encoding="utf-8"))["class_order"]
    fewshot = json.loads((ROOT / "data/protocols/trace_fewshot_v1/protocol.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(ARCHIVE) as outer:
        raw_columns, training = read_labels(outer, "Training_set.zip")
        validation_columns, validation = read_labels(outer, "Validation_set.zip")
    if raw_columns != validation_columns:
        raise ValueError("Training and validation schemas differ")
    normalized = [column.strip() for column in raw_columns]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Normalization collapses distinct v2 fields")
    v1_set = set(v1_order)
    rows: list[dict[str, object]] = []
    for raw, code in zip(raw_columns, normalized, strict=True):
        if code == "ID":
            continue
        train_positives = sum(row[raw].strip() == "1" for row in training)
        query_positives = sum(row[raw].strip() == "1" for row in validation)
        matches_v1 = code in v1_set
        mapping_status = (
            "paper_csv_code_conflict" if code in PAPER_CSV_CONFLICT else
            "source_definition_aligned_provisional" if code in SOURCE_DEFINITION_ALIGNED else
            "same_abbreviation_definition_review_pending" if matches_v1 else "unmapped"
        )
        role = (
            "novel" if code in fewshot["novel_labels"] else
            "base" if code in fewshot["base_labels"] else "other"
        )
        rows.append({
            "v2_raw_header": raw,
            "normalized_code": code,
            "v1_code": code if matches_v1 else "",
            "mapping_status": mapping_status,
            "v1_fewshot_role": role,
            "v2_train_positives": train_positives,
            "v2_validation_positives": query_positives,
            "development_query_eligible_5_positive_minimum": bool(
                mapping_status == "source_definition_aligned_provisional"
                and role == "novel" and query_positives >= 5
            ),
            "v1_paper": PAPER_V1,
            "v2_paper": PAPER_V2,
        })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "rfmid2_v1_label_mapping_draft.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "DRAFT_DEVELOPMENT_FEASIBILITY_ONLY",
        "same_abbreviation_candidates": sum(row["normalized_code"] in v1_set for row in rows),
        "source_definition_aligned_provisional": sorted(SOURCE_DEFINITION_ALIGNED),
        "paper_csv_code_conflicts": sorted(PAPER_CSV_CONFLICT),
        "novel_codes_with_at_least_five_v2_validation_positives": [
            row["normalized_code"] for row in rows if row["development_query_eligible_5_positive_minimum"]
        ],
        "training_records": len(training),
        "validation_records": len(validation),
        "unresolved_fields": [row["v2_raw_header"] for row in rows if row["normalized_code"] in {"", "ME", "OPDM", "EDN"}],
        "protected_test_members_opened": 0,
        "caution": "Matching abbreviations are candidate mappings, not confirmed identical clinical labels.",
    }
    (OUTPUT / "rfmid2_mapping_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
