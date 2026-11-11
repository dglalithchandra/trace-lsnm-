"""Training-split-only smoke run for the TRACE preprocessing structure."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_preprocessing import PreprocessingConfig, TracePreprocessor
from trace_preprocessing.audit import write_audit_record
from trace_preprocessing.loader import ManifestImageLoader
from trace_preprocessing.reference import build_reference_profile


def main() -> None:
    workspace = WORKSPACE
    output_dir = workspace / "outputs" / "preprocessing_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = ManifestImageLoader(workspace, allow_test=False)
    record_ids = loader.record_ids(split="train", dataset="RFMiD", limit=8)
    if len(record_ids) < 8:
        raise RuntimeError("Not enough RFMiD training records for smoke run")

    geometry_pipeline = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    )
    reference_stats = []
    for record_id in record_ids[:6]:
        image, row = loader.load_image(record_id)
        result = geometry_pipeline.process(
            image,
            image_id=record_id,
            metadata={"dataset": row["dataset"], "final_split": row["final_split"]},
        )
        if result.stats_before is None:
            raise RuntimeError(f"Training smoke reference failed: {record_id}")
        reference_stats.append(result.stats_before)

    reference = build_reference_profile(
        reference_stats,
        ["train"] * len(reference_stats),
        profile_name="smoke-only-reference",
    )
    (output_dir / "reference_smoke_only.json").write_text(
        json.dumps(reference.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    statuses: dict[str, int] = {}
    for record_id in record_ids[6:]:
        image, row = loader.load_image(record_id)
        result = TracePreprocessor().process(
            image,
            image_id=record_id,
            reference=reference,
            metadata={"dataset": row["dataset"], "final_split": row["final_split"]},
        )
        statuses[result.status.value] = statuses.get(result.status.value, 0) + 1
        safe_name = record_id.replace(":", "_")
        if result.geometry_only_image is not None:
            Image.fromarray(result.geometry_only_image).save(
                output_dir / f"{safe_name}_geometry.png"
            )
        if result.processed_image is not None:
            Image.fromarray(result.processed_image).save(
                output_dir / f"{safe_name}_processed.png"
            )
        write_audit_record(result, output_dir / f"{safe_name}_audit.json")

    report = {
        "status": "PASS",
        "scope": "training_split_only_structure_smoke_test",
        "reference_records": record_ids[:6],
        "processed_records": record_ids[6:],
        "result_statuses": statuses,
        "warning": "This small reference is for code verification only and is not the final TRACE reference standard.",
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
