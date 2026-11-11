"""Build train-only photometric reference candidates for TRACE.

The candidates are deliberately not declared final. They let validation decide
whether a model-source or domain-balanced target is safer and more useful.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader
from trace_preprocessing.reference import build_reference_profile
from trace_preprocessing.types import PhotometricStats


SEED = 20261108


def proxy_group(dataset: str, camera_domain: str) -> str:
    if dataset == "FIVES":
        return "fives_dataset_camera_not_provided"
    return camera_domain


def main() -> None:
    output_dir = WORKSPACE / "outputs" / "photometric_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(
        WORKSPACE / "outputs" / "geometry_audit" / "geometry_audit_records.csv",
        dtype=str,
        keep_default_na=False,
    )
    audit = audit[audit["effective_split"] == "train"]
    audit_by_id = audit.set_index("record_id", drop=False)

    trace_loader = ManifestImageLoader(WORKSPACE, allow_test=False)
    fives_loader = ManifestImageLoader(
        WORKSPACE,
        WORKSPACE / "data" / "manifests" / "fives_manifest.csv",
        allow_test=False,
    )
    fives_manifest = fives_loader.manifest
    primary_column = "is_primary_analysis_record"
    primary_fives = set(
        fives_manifest.loc[
            fives_manifest[primary_column].str.lower().eq("true"), "record_id"
        ].tolist()
    )
    planned: list[tuple[str, str]] = [
        ("trace", record_id) for record_id in trace_loader.record_ids(split="train")
    ]
    planned.extend(
        ("fives", record_id)
        for record_id in fives_loader.record_ids(split="train")
        if record_id in primary_fives
    )
    loaders = {"trace": trace_loader, "fives": fives_loader}
    pipeline = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    )
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    try:
        for index, (loader_name, record_id) in enumerate(planned, start=1):
            loader = loaders[loader_name]
            try:
                image, manifest_row = loader.load_image(record_id)
                result = pipeline.process(image, image_id=record_id)
                if (
                    result.status == ProcessingStatus.REJECT
                    or result.stats_before is None
                    or result.quality_before is None
                ):
                    failures.append(
                        {
                            "record_id": record_id,
                            "reason": "geometry_or_measurement_rejected",
                        }
                    )
                    continue
                if record_id not in audit_by_id.index:
                    raise RuntimeError(f"Missing geometry-audit row: {record_id}")
                audit_row = audit_by_id.loc[record_id]
                dataset = str(manifest_row["dataset"])
                domain = str(audit_row["camera_domain"])
                rows.append(
                    {
                        "record_id": record_id,
                        "dataset": dataset,
                        "effective_split": "train",
                        "proxy_group": proxy_group(dataset, domain),
                        "pipeline_status": result.status.value,
                        "warnings": "|".join(result.warnings),
                        "fundus_detection_confidence": float(
                            result.metadata["fundus_detection_confidence"]
                        ),
                        "normalized_detected_retina_fill_fraction": float(
                            result.metadata["normalized_detected_retina_fill_fraction"]
                        ),
                        **result.stats_before.to_dict(),
                        **{
                            f"quality_{name}": value
                            for name, value in result.quality_before.to_dict().items()
                        },
                    }
                )
            except Exception as error:  # audit every failure, then fail the run below
                failures.append(
                    {"record_id": record_id, "reason": f"{type(error).__name__}:{error}"}
                )
            if index % 100 == 0 or index == len(planned):
                print(f"reference_progress {index}/{len(planned)}", flush=True)
    finally:
        trace_loader.close()
        fives_loader.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "training_photometric_statistics.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "reference_failures.csv", index=False)
    if failures or len(frame) != len(planned):
        raise RuntimeError(
            f"Reference measurement incomplete: {len(frame)}/{len(planned)}; "
            f"failures={len(failures)}"
        )

    stats_fields = list(PhotometricStats.__dataclass_fields__)

    def stats_for(indices: list[int]) -> list[PhotometricStats]:
        return [
            PhotometricStats(
                **{
                    field: float(frame.loc[index, field])
                    for field in stats_fields
                }
            )
            for index in indices
        ]

    def save_candidate(
        *,
        name: str,
        indices: list[int],
        role: str,
        selection: dict[str, object],
    ) -> dict[str, object]:
        profile = build_reference_profile(
            stats_for(indices),
            ["train"] * len(indices),
            profile_name=name,
        )
        payload = profile.to_dict()
        payload.update(
            {
                "candidate_role": role,
                "selection": selection,
                "record_ids": frame.loc[indices, "record_id"].tolist(),
                "test_records_accessed": 0,
            }
        )
        path = output_dir / f"{name}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return {
            "name": name,
            "path": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
            "profile_id": profile.profile_id,
            "records": len(indices),
            "role": role,
        }

    candidates: list[dict[str, object]] = []
    rfmid_indices = frame.index[frame["dataset"] == "RFMiD"].tolist()
    candidates.append(
        save_candidate(
            name="rfmid_model_source_reference",
            indices=rfmid_indices,
            role="candidate expected to best match the disease head's training domain",
            selection={"dataset": "RFMiD", "split": "train", "quality_screen": "none"},
        )
    )
    all_indices = frame.index.tolist()
    candidates.append(
        save_candidate(
            name="pooled_training_reference",
            indices=all_indices,
            role="candidate representing all available permitted training domains",
            selection={
                "split": "train",
                "fives_exact_duplicate_secondaries_excluded": True,
                "quality_screen": "none",
            },
        )
    )

    quality_eligible = frame[
        (frame["pipeline_status"] == ProcessingStatus.ACCEPT.value)
        & (frame["fundus_detection_confidence"] >= 0.75)
        & (frame["normalized_detected_retina_fill_fraction"] >= 0.80)
    ]
    group_counts = quality_eligible["proxy_group"].value_counts().sort_index()
    if len(group_counts) < 2:
        raise RuntimeError("Too few proxy groups for balanced reference")
    samples_per_group = min(150, int(group_counts.min()))
    rng = np.random.default_rng(SEED)
    balanced_indices: list[int] = []
    selected_counts: dict[str, int] = {}
    for group in sorted(group_counts.index):
        available = quality_eligible.index[
            quality_eligible["proxy_group"] == group
        ].to_numpy()
        selected = np.sort(
            rng.choice(available, size=samples_per_group, replace=False)
        )
        balanced_indices.extend(selected.tolist())
        selected_counts[str(group)] = int(len(selected))
    balanced_indices.sort()
    candidates.append(
        save_candidate(
            name="balanced_proxy_quality_reference",
            indices=balanced_indices,
            role=(
                "candidate equally weighting available unverified acquisition-domain proxies"
            ),
            selection={
                "split": "train",
                "proxy_disclaimer": "groups are not verified camera identities",
                "quality_screen": {
                    "pipeline_status": "accept",
                    "minimum_fundus_detection_confidence": 0.75,
                    "minimum_normalized_retina_fill_fraction": 0.80,
                },
                "seed": SEED,
                "samples_per_group": samples_per_group,
                "selected_counts": selected_counts,
            },
        )
    )

    summary = {
        "status": "PASS",
        "planned_train_records": len(planned),
        "measured_train_records": len(frame),
        "datasets": dict(Counter(frame["dataset"])),
        "proxy_groups": dict(Counter(frame["proxy_group"])),
        "quality_eligible_records": int(len(quality_eligible)),
        "quality_eligible_proxy_groups": {
            str(name): int(value) for name, value in group_counts.items()
        },
        "candidates": candidates,
        "candidate_status": "development candidates; validation has not selected a final reference",
        "test_records_accessed": 0,
    }
    (output_dir / "reference_candidates_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
