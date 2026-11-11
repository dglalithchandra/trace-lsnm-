"""Build robust photometric target ranges from training data only."""

from __future__ import annotations

import hashlib
import json
import numpy as np

from .types import PhotometricStats, ReferenceProfile, ReferenceTarget


REFERENCE_FIELDS = (
    "red_green_ratio",
    "blue_green_ratio",
    "saturation_median",
    "brightness_median",
    "contrast_p95_p05",
    "gamma_to_midgray",
    "illumination_nonuniformity",
)


def build_reference_profile(
    stats: list[PhotometricStats],
    source_splits: list[str],
    *,
    profile_name: str = "trace-training-reference",
) -> ReferenceProfile:
    if not stats:
        raise ValueError("reference_profile_requires_samples")
    if len(stats) != len(source_splits):
        raise ValueError("stats_and_splits_length_mismatch")
    forbidden = sorted({split for split in source_splits if split != "train"})
    if forbidden:
        raise ValueError(f"reference_profile_must_use_train_only:{forbidden}")

    targets: dict[str, ReferenceTarget] = {}
    for name in REFERENCE_FIELDS:
        values = np.asarray([getattr(item, name) for item in stats], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non_finite_reference_values:{name}")
        lower, median, upper = np.percentile(values, [10, 50, 90])
        targets[name] = ReferenceTarget(
            median=float(median),
            lower=float(lower),
            upper=float(upper),
        )

    digest_payload = json.dumps(
        {
            "name": profile_name,
            "sample_count": len(stats),
            "targets": {
                name: [target.lower, target.median, target.upper]
                for name, target in sorted(targets.items())
            },
        },
        sort_keys=True,
    ).encode("utf-8")
    profile_id = f"{profile_name}-{hashlib.sha256(digest_payload).hexdigest()[:12]}"
    return ReferenceProfile(
        profile_id=profile_id,
        source_split="train",
        sample_count=len(stats),
        targets=targets,
    )


def reference_from_dict(payload: dict[str, object]) -> ReferenceProfile:
    targets_payload = payload.get("targets")
    if not isinstance(targets_payload, dict):
        raise ValueError("reference_targets_missing")
    targets = {
        str(name): ReferenceTarget(**value)
        for name, value in targets_payload.items()
        if isinstance(value, dict)
    }
    return ReferenceProfile(
        profile_id=str(payload["profile_id"]),
        source_split=str(payload["source_split"]),
        sample_count=int(payload["sample_count"]),
        status=str(payload.get("status", "development_reference_not_final")),
        targets=targets,
    )
