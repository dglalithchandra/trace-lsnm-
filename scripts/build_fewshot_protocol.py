"""Freeze TRACE's count-only base/novel split and paired few-shot episodes."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_protocols import (
    FewShotProtocolConfig,
    build_episode_plan,
    classify_labels,
    load_feature_split,
    validate_episode_plan,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = FewShotProtocolConfig()
    feature_dir = WORKSPACE / "data" / "features"
    protocol_dir = WORKSPACE / "data" / "protocols" / "trace_fewshot_v1"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    train_path = feature_dir / "rfmid_train_dinov2_vits14_geometry_v1.npz"
    validation_path = feature_dir / "rfmid_validation_dinov2_vits14_geometry_v1.npz"

    train = load_feature_split(train_path, "train")
    validation = load_feature_split(validation_path, "validation")
    if train.class_order != validation.class_order:
        raise RuntimeError("Train/validation class orders differ")

    labels = classify_labels(
        train.labels, validation.labels, train.class_order, config
    )
    episodes = build_episode_plan(train, labels, config)
    checks = validate_episode_plan(train, validation, labels, episodes, config)
    base_tuning_episodes = build_episode_plan(
        train, labels, config, roles=("base",)
    )
    base_checks = validate_episode_plan(
        train,
        validation,
        labels,
        base_tuning_episodes,
        config,
        roles=("base",),
    )

    novel_indices = labels.loc[labels["role"].eq("novel"), "class_index"].to_numpy()
    base_indices = labels.loc[labels["role"].eq("base"), "class_index"].to_numpy()
    novel_positive = train.labels[:, novel_indices].any(axis=1)
    base_positive = train.labels[:, base_indices].any(axis=1)
    adapter_pool = pd.DataFrame(
        {
            "record_id": train.record_ids,
            "eligible_for_base_adapter": (~novel_positive) & base_positive,
            "has_base_positive": base_positive,
            "has_novel_positive": novel_positive,
        }
    )
    adapter_pool["policy"] = (
        "base_positive_required;exclude_any_novel_positive_to_prevent_label_leakage"
    )

    labels.to_csv(protocol_dir / "label_eligibility.csv", index=False)
    episodes.to_csv(protocol_dir / "episode_plan.csv", index=False)
    base_tuning_episodes.to_csv(
        protocol_dir / "base_tuning_episode_plan.csv", index=False
    )
    adapter_pool.to_csv(protocol_dir / "base_adapter_pool.csv", index=False)
    pd.DataFrame({"record_id": validation.record_ids}).to_csv(
        protocol_dir / "fixed_query_records.csv", index=False
    )

    protocol = {
        "version": config.version,
        "status": "FROZEN_DEVELOPMENT_PROTOCOL",
        "config": asdict(config),
        "selection_rule": {
            "primary_query_minimum_positives": config.minimum_query_positives,
            "base": (
                f"all primary-grid-eligible labels with at least "
                f"{config.base_minimum_train_positives} training positives"
            ),
            "novel": (
                f"all primary-grid-eligible labels with {config.maximum_k} to "
                f"{config.base_minimum_train_positives - 1} training positives"
            ),
            "warning": (
                "This is an operational low-support split, not a claim that RFMiD "
                "frequency equals real-world disease rarity."
            ),
        },
        "base_labels": labels.loc[labels["role"].eq("base"), "label"].tolist(),
        "novel_labels": labels.loc[labels["role"].eq("novel"), "label"].tolist(),
        "descriptive_only_labels": labels.loc[
            labels["role"].eq("descriptive_only_low_query_support"), "label"
        ].tolist(),
        "support_protocols": {
            "primary": "K positives plus every available training negative",
            "secondary": (
                f"K positives plus {config.strict_negative_ratio}K negatives"
            ),
            "query": "all fixed validation records for every method",
        },
        "adapter_rule": (
            "Shared adapters train on base-label-positive records only after "
            "excluding every record positive for any novel label."
        ),
        "files": {
            "label_eligibility": "label_eligibility.csv",
            "episode_plan": "episode_plan.csv",
            "base_tuning_episode_plan": "base_tuning_episode_plan.csv",
            "base_adapter_pool": "base_adapter_pool.csv",
            "fixed_query_records": "fixed_query_records.csv",
        },
        "source_hashes": {
            train_path.name: sha256_file(train_path),
            validation_path.name: sha256_file(validation_path),
        },
        "checks": checks,
        "base_tuning_checks": base_checks,
        "final_test_policy": (
            "No test feature file is loaded or required. Final test remains protected."
        ),
    }
    protocol_path = protocol_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(protocol_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (protocol_dir / "artifact_hashes.json").write_text(
        json.dumps(artifact_hashes, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

