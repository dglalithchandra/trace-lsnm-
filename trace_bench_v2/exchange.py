"""Validation for the TRACE development feature/prediction exchange.

The JSON Schema documents the serialised contract.  These semantic checks cover
cross-field constraints that JSON Schema cannot express compactly, including
row identity, class alignment and finite model outputs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


class ExchangeValidationError(ValueError):
    """Raised when a development exchange violates the v1 contract."""


_DEVELOPMENT_SPLITS = {
    "development_train",
    "development_tuning",
    "calibration",
    "evaluation",
    "synthetic",
}
_KNOWN_ID_STATUSES = {"verified", "source_reported"}
_MISSING_ID_STATUSES = {"not_available", "unknown"}
_IDENTITY_STATUSES = _KNOWN_ID_STATUSES | _MISSING_ID_STATUSES
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMMIT = re.compile(r"^[a-f0-9]{7,40}$")


def _reject_non_json_constant(value: str) -> None:
    raise ExchangeValidationError(f"non-finite JSON constant is forbidden: {value}")


def load_exchange(path: str | Path) -> dict[str, Any]:
    """Load and validate one JSON exchange without reading referenced arrays."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=_reject_non_json_constant)
    validate_exchange(payload)
    return payload


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExchangeValidationError(f"{location} must be an object")
    return value


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExchangeValidationError(f"{location} must be an array")
    return value


def _check_identity(identity: Mapping[str, Any], location: str) -> None:
    status = identity.get("status")
    value = identity.get("value")
    if status not in _IDENTITY_STATUSES:
        raise ExchangeValidationError(f"{location}.status is invalid")
    if status in _KNOWN_ID_STATUSES and not isinstance(value, str):
        raise ExchangeValidationError(f"{location}.value is required for status {status}")
    if status in _MISSING_ID_STATUSES and value is not None:
        raise ExchangeValidationError(f"{location}.value must be null for status {status}")


def _aligned_values(
    values: Any,
    class_order: list[str],
    location: str,
    *,
    probability: bool,
) -> None:
    entries = _require_list(values, location)
    label_ids = []
    for index, raw_entry in enumerate(entries):
        entry = _require_mapping(raw_entry, f"{location}[{index}]")
        label_ids.append(entry.get("label_id"))
        value = entry.get("value")
        if value is None and location.endswith("targets"):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExchangeValidationError(f"{location}[{index}].value must be numeric")
        if not math.isfinite(float(value)):
            raise ExchangeValidationError(f"{location}[{index}].value must be finite")
        if probability and not 0.0 <= float(value) <= 1.0:
            raise ExchangeValidationError(
                f"{location}[{index}].value must be within [0, 1] for probabilities"
            )
        if location.endswith("targets") and value not in (0, 1):
            raise ExchangeValidationError(f"{location}[{index}].value must be 0, 1 or null")
    if label_ids != class_order:
        raise ExchangeValidationError(
            f"{location} label IDs must exactly match class_order in the same order"
        )


def validate_exchange(payload: Mapping[str, Any]) -> None:
    """Validate semantic invariants of a TRACE development exchange."""

    root = _require_mapping(payload, "exchange")
    if root.get("contract_version") != "trace_dev_exchange_v1":
        raise ExchangeValidationError("contract_version must be trace_dev_exchange_v1")
    if root.get("data_scope") not in {"development", "synthetic"}:
        raise ExchangeValidationError("data_scope must be development or synthetic")
    if root.get("row_order") != "explicit_zero_based_row_index":
        raise ExchangeValidationError("row_order must explicitly use zero-based row_index")

    class_order = _require_list(root.get("class_order"), "class_order")
    if not class_order or any(not isinstance(item, str) or not item for item in class_order):
        raise ExchangeValidationError("class_order must contain non-empty label IDs")
    if len(class_order) != len(set(class_order)):
        raise ExchangeValidationError("class_order contains duplicate label IDs")

    label_definitions = _require_list(root.get("label_definitions"), "label_definitions")
    definition_order = [
        _require_mapping(item, f"label_definitions[{index}]").get("label_id")
        for index, item in enumerate(label_definitions)
    ]
    if definition_order != class_order:
        raise ExchangeValidationError("label_definitions must exactly match class_order")

    model = _require_mapping(root.get("model"), "model")
    if not _SHA256.fullmatch(str(model.get("checkpoint_sha256", ""))):
        raise ExchangeValidationError("model.checkpoint_sha256 must be lowercase SHA-256")
    if not _COMMIT.fullmatch(str(model.get("code_commit", ""))):
        raise ExchangeValidationError("model.code_commit must be a Git commit hash")
    output = _require_mapping(model.get("output"), "model.output")
    output_type = output.get("type")
    calibration_status = output.get("calibration_status")
    if output_type not in {"ranking_score", "probability"}:
        raise ExchangeValidationError("model.output.type must be ranking_score or probability")
    if output_type == "ranking_score" and calibration_status != "not_probability":
        raise ExchangeValidationError("ranking scores must use calibration_status=not_probability")
    if output_type == "probability" and calibration_status == "not_probability":
        raise ExchangeValidationError("probabilities must record their actual calibration status")

    feature_spec = _require_mapping(model.get("feature_spec"), "model.feature_spec")
    feature_width = feature_spec.get("width")
    if isinstance(feature_width, bool) or not isinstance(feature_width, int) or feature_width < 1:
        raise ExchangeValidationError("model.feature_spec.width must be a positive integer")

    preprocessing = _require_mapping(root.get("preprocessing"), "preprocessing")
    if not _SHA256.fullmatch(str(preprocessing.get("config_sha256", ""))):
        raise ExchangeValidationError("preprocessing.config_sha256 must be lowercase SHA-256")

    records = _require_list(root.get("records"), "records")
    record_ids: list[str] = []
    for index, raw_record in enumerate(records):
        record = _require_mapping(raw_record, f"records[{index}]")
        if record.get("row_index") != index:
            raise ExchangeValidationError("row_index values must be contiguous and zero-based")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ExchangeValidationError(f"records[{index}].record_id must be non-empty")
        record_ids.append(record_id)
        if record.get("split") not in _DEVELOPMENT_SPLITS:
            raise ExchangeValidationError(
                f"records[{index}].split is not a permitted development split"
            )
        for field in ("dataset_id", "source_id"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ExchangeValidationError(f"records[{index}].{field} must be non-empty")
        identities = _require_mapping(record.get("identities"), f"records[{index}].identities")
        for name in ("patient", "eye", "group", "device"):
            identity = _require_mapping(
                identities.get(name), f"records[{index}].identities.{name}"
            )
            _check_identity(identity, f"records[{index}].identities.{name}")
        _aligned_values(
            record.get("targets"),
            class_order,
            f"records[{index}].targets",
            probability=False,
        )
        _aligned_values(
            record.get("outputs"),
            class_order,
            f"records[{index}].outputs",
            probability=output_type == "probability",
        )
    if len(record_ids) != len(set(record_ids)):
        raise ExchangeValidationError("record_id values must be unique")

    artifact = _require_mapping(root.get("feature_artifact"), "feature_artifact")
    if not _SHA256.fullmatch(str(artifact.get("sha256", ""))):
        raise ExchangeValidationError("feature_artifact.sha256 must be lowercase SHA-256")
    shape = _require_list(artifact.get("shape"), "feature_artifact.shape")
    if shape != [len(records), feature_width]:
        raise ExchangeValidationError(
            "feature_artifact.shape must equal [record count, feature width]"
        )
