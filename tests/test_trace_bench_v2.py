import copy
import json
from pathlib import Path

import pytest

from trace_bench_v2 import ExchangeValidationError, load_exchange, validate_exchange


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_EXCHANGE = (
    ROOT / "trace_bench_v2" / "examples" / "trace_dev_exchange_v1.synthetic.json"
)


def synthetic_exchange():
    return json.loads(SYNTHETIC_EXCHANGE.read_text(encoding="utf-8"))


def test_synthetic_exchange_is_valid_and_explicitly_development_safe():
    payload = load_exchange(SYNTHETIC_EXCHANGE)
    assert payload["data_scope"] == "synthetic"
    assert all(record["split"] == "synthetic" for record in payload["records"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["records"][1].update(row_index=7), "row_index"),
        (
            lambda value: value["records"][1].update(
                record_id=value["records"][0]["record_id"]
            ),
            "record_id",
        ),
        (
            lambda value: value["records"][0]["outputs"].reverse(),
            "class_order",
        ),
        (
            lambda value: value["records"][0]["outputs"][0].update(value=float("inf")),
            "finite",
        ),
        (
            lambda value: value["records"][0].update(split="protected_final_test"),
            "development split",
        ),
    ],
)
def test_exchange_rejects_ambiguous_or_unsafe_alignment(mutation, message):
    payload = copy.deepcopy(synthetic_exchange())
    mutation(payload)
    with pytest.raises(ExchangeValidationError, match=message):
        validate_exchange(payload)


def test_probability_outputs_are_bounded_and_calibration_state_is_recorded():
    payload = synthetic_exchange()
    payload["model"]["output"] = {
        "type": "probability",
        "calibration_status": "uncalibrated_probability",
    }
    payload["records"][0]["outputs"][0]["value"] = 1.01
    with pytest.raises(ExchangeValidationError, match=r"within \[0, 1\]"):
        validate_exchange(payload)


def test_dataset_registry_keeps_unapproved_sources_out_of_training():
    registry = json.loads(
        (ROOT / "data" / "registries" / "trace_dataset_registry_v2.json").read_text(
            encoding="utf-8"
        )
    )
    entries = {entry["dataset_id"]: entry for entry in registry["datasets"]}
    assert entries["palm"]["training_admission"] == "prohibited_by_current_sprint_role"
    assert entries["edid"]["training_admission"] == "prohibited_by_current_sprint_role"
    assert entries["jsiec_1000"]["training_admission"] == "prohibited_by_current_sprint_role"
    assert entries["rfmid_2_0"]["overlap_risk"]["status"] == "test_blocked"
    assert entries["eyepacs_eyeq"]["access"]["status"] == "metadata_only_images_absent"
    assert entries["aptos_2019"]["access"]["status"] == "not_available_locally"
