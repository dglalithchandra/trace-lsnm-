import copy
import json
from pathlib import Path

import pytest

from trace_bench_v2 import ExchangeValidationError, load_exchange, validate_exchange
from trace_bench_v2 import (
    CandidateRecord,
    EpisodeConfigurationError,
    PairedObservation,
    build_episode_plan,
    episode_plan_hash,
    evaluate_binary,
    macro_metrics,
    paired_bootstrap_difference,
)


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


def episode_records():
    records = []
    for index in range(12):
        records.append(
            CandidateRecord(
                record_id=f"train:{index}",
                split="development_train",
                labels={"A": int(index < 6)},
                patient_id=f"train_patient:{index}",
                group_id=f"train_group:{index}",
            )
        )
    for index in range(6):
        records.append(
            CandidateRecord(
                record_id=f"query:{index}",
                split="evaluation",
                labels={"A": int(index % 2 == 0)},
                patient_id=f"query_patient:{index}",
                group_id=f"query_group:{index}",
            )
        )
    return records


def test_episode_plan_is_deterministic_and_reports_conditional_k_settings():
    first = build_episode_plan(
        episode_records(), ["A"], k_values=(1, 3, 5, 10, 20), replicates=3, global_seed=9
    )
    second = build_episode_plan(
        episode_records(), ["A"], k_values=(1, 3, 5, 10, 20), replicates=3, global_seed=9
    )
    assert first == second
    assert episode_plan_hash(first) == episode_plan_hash(second)
    assert len(first.episodes) == 9
    assert [(item.k, item.reason) for item in first.unsupported] == [
        (10, "insufficient_positive_support_units"),
        (20, "insufficient_positive_support_units"),
    ]
    for episode in first.episodes:
        assert not set(episode.support_positive_record_ids) & set(episode.query_record_ids)
        assert not set(episode.support_negative_record_ids) & set(episode.query_record_ids)
    assert first.patient_independence == "verified_from_supplied_patient_ids"


def test_episode_plan_rejects_known_group_leakage():
    records = episode_records()
    records[-1] = CandidateRecord(
        record_id=records[-1].record_id,
        split="evaluation",
        labels={"A": 0},
        group_id="train_group:0",
    )
    with pytest.raises(EpisodeConfigurationError, match="patient/group overlap"):
        build_episode_plan(records, ["A"], k_values=(1,), replicates=1)


def test_metrics_match_independently_computed_small_example():
    result = evaluate_binary(
        [1, 0, 1],
        [0.9, 0.8, 0.7],
        score_type="probability",
        calibration_status="uncalibrated_probability",
    )
    metrics = result["metrics"]
    assert metrics["auroc"]["value"] == pytest.approx(0.5)
    assert metrics["average_precision"]["value"] == pytest.approx(5 / 6)
    assert metrics["prevalence"]["value"] == pytest.approx(2 / 3)
    assert metrics["ap_over_prevalence"]["value"] == pytest.approx(1.25)
    assert metrics["brier_score"]["value"] == pytest.approx((0.01 + 0.64 + 0.09) / 3)


def test_non_probability_and_one_class_metrics_retain_failure_reasons():
    result = evaluate_binary(
        [0, 0], [2.0, -1.0], score_type="ranking_score", calibration_status="not_probability"
    )
    metrics = result["metrics"]
    assert metrics["auroc"]["value"] is None
    assert metrics["auroc"]["reason"] == "requires_both_evaluation_classes"
    assert metrics["average_precision"]["reason"] == "requires_positive_evaluation_examples"
    assert metrics["brier_score"]["reason"] == "requires_genuine_probability_inputs"
    assert metrics["ece_equal_width"]["reason"] == "requires_genuine_probability_inputs"
    macro = macro_metrics([result])
    assert macro["auroc"]["defined_count"] == 0
    assert macro["auroc"]["undefined_reasons"] == {"requires_both_evaluation_classes": 1}


def test_threshold_metrics_require_a_separate_partition():
    with pytest.raises(ValueError, match="separate_permitted_partition"):
        evaluate_binary(
            [0, 1],
            [0.1, 0.9],
            score_type="probability",
            calibration_status="uncalibrated_probability",
            threshold=0.5,
            threshold_source_partition="evaluation",
        )


def paired_observations(identical=True):
    observations = []
    for label in ("A", "B"):
        for episode in ("e0", "e1"):
            for index, target in enumerate((0, 1, 0, 1)):
                score = (0.1, 0.8, 0.2, 0.9)[index]
                observations.append(
                    PairedObservation(
                        label_id=label,
                        episode_id=f"{label}:{episode}",
                        query_record_id=f"q{index}",
                        query_group_id=f"g{index}",
                        target=target,
                        score_a=score,
                        score_b=score if identical else 1.0 - score,
                    )
                )
    return observations


def test_paired_identical_predictions_have_exact_zero_difference():
    result = paired_bootstrap_difference(
        paired_observations(), resamples=100, seed=12
    )
    assert result["point_estimate"] == 0.0
    assert result["interval"] == [0.0, 0.0]
    assert result["interval_excludes_zero"] is False
    assert result["query_resampling_unit"] == "known_group"


def test_paired_bootstrap_falls_back_to_image_level_with_a_limit():
    observations = [
        PairedObservation(
            label_id=item.label_id,
            episode_id=item.episode_id,
            query_record_id=item.query_record_id,
            target=item.target,
            score_a=item.score_a,
            score_b=item.score_b,
        )
        for item in paired_observations()
    ]
    result = paired_bootstrap_difference(observations, resamples=20, seed=3)
    assert result["query_resampling_unit"] == "image_record"
    assert "image-level" in result["limitation"]
