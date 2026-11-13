from __future__ import annotations

import pytest

import torch

from trace_models.dinov2 import DINOV2_VITS14_CHECKPOINT_SHA256
from trace_models.source_balance import (
    A1_CHECKPOINT_SHA256,
    validate_a1_adapter_pack,
    validate_a1_checkpoint_state,
)


def valid_state() -> dict[str, object]:
    return {
        "config": {
            "version": "trace-source-balance-v1",
            "manifest": "data/manifests/trace_ssl_pool_v2.csv",
            "record_count": 1917,
            "sources": ["rfmid_v1_train"],
            "sampling": "natural",
            "updates": 9580,
            "view": "plain_crop_matched",
            "rank": 8,
            "alpha": 8.0,
            "backbone_trainable_parameters": 73_728,
            "seed": 20261111,
        },
        "step": 9580,
        "student": {},
    }


def test_completed_a1_checkpoint_contract_is_accepted() -> None:
    assert validate_a1_checkpoint_state(valid_state())["seed"] == 20261111


def test_expanded_pool_checkpoint_is_rejected() -> None:
    state = valid_state()
    state["config"]["manifest"] = "data/manifests/trace_ssl_pool_v2_fives.csv"
    with pytest.raises(RuntimeError, match="not the selected A1"):
        validate_a1_checkpoint_state(state)


def test_incomplete_checkpoint_is_rejected() -> None:
    state = valid_state()
    state["step"] = 100
    with pytest.raises(RuntimeError, match="incomplete"):
        validate_a1_checkpoint_state(state)


def test_adapter_pack_provenance_is_validated() -> None:
    state = valid_state()
    tensors = {
        f"tensor_{index}": torch.zeros(4608) for index in range(16)
    }
    pack = {
        "format": "trace-a1-lora-adapter-v1",
        "source_checkpoint_sha256": A1_CHECKPOINT_SHA256[20261111],
        "base_checkpoint_sha256": DINOV2_VITS14_CHECKPOINT_SHA256,
        "config": state["config"],
        "lora_state": tensors,
    }
    assert validate_a1_adapter_pack(pack)["seed"] == 20261111
    pack["source_checkpoint_sha256"] = "bad"
    with pytest.raises(RuntimeError, match="provenance"):
        validate_a1_adapter_pack(pack)
