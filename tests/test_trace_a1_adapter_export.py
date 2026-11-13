from __future__ import annotations

import torch

from scripts.export_trace_a1_adapter import build_adapter_pack
from trace_models.source_balance import A1_CHECKPOINT_SHA256


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
    }


def test_adapter_export_keeps_only_expected_lora_tensors() -> None:
    state = valid_state()
    state["student"] = {
        **{
            f"blocks.{block}.attn.{layer}.lora_{side}": torch.zeros(
                (8, 384) if side == "a" else ((1152, 8) if layer == "qkv" else (384, 8))
            )
            for block in range(8, 12)
            for layer in ("qkv", "proj")
            for side in ("a", "b")
        },
        "blocks.0.norm.weight": torch.ones(384),
    }
    pack = build_adapter_pack(state, A1_CHECKPOINT_SHA256[20261111])
    assert pack["format"] == "trace-a1-lora-adapter-v1"
    assert pack["tensor_count"] == 16
    assert pack["parameter_count"] == 73_728
    assert "blocks.0.norm.weight" not in pack["lora_state"]
