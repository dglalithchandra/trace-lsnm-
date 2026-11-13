"""Research-only image-to-embedding interface for the selected TRACE A1 arm."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.adapters import prepare_dinov2_tensor
from trace_preprocessing.types import PreprocessingResult

from .dinov2 import DINOV2_VITS14_CHECKPOINT_SHA256, load_dinov2_vits14, sha256_file


A1_CHECKPOINT_SHA256 = {
    20261111: "fead3b6163e1ee29a3f5091a899270e265c24c42c52dc679fc11a941318f7aec",
    20261112: "2504939a66ebbdaaf907f0cf559af1bc36b3528ef68ed38b60d65b23f8a4f82b",
    20261113: "3d6a69c34b20c30553d5f083e6a8db233e1d9887ddf7dd6322f586e4580592e7",
}
A1_ADAPTER_SHA256 = {
    20261111: "8857f811b723cc0b5559b5a02d32ae116b6754c84fcc9c3603e835ba967e529c",
}


class LoRALinear(nn.Module):
    """LoRA layer matching the source-balance-v1 training architecture."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 8.0) -> None:
        super().__init__()
        self.base = linear
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_b = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


def add_source_balance_lora(model: nn.Module) -> int:
    """Attach the exact rank-8 LoRA structure used in A1 training."""

    for block in model.blocks[-4:]:
        block.attn.qkv = LoRALinear(block.attn.qkv)
        block.attn.proj = LoRALinear(block.attn.proj)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable != 73_728:
        raise RuntimeError(f"Unexpected TRACE A1 LoRA parameter count: {trainable}")
    return trainable


def validate_a1_checkpoint_state(state: dict[str, Any]) -> dict[str, Any]:
    """Validate identity and completion without consulting evaluation outputs."""

    config = state.get("config")
    if not isinstance(config, dict) or config.get("version") != "trace-source-balance-v1":
        raise RuntimeError("Not a TRACE source-balance-v1 checkpoint")
    required = {
        "manifest": "data/manifests/trace_ssl_pool_v2.csv",
        "record_count": 1917,
        "sources": ["rfmid_v1_train"],
        "sampling": "natural",
        "updates": 9580,
        "view": "plain_crop_matched",
        "rank": 8,
        "alpha": 8.0,
        "backbone_trainable_parameters": 73_728,
    }
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint is not the selected A1 configuration: {mismatches}")
    if int(state.get("step", -1)) != int(config["updates"]):
        raise RuntimeError("A1 checkpoint is incomplete")
    seed = int(config.get("seed", -1))
    if seed not in A1_CHECKPOINT_SHA256:
        raise RuntimeError(f"Unrecognised A1 replication seed: {seed}")
    if "student" not in state:
        raise RuntimeError("A1 checkpoint has no student state")
    return config


def validate_a1_adapter_pack(pack: dict[str, Any]) -> dict[str, Any]:
    if pack.get("format") != "trace-a1-lora-adapter-v1":
        raise RuntimeError("Not a TRACE A1 LoRA adapter pack")
    config = validate_a1_checkpoint_state(
        {"config": pack.get("config"), "step": pack.get("config", {}).get("updates"), "student": {}}
    )
    seed = int(config["seed"])
    if pack.get("source_checkpoint_sha256") != A1_CHECKPOINT_SHA256[seed]:
        raise RuntimeError("Adapter source-checkpoint provenance mismatch")
    if pack.get("base_checkpoint_sha256") != DINOV2_VITS14_CHECKPOINT_SHA256:
        raise RuntimeError("Adapter base-checkpoint provenance mismatch")
    lora_state = pack.get("lora_state")
    if not isinstance(lora_state, dict):
        raise RuntimeError("Adapter pack has no LoRA tensor state")
    if len(lora_state) != 16 or sum(value.numel() for value in lora_state.values()) != 73_728:
        raise RuntimeError("Adapter pack tensor set is incomplete")
    return config


class TraceA1Encoder:
    """Apply frozen TRACE preprocessing and a verified completed A1 encoder."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        checkpoint: Path | str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        default_dir = self.workspace / "models" / "trace_source_balance_v1" / "a1_seed20261111"
        self.checkpoint = Path(checkpoint) if checkpoint else (
            default_dir / "adapter.pt" if (default_dir / "adapter.pt").is_file() else default_dir / "last.pt"
        )
        if not self.checkpoint.is_absolute():
            self.checkpoint = self.workspace / self.checkpoint
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"TRACE A1 checkpoint not found: {self.checkpoint}")

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        observed_hash = sha256_file(self.checkpoint)
        is_adapter = state.get("format") == "trace-a1-lora-adapter-v1"
        self.config = (
            validate_a1_adapter_pack(state) if is_adapter else validate_a1_checkpoint_state(state)
        )
        expected_hash = (
            A1_ADAPTER_SHA256.get(int(self.config["seed"]))
            if is_adapter
            else A1_CHECKPOINT_SHA256[int(self.config["seed"])]
        )
        if expected_hash is None:
            raise RuntimeError("No frozen hash is registered for this TRACE A1 adapter")
        if observed_hash != expected_hash:
            raise RuntimeError(f"TRACE A1 checkpoint SHA-256 mismatch: {observed_hash}")
        self.checkpoint_sha256 = observed_hash
        self.source_checkpoint_sha256 = (
            state["source_checkpoint_sha256"] if is_adapter else observed_hash
        )
        self.artifact_kind = "lora_adapter" if is_adapter else "training_checkpoint"

        self.encoder = load_dinov2_vits14(self.workspace, device=self.device)
        add_source_balance_lora(self.encoder)
        if is_adapter:
            parameters = dict(self.encoder.named_parameters())
            expected_keys = {name for name in parameters if name.endswith((".lora_a", ".lora_b"))}
            if set(state["lora_state"]) != expected_keys:
                raise RuntimeError("Adapter tensor names do not match the TRACE A1 architecture")
            with torch.no_grad():
                for name, value in state["lora_state"].items():
                    parameters[name].copy_(value)
        else:
            self.encoder.load_state_dict(state["student"], strict=True)
        self.encoder.eval().requires_grad_(False).to(self.device)
        self.preprocessor = TracePreprocessor(
            PreprocessingConfig(apply_photometric_correction=False)
        )

    def embed_image(self, image: np.ndarray, *, image_id: str = "unidentified") -> dict[str, Any]:
        return self.embed_preprocessed(self.preprocessor.process(image, image_id=image_id))

    def embed_preprocessed(self, result: PreprocessingResult) -> dict[str, Any]:
        if result.status == ProcessingStatus.REJECT or result.processed_image is None:
            return {
                "image_id": result.image_id,
                "status": "rejected",
                "preprocessing_status": result.status.value,
                "reasons": list(result.reasons),
                "warnings": list(result.warnings),
                "embedding": None,
                "research_only": True,
            }
        tensor = prepare_dinov2_tensor(result.processed_image).unsqueeze(0).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            embedding = self.encoder(tensor)
        if embedding.shape != (1, 384) or not torch.isfinite(embedding).all():
            raise RuntimeError(f"Invalid TRACE A1 embedding: shape={tuple(embedding.shape)}")
        return {
            "image_id": result.image_id,
            "status": "embedded",
            "preprocessing_status": result.status.value,
            "reasons": list(result.reasons),
            "warnings": list(result.warnings),
            "model_id": f"trace_a1_seed_{self.config['seed']}",
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "artifact_kind": self.artifact_kind,
            "preprocessing_version": result.preprocessing_version,
            "embedding": embedding[0].float().cpu().numpy(),
            "embedding_interpretation": (
                "A 384-value representation for research retrieval or a separately fitted "
                "few-shot head; it is not a disease probability or confidence score."
            ),
            "research_only": True,
        }
