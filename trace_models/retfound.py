"""Verified loader for the official RETFound-DINOv2 MEH teacher backbone."""

from __future__ import annotations

from pathlib import Path

import torch

from trace_models.dinov2 import DINOV2_SOURCE_COMMIT, sha256_file


RETFOUND_DINOV2_MEH_CHECKPOINT_SHA256 = (
    "a3feeaf69b44a0baa0934162d06b11ca517d1ca71a8459e04ef72c287d4e7124"
)
RETFOUND_HF_REPOSITORY = "YukunZhou/RETFound_dinov2_meh"
RETFOUND_HF_REVISION = "6dff4c76632983267a6ae3000f7852b1f8667d9c"


def load_retfound_dinov2_meh(
    workspace: Path | str,
    *,
    device: torch.device | str,
) -> torch.nn.Module:
    """Load only the verified RETFound teacher backbone with pinned DINOv2 code."""

    workspace = Path(workspace)
    checkpoint = (
        workspace
        / "models"
        / "retfound_dinov2_meh"
        / "RETFound_dinov2_meh.pth"
    )
    source = workspace / "models" / "torch" / "hub" / "facebookresearch_dinov2_main"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RETFound checkpoint not found: {checkpoint}")
    if not source.is_dir():
        raise FileNotFoundError(f"Pinned DINOv2 source not found: {source}")
    observed_hash = sha256_file(checkpoint)
    if observed_hash != RETFOUND_DINOV2_MEH_CHECKPOINT_SHA256:
        raise RuntimeError(
            "RETFound checkpoint SHA-256 mismatch: "
            f"{observed_hash} != {RETFOUND_DINOV2_MEH_CHECKPOINT_SHA256}"
        )

    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(bundle) != {"teacher"} or not isinstance(bundle["teacher"], dict):
        raise RuntimeError("Unexpected RETFound checkpoint container")
    teacher = bundle["teacher"]
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in teacher.items()
        if key.startswith("backbone.")
    }
    if len(backbone_state) != 343:
        raise RuntimeError(
            f"Unexpected RETFound teacher-backbone key count: {len(backbone_state)}"
        )

    model = torch.hub.load(
        str(source), "dinov2_vitl14", source="local", pretrained=False
    )
    model.load_state_dict(backbone_state, strict=True)
    return model.eval().requires_grad_(False).to(device)


__all__ = [
    "DINOV2_SOURCE_COMMIT",
    "RETFOUND_DINOV2_MEH_CHECKPOINT_SHA256",
    "RETFOUND_HF_REPOSITORY",
    "RETFOUND_HF_REVISION",
    "load_retfound_dinov2_meh",
]
