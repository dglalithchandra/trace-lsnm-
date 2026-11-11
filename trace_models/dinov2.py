"""Verified local loader for the frozen DINOv2-S/14 retinal encoder."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


DINOV2_VITS14_CHECKPOINT_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
DINOV2_VITL14_CHECKPOINT_SHA256 = (
    "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"
)
DINOV2_SOURCE_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dinov2_vits14(
    workspace: Path | str,
    *,
    device: torch.device | str,
) -> torch.nn.Module:
    """Load the pinned local DINOv2 code and verified official checkpoint."""

    workspace = Path(workspace)
    checkpoint = workspace / "models" / "dinov2_vits14" / "dinov2_vits14_pretrain.pth"
    source = workspace / "models" / "torch" / "hub" / "facebookresearch_dinov2_main"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv2 checkpoint not found: {checkpoint}")
    if not source.is_dir():
        raise FileNotFoundError(f"Pinned DINOv2 source not found: {source}")
    observed_hash = sha256_file(checkpoint)
    if observed_hash != DINOV2_VITS14_CHECKPOINT_SHA256:
        raise RuntimeError(
            "DINOv2 checkpoint SHA-256 mismatch: "
            f"{observed_hash} != {DINOV2_VITS14_CHECKPOINT_SHA256}"
        )
    model = torch.hub.load(
        str(source), "dinov2_vits14", source="local", pretrained=False
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    return model.eval().requires_grad_(False).to(device)


def load_dinov2_vitl14(
    workspace: Path | str,
    *,
    device: torch.device | str,
) -> torch.nn.Module:
    """Load the pinned local DINOv2-L/14 code and verified official checkpoint."""

    workspace = Path(workspace)
    checkpoint = workspace / "models" / "dinov2_vitl14" / "dinov2_vitl14_pretrain.pth"
    source = workspace / "models" / "torch" / "hub" / "facebookresearch_dinov2_main"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv2-L checkpoint not found: {checkpoint}")
    if not source.is_dir():
        raise FileNotFoundError(f"Pinned DINOv2 source not found: {source}")
    observed_hash = sha256_file(checkpoint)
    if observed_hash != DINOV2_VITL14_CHECKPOINT_SHA256:
        raise RuntimeError(
            "DINOv2-L checkpoint SHA-256 mismatch: "
            f"{observed_hash} != {DINOV2_VITL14_CHECKPOINT_SHA256}"
        )
    model = torch.hub.load(
        str(source), "dinov2_vitl14", source="local", pretrained=False
    )
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    return model.eval().requires_grad_(False).to(device)
