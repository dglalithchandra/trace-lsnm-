"""Offline integrity and GPU smoke test for the pinned DINOv2-S/14 encoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


EXPECTED_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
EXPECTED_BYTES = 88_283_115
EXPECTED_PARAMETERS = 22_056_576
EXPECTED_OUTPUT_WIDTH = 384


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    checkpoint = workspace / "models" / "dinov2_vits14" / "dinov2_vits14_pretrain.pth"
    source = workspace / "models" / "torch" / "hub" / "facebookresearch_dinov2_main"

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if checkpoint.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError("Unexpected DINOv2 checkpoint size")

    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"DINOv2 checkpoint hash mismatch: {actual_sha256}")

    model = torch.hub.load(
        str(source),
        "dinov2_vits14",
        source="local",
        pretrained=False,
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.requires_grad_(False)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(f"Unexpected parameter count: {parameter_count}")
    if trainable_count != 0:
        raise RuntimeError(f"Encoder is not frozen: {trainable_count} trainable parameters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_input = torch.zeros((1, 3, 448, 448), device=device)
    with torch.inference_mode():
        output = model(test_input)

    if tuple(output.shape) != (1, EXPECTED_OUTPUT_WIDTH):
        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("DINOv2 output contains non-finite values")

    report = {
        "status": "PASS",
        "model": "dinov2_vits14",
        "checkpoint_sha256": actual_sha256,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "input_shape": [1, 3, 448, 448],
        "output_shape": list(output.shape),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
