"""Export the selected TRACE A1 LoRA weights without training-only state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_models.dinov2 import DINOV2_VITS14_CHECKPOINT_SHA256, sha256_file
from trace_models.source_balance import A1_CHECKPOINT_SHA256, validate_a1_checkpoint_state


def build_adapter_pack(state: dict[str, object], source_sha256: str) -> dict[str, object]:
    config = validate_a1_checkpoint_state(state)
    expected_source = A1_CHECKPOINT_SHA256[int(config["seed"])]
    if source_sha256 != expected_source:
        raise RuntimeError("Source checkpoint hash does not match the verified A1 seed")
    student = state["student"]
    adapter = {
        key: value.detach().cpu()
        for key, value in student.items()
        if key.endswith(".lora_a") or key.endswith(".lora_b")
    }
    if len(adapter) != 16 or sum(value.numel() for value in adapter.values()) != 73_728:
        raise RuntimeError("Unexpected TRACE A1 adapter tensor set")
    return {
        "format": "trace-a1-lora-adapter-v1",
        "source_checkpoint_sha256": source_sha256,
        "base_checkpoint_sha256": DINOV2_VITS14_CHECKPOINT_SHA256,
        "config": config,
        "lora_state": adapter,
        "tensor_count": len(adapter),
        "parameter_count": 73_728,
        "research_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/trace_source_balance_v1/a1_seed20261111/last.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/trace_source_balance_v1/a1_seed20261111/adapter.pt"),
    )
    args = parser.parse_args()
    source = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite adapter: {output}")
    state = torch.load(source, map_location="cpu", weights_only=False)
    source_hash = sha256_file(source)
    pack = build_adapter_pack(state, source_hash)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(pack, temporary)
    temporary.replace(output)
    summary = {
        "status": "TRACE_A1_ADAPTER_EXPORTED",
        "source": str(source.relative_to(ROOT)).replace("\\", "/"),
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash,
        "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "tensor_count": pack["tensor_count"],
        "parameter_count": pack["parameter_count"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
