"""Extract official RETFound-DINOv2 MEH RFMiD train/validation embeddings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.extract_rfmid_dinov2_features import extract_split, official_class_order
from trace_models.retfound import (
    DINOV2_SOURCE_COMMIT,
    RETFOUND_DINOV2_MEH_CHECKPOINT_SHA256,
    RETFOUND_HF_REPOSITORY,
    RETFOUND_HF_REVISION,
    load_retfound_dinov2_meh,
)
from trace_preprocessing import PreprocessingConfig, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the full RETFound extraction run")

    encoder = load_retfound_dinov2_meh(WORKSPACE, device=device)
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    pipeline = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    )
    loader = ManifestImageLoader(WORKSPACE, allow_test=False)
    class_order = official_class_order(WORKSPACE)
    reports: list[dict[str, object]] = []
    output_dir = WORKSPACE / "data" / "features"
    try:
        for split in ("train", "validation"):
            reports.append(
                extract_split(
                    split=split,
                    loader=loader,
                    pipeline=pipeline,
                    encoder=encoder,
                    class_order=class_order,
                    device=device,
                    batch_size=8,
                    embedding_dim=1024,
                    output_path=(
                        output_dir
                        / f"rfmid_{split}_retfound_dinov2_meh_geometry448_v1.npz"
                    ),
                )
            )
    finally:
        loader.close()

    summary = {
        "status": "PASS",
        "model": "RETFound_dinov2_meh teacher backbone",
        "model_repository": RETFOUND_HF_REPOSITORY,
        "model_repository_revision": RETFOUND_HF_REVISION,
        "model_checkpoint_sha256": RETFOUND_DINOV2_MEH_CHECKPOINT_SHA256,
        "architecture_source_commit": DINOV2_SOURCE_COMMIT,
        "parameter_count": parameter_count,
        "input_size": 448,
        "embedding_dim": 1024,
        "preprocessing": pipeline.config.to_dict(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "pretraining_disclosure": (
            "Official model card states DINOv2 pretraining on part of AlzEye; "
            "RFMiD is not listed, but absence of overlap cannot be proven."
        ),
        "contamination_tag": "declared_alzeye_pretraining_rfmid_not_listed",
        "license": "CC BY-NC 4.0",
        "test_records_accessed": 0,
        "splits": reports,
    }
    (output_dir / "rfmid_retfound_feature_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
