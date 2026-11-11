"""Extract frozen official DINOv2-L/14 RFMiD train/validation embeddings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts.extract_rfmid_dinov2_features import extract_split, official_class_order
from trace_models.dinov2 import (
    DINOV2_SOURCE_COMMIT,
    DINOV2_VITL14_CHECKPOINT_SHA256,
    load_dinov2_vitl14,
)
from trace_preprocessing import PreprocessingConfig, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the full DINOv2-L extraction run")

    encoder = load_dinov2_vitl14(WORKSPACE, device=device)
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
                        / f"rfmid_{split}_dinov2_vitl14_geometry448_v1.npz"
                    ),
                )
            )
    finally:
        loader.close()

    summary = {
        "status": "PASS",
        "model": "dinov2_vitl14",
        "parameter_count": parameter_count,
        "model_source_commit": DINOV2_SOURCE_COMMIT,
        "model_checkpoint_sha256": DINOV2_VITL14_CHECKPOINT_SHA256,
        "input_size": 448,
        "embedding_dim": 1024,
        "preprocessing": pipeline.config.to_dict(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "contamination_tag": "generic_pretraining_overlap_unknown",
        "test_records_accessed": 0,
        "splits": reports,
    }
    (output_dir / "rfmid_dinov2_vitl14_feature_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
