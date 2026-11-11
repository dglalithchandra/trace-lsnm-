"""Apply the frozen base-only rare-aware adapter to DINOv2 feature splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_models import RareAwareAdapter
from trace_protocols import l2_normalize


def transform(model: RareAwareAdapter, values: np.ndarray, device: torch.device) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), 512):
            tensor = torch.from_numpy(values[start : start + 512]).to(device)
            outputs.append(model(tensor).cpu().numpy())
    result = np.concatenate(outputs).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("nonfinite_rare_adapter_features")
    return result


def main() -> None:
    checkpoint_path = (
        WORKSPACE / "models/rare_aware_adapter_v1/rare_aware_adapter_v1.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["development_decision"] != "frozen_for_novel_evaluation":
        raise RuntimeError("rare_adapter_not_frozen_for_evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RareAwareAdapter(width=384, hidden_width=128).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    output_dir = WORKSPACE / "data/features"
    reports = []
    transformed: dict[str, np.ndarray] = {}
    clean_original: dict[str, np.ndarray] = {}
    for split in ("train", "validation"):
        source = output_dir / f"rfmid_{split}_dinov2_vits14_geometry_v1.npz"
        with np.load(source, allow_pickle=False) as payload:
            embeddings = l2_normalize(payload["embeddings"].astype(np.float32))
            labels = payload["labels"].astype(np.uint8)
            record_ids = payload["record_ids"].astype(str)
            class_order = payload["class_order"].astype(str)
            if str(payload["split"][0]) != split:
                raise RuntimeError(f"rare_adapter_source_split_mismatch:{split}")
        adapted = transform(model, embeddings, device)
        destination = output_dir / f"rfmid_{split}_trace_rare_adapter_v1.npz"
        np.savez_compressed(
            destination,
            record_ids=record_ids.astype(np.str_),
            embeddings=adapted,
            labels=labels,
            class_order=class_order,
            split=np.asarray([split], dtype=np.str_),
            source=np.asarray(["frozen_dinov2_vits14_geometry_v1"], dtype=np.str_),
            adapter=np.asarray(["rare_aware_adapter_v1"], dtype=np.str_),
        )
        cosine = np.sum(embeddings * adapted, axis=1)
        reports.append(
            {
                "split": split,
                "records": len(record_ids),
                "shape": list(adapted.shape),
                "mean_cosine_to_original": float(cosine.mean()),
                "p05_cosine_to_original": float(np.quantile(cosine, 0.05)),
                "output": str(destination.relative_to(WORKSPACE)).replace("\\", "/"),
            }
        )
        transformed[split] = adapted
        clean_original[split] = embeddings

    with np.load(
        output_dir / "rfmid_validation_dinov2_camera_stress_v1.npz",
        allow_pickle=False,
    ) as payload:
        if payload["record_ids"].astype(str).shape[0] != len(transformed["validation"]):
            raise RuntimeError("rare_adapter_stress_validation_count_mismatch")
        stress = l2_normalize(payload["stressed_embeddings"].astype(np.float32))
    adapted_stress = transform(model, stress, device)
    baseline_pair_cosine = np.sum(clean_original["validation"] * stress, axis=1)
    adapted_pair_cosine = np.sum(transformed["validation"] * adapted_stress, axis=1)
    summary = {
        "status": "PASS",
        "checkpoint": str(checkpoint_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "selected_candidate": checkpoint["selected_candidate"],
        "splits": reports,
        "paired_camera_stress": {
            "baseline_mean_cosine": float(baseline_pair_cosine.mean()),
            "adapted_mean_cosine": float(adapted_pair_cosine.mean()),
            "delta": float(adapted_pair_cosine.mean() - baseline_pair_cosine.mean()),
        },
        "novel_labels_used_for_training_or_selection": [],
        "validation_labels_used_during_transformation": False,
        "protected_test_records_accessed": 0,
    }
    path = output_dir / "trace_rare_adapter_feature_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
