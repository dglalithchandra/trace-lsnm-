"""Check that cached geometry reproduces frozen DINOv2-S development features."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_models.dinov2 import load_dinov2_vits14
from trace_preprocessing.adapters import prepare_dinov2_tensor
from trace_protocols import load_feature_split

OUT = ROOT / "data/protocols/trace_ssl_v1/cache_equivalence_summary.json"


def main() -> None:
    device = torch.device("cuda")
    model = load_dinov2_vits14(ROOT, device=device)
    checks = {}
    for split, index_name in (("train", "cached_image_index.csv"), ("validation", "cached_validation_index.csv")):
        baseline = load_feature_split(ROOT / "data/features" / f"rfmid_{split}_dinov2_vits14_geometry_v1.npz", split)
        with (ROOT / "data/protocols/trace_ssl_v1" / index_name).open(newline="", encoding="utf-8") as stream:
            index = {row["record_id"]: row for row in csv.DictReader(stream)}
        if set(index) != set(baseline.record_ids):
            raise RuntimeError(f"Cache IDs differ from baseline {split}")
        positions = np.linspace(0, len(baseline.record_ids) - 1, 24, dtype=int)
        tensors = []
        for position in positions:
            path = ROOT / index[baseline.record_ids[position]]["cache_path"]
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Cannot read {path}")
            tensors.append(prepare_dinov2_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        outputs = []
        for offset in range(0, len(tensors), 8):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                outputs.append(model(torch.stack(tensors[offset:offset + 8]).to(device)).float().cpu().numpy())
        observed = np.concatenate(outputs)
        expected = baseline.embeddings[positions]
        cosine = np.sum(observed * expected, axis=1) / (np.linalg.norm(observed, axis=1) * np.linalg.norm(expected, axis=1))
        checks[split] = {"sampled_images": len(positions), "min_cosine_to_original_extraction": float(cosine.min()), "mean_cosine_to_original_extraction": float(cosine.mean()), "max_absolute_embedding_difference": float(np.max(np.abs(observed - expected)))}
        if cosine.min() < 0.9999:
            raise RuntimeError(f"Cached {split} preprocessing differs from original extraction")
    summary = {"status": "CACHE_EQUIVALENT_TO_FROZEN_BASELINE_ON_SAMPLED_IMAGES", "checks": checks, "test_records_accessed": 0}
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
