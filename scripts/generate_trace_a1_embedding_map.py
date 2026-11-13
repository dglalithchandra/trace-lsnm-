"""Generate a development-only TRACE A1 embedding map without protected-test access."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

try:
    import umap
except Exception as error:  # Windows policy may block Numba's extension DLL.
    umap = None
    UMAP_IMPORT_ERROR = f"{type(error).__name__}: {error}"
else:
    UMAP_IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_models.dinov2 import sha256_file
from trace_protocols import l2_normalize, load_feature_split


TAG = "a1_seed20261111"
OUTPUT = ROOT / "outputs" / "trace_source_balance_v1" / "a1_embedding_map"


def novel_group(row: np.ndarray, class_order: list[str], novel: list[str]) -> str:
    present = [label for label in novel if row[class_order.index(label)] == 1]
    if not present:
        return "No eligible low-support label"
    if len(present) > 1:
        return "Multiple low-support labels"
    return present[0]


def main() -> None:
    protocol = json.loads(
        (ROOT / "data/protocols/trace_fewshot_v1/protocol.json").read_text(encoding="utf-8")
    )
    novel = list(protocol["novel_labels"])
    paths = {
        split: ROOT / "data/features" / f"rfmid_{split}_trace_source_balance_{TAG}.npz"
        for split in ("train", "validation")
    }
    features = [load_feature_split(paths[split], split) for split in ("train", "validation")]
    if features[0].class_order != features[1].class_order:
        raise RuntimeError("A1 train/validation class order mismatch")
    if [len(item.record_ids) for item in features] != [1917, 639]:
        raise RuntimeError("Unexpected A1 development record counts")

    record_ids = np.concatenate([item.record_ids for item in features])
    matrix = l2_normalize(np.concatenate([item.embeddings for item in features]))
    labels = np.concatenate([item.labels for item in features])
    splits = np.concatenate(
        [np.repeat(split, len(item.record_ids)) for split, item in zip(("train", "validation"), features)]
    )
    manifest = pd.read_csv(ROOT / "data/manifests/trace_analysis_manifest.csv", low_memory=False)
    rfmid = manifest.loc[manifest.record_id.isin(record_ids), ["record_id", "camera_domain"]]
    camera_by_id = dict(zip(rfmid.record_id.astype(str), rfmid.camera_domain.astype(str), strict=True))
    missing = sorted(set(record_ids.astype(str)) - set(camera_by_id))
    if missing:
        raise RuntimeError(f"Missing reported camera domain for {len(missing)} A1 records")
    cameras = np.asarray([camera_by_id[str(record_id)] for record_id in record_ids])
    groups = np.asarray(
        [novel_group(row, features[0].class_order, novel) for row in labels], dtype=object
    )

    if umap is not None:
        projection_method = "UMAP"
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="cosine",
            random_state=20261111,
            transform_seed=20261111,
            n_jobs=1,
        )
    else:
        projection_method = "PCA_FALLBACK_UMAP_RUNTIME_BLOCKED"
        reducer = PCA(n_components=2, random_state=20261111)
    projection_label = "UMAP" if projection_method == "UMAP" else "PCA (fallback; UMAP blocked)"
    coordinates = reducer.fit_transform(matrix)
    if coordinates.shape != (2556, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError(f"Invalid projection coordinates: {coordinates.shape}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "record_id": record_ids,
            "split": splits,
            "novel_group": groups,
            "reported_camera_domain": cameras,
            "projection_x": coordinates[:, 0],
            "projection_y": coordinates[:, 1],
        }
    )
    frame.to_csv(OUTPUT / "coordinates.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    base = groups == "No eligible low-support label"
    axes[0].scatter(coordinates[base, 0], coordinates[base, 1], s=5, c="#d1d5db", alpha=0.28, linewidths=0)
    categories = [label for label in novel if label in set(groups)]
    if "Multiple low-support labels" in set(groups):
        categories.append("Multiple low-support labels")
    palette = plt.get_cmap("tab20")
    for index, category in enumerate(categories):
        mask = groups == category
        axes[0].scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=16, alpha=0.78,
            color=palette(index % 20), label=f"{category} (n={int(mask.sum())})", linewidths=0,
        )
    axes[0].set_title("Low-support RFMiD labels (development only)")
    axes[0].legend(fontsize=7, markerscale=1.3, frameon=False, loc="best")

    camera_categories = sorted(set(cameras))
    camera_palette = plt.get_cmap("Set2")
    for index, category in enumerate(camera_categories):
        mask = cameras == category
        axes[1].scatter(
            coordinates[mask, 0], coordinates[mask, 1], s=8, alpha=0.55,
            color=camera_palette(index), label=f"{category} (n={int(mask.sum())})", linewidths=0,
        )
    axes[1].set_title("Reported camera domain (partly inferred from dimensions)")
    axes[1].legend(fontsize=7, markerscale=1.5, frameon=False, loc="best")
    for axis in axes:
        axis.set_xlabel(f"{projection_label} 1")
        axis.set_ylabel(f"{projection_label} 2")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"TRACE A1 embeddings — RFMiD train + validation — {projection_label}",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUTPUT / "trace_a1_embedding_map.png", dpi=220, facecolor="white")
    plt.close(fig)

    summary = {
        "status": "DEVELOPMENT_ONLY_VISUALIZATION",
        "model": "trace_a1_seed_20261111",
        "projection_method": projection_method,
        "umap_import_error": UMAP_IMPORT_ERROR,
        "records": len(frame),
        "split_counts": dict(Counter(splits)),
        "novel_group_counts": dict(Counter(groups)),
        "reported_camera_domain_counts": dict(Counter(cameras)),
        "feature_sha256": {split: sha256_file(path) for split, path in paths.items()},
        "parameters": (
            {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine", "random_state": 20261111}
            if projection_method == "UMAP"
            else {"n_components": 2, "input": "l2_normalized_embeddings", "random_state": 20261111}
        ),
        "protected_test_records_accessed": 0,
        "limitations": [
            "The 2D projection is descriptive and does not prove diagnostic separation or domain invariance.",
            "If projection_method reports PCA_FALLBACK, this output is not a UMAP and must not be labelled as one.",
            "RFMiD camera domains are partly inferred from image dimensions, not verified per-image metadata.",
            "Multilabel records with more than one eligible low-support label are grouped together.",
            "Train and validation were previously used for development and model selection.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
