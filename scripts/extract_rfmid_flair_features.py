"""Extract frozen official FLAIR image embeddings for RFMiD train/validation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from flair import FLAIRModel

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.loader import ManifestImageLoader


FLAIR_REPOSITORY = "jusiro2/FLAIR"
FLAIR_MODEL_REVISION = "5f6bdd0a068353dc41a896ba3abdd7c0f6d35938"
FLAIR_SOURCE_COMMIT = "d6652d53389ff49e5f73efaccf4246e9de88d1a3"
FLAIR_WEIGHT_SHA256 = (
    "050334cd934fa3126435c202c41f493ca222fc3fe3ea9c50cf21f67c792a1440"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_class_order(workspace: Path) -> list[str]:
    archive_path = workspace / "A. RFMiD_All_Classes_Dataset.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name for name in archive.namelist() if name.endswith("RFMiD_Training_Labels.csv")
        ]
        if len(members) != 1:
            raise RuntimeError(f"Unexpected RFMiD training label files: {members}")
        header = next(csv.reader(io.StringIO(archive.read(members[0]).decode("utf-8-sig"))))
    if header[:2] != ["ID", "Disease_Risk"] or len(header[2:]) != 45:
        raise RuntimeError(f"Unexpected RFMiD header: {header}")
    return header[2:]


def label_vector(raw_json: str, class_order: list[str]) -> np.ndarray:
    values = json.loads(raw_json)
    if set(values) != set(class_order):
        raise RuntimeError("RFMiD label keys do not match official class order")
    return np.asarray([int(values[name]) for name in class_order], dtype=np.uint8)


def verify_cached_weights(cache_dir: Path) -> Path:
    weight_path = (
        cache_dir
        / "models--jusiro2--FLAIR"
        / "snapshots"
        / FLAIR_MODEL_REVISION
        / "model.safetensors"
    )
    if not weight_path.is_file():
        raise FileNotFoundError(f"Cached FLAIR weights not found: {weight_path}")
    observed = sha256_file(weight_path)
    if observed != FLAIR_WEIGHT_SHA256:
        raise RuntimeError(
            f"FLAIR weight SHA-256 mismatch: {observed} != {FLAIR_WEIGHT_SHA256}"
        )
    return weight_path


def extract_split(
    *,
    split: str,
    loader: ManifestImageLoader,
    pipeline: TracePreprocessor,
    model: FLAIRModel,
    class_order: list[str],
    batch_size: int,
    output_path: Path,
) -> dict[str, object]:
    record_ids = loader.record_ids(split=split, dataset="RFMiD")
    if not record_ids:
        raise RuntimeError(f"No RFMiD records for split: {split}")
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    completed_ids: list[str] = []
    batch_tensors: list[torch.Tensor] = []
    batch_ids: list[str] = []
    batch_labels: list[np.ndarray] = []

    def flush_batch() -> None:
        if not batch_tensors:
            return
        inputs = torch.cat(batch_tensors, dim=0)
        with torch.inference_mode():
            output = model.vision_model(inputs)
        if output.ndim != 2 or output.shape[1] != 512:
            raise RuntimeError(f"Unexpected FLAIR embedding shape: {output.shape}")
        if not torch.isfinite(output).all():
            raise RuntimeError("Non-finite FLAIR embedding")
        embeddings.extend(output.float().cpu().numpy())
        completed_ids.extend(batch_ids)
        labels.extend(batch_labels)
        batch_tensors.clear()
        batch_ids.clear()
        batch_labels.clear()

    for index, record_id in enumerate(record_ids, start=1):
        image, row = loader.load_image(record_id)
        if row["_effective_split"] != split:
            raise RuntimeError(f"Split mismatch for {record_id}")
        result = pipeline.process(image, image_id=record_id)
        if result.status == ProcessingStatus.REJECT or result.processed_image is None:
            raise RuntimeError(f"Geometry rejected RFMiD record: {record_id}")
        batch_tensors.append(model.preprocess_image(result.processed_image))
        batch_ids.append(record_id)
        batch_labels.append(label_vector(row["disease_labels_json"], class_order))
        if len(batch_tensors) >= batch_size:
            flush_batch()
        if index % 100 == 0 or index == len(record_ids):
            print(f"flair_progress split={split} {index}/{len(record_ids)}", flush=True)
    flush_batch()

    embedding_array = np.asarray(embeddings, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.uint8)
    id_array = np.asarray(completed_ids, dtype=np.str_)
    if embedding_array.shape != (len(record_ids), 512):
        raise RuntimeError(f"Unexpected final FLAIR shape: {embedding_array.shape}")
    if label_array.shape != (len(record_ids), 45):
        raise RuntimeError(f"Unexpected label shape: {label_array.shape}")
    if len(set(completed_ids)) != len(completed_ids):
        raise RuntimeError("Duplicate record IDs in FLAIR features")
    norms = np.linalg.norm(embedding_array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError("FLAIR embeddings are not unit normalised as expected")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        record_ids=id_array,
        embeddings=embedding_array,
        labels=label_array,
        class_order=np.asarray(class_order, dtype=np.str_),
        split=np.asarray([split], dtype=np.str_),
    )
    return {
        "split": split,
        "records": len(record_ids),
        "embedding_shape": list(embedding_array.shape),
        "label_shape": list(label_array.shape),
        "embedding_norm_min": float(norms.min()),
        "embedding_norm_max": float(norms.max()),
        "output": str(output_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "sha256": sha256_file(output_path),
    }


def main() -> None:
    cache_dir = WORKSPACE / "models" / "huggingface"
    weight_path = verify_cached_weights(cache_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the full FLAIR extraction run")
    model = FLAIRModel.from_pretrained(
        FLAIR_REPOSITORY,
        revision=FLAIR_MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=True,
        vision_pretrained=False,
    ).eval().requires_grad_(False)
    pipeline = TracePreprocessor(
        PreprocessingConfig(apply_photometric_correction=False)
    )
    class_order = official_class_order(WORKSPACE)
    loader = ManifestImageLoader(WORKSPACE, allow_test=False)
    reports: list[dict[str, object]] = []
    try:
        for split in ("train", "validation"):
            reports.append(
                extract_split(
                    split=split,
                    loader=loader,
                    pipeline=pipeline,
                    model=model,
                    class_order=class_order,
                    batch_size=8,
                    output_path=(
                        WORKSPACE
                        / "data"
                        / "features"
                        / f"rfmid_{split}_flair_geometry_v1.npz"
                    ),
                )
            )
    finally:
        loader.close()

    summary = {
        "status": "PASS",
        "model": "FLAIR",
        "model_repository": FLAIR_REPOSITORY,
        "model_revision": FLAIR_MODEL_REVISION,
        "source_commit": FLAIR_SOURCE_COMMIT,
        "weight_file": str(weight_path.relative_to(WORKSPACE)).replace("\\", "/"),
        "weight_sha256": FLAIR_WEIGHT_SHA256,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "image_size": model.image_size,
        "preprocessing": pipeline.config.to_dict(),
        "contamination_tag": "possibly_seen",
        "contamination_reason": (
            "The official FLAIR pretraining assembly explicitly lists RFMiD."
        ),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "test_records_accessed": 0,
        "splits": reports,
    }
    summary_path = WORKSPACE / "data" / "features" / "rfmid_flair_feature_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

