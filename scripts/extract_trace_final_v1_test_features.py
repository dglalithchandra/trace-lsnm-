"""Extract the five locked RFMiD v1 test embeddings only after audit release.

This script intentionally has no development mode. The release file is created
only after ambiguous overlap decisions and the final code lock are recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_rfmid_dinov2_features import label_vector, official_class_order
from scripts.freeze_trace_final_plan import sha256
from scripts.train_trace_ssl_v1 import add_lora
from trace_models.dinov2 import load_dinov2_vitl14, load_dinov2_vits14
from trace_models.retfound import load_retfound_dinov2_meh
from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.adapters import prepare_dinov2_tensor
from trace_preprocessing.loader import ManifestImageLoader

PROTOCOL = ROOT / "data/protocols/trace_final_v1"
LOCK = PROTOCOL / "plan_lock.json"
RELEASE = PROTOCOL / "scoring_release.json"
DECISIONS = PROTOCOL / "audit_decisions.csv"
QUEUE = PROTOCOL / "AUDIT_REVIEW_QUEUE.csv"
OUTPUT = ROOT / "outputs/trace_final_v1"
MODELS = (
    ("frozen_dinov2_s", 384),
    ("frozen_dinov2_l", 1024),
    ("frozen_retfound", 1024),
    ("plain_crop_matched_epoch5", 384),
    ("full_field_epoch5", 384),
)
EXPLICIT_CODE_PATHS = (
    "scripts/extract_trace_final_v1_test_features.py",
    "scripts/evaluate_trace_final_v1.py",
    "scripts/extract_rfmid_dinov2_features.py",
    "scripts/train_trace_ssl_v1.py",
    "scripts/run_dinov2_fewshot_benchmark.py",
    "scripts/freeze_trace_final_plan.py",
    "scripts/freeze_trace_final_scoring_release.py",
)
REQUIRED_REVIEW_PAIRS = {
    ("rfmid:testing:377", "rfmid:training:835"),
    ("rfmid:testing:506", "rfmid:training:1577"),
    ("rfmid:testing:541", "rfmid:training:1877"),
    ("rfmid:testing:571", "rfmid:training:1818"),
    ("rfmid:testing:617", "rfmid:training:1852"),
    ("rfmid:testing:620", "rfmid:training:1787"),
    ("rfmid:testing:226", "rfmid:training:1292"),
    ("rfmid:testing:175", "rfmid:training:709"),
    ("rfmid:testing:36", "rfmid:validation:186"),
    ("rfmid:testing:261", "rfmid:training:347"),
    ("rfmid:testing:113", "rfmid:validation:79"),
    ("rfmid:testing:8", "rfmid:training:1201"),
}


def required_code_paths() -> list[str]:
    package_paths = [path.relative_to(ROOT).as_posix() for directory in ("trace_models", "trace_preprocessing", "trace_protocols") for path in (ROOT / directory).rglob("*.py")]
    return sorted(set(EXPLICIT_CODE_PATHS) | set(package_paths))


def verify_release() -> tuple[list[str], dict[str, object]]:
    if not RELEASE.is_file():
        raise RuntimeError("Test scoring is blocked: scoring_release.json does not exist")
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    if release.get("status") != "PASSED" or release.get("v1_audit_decision") != "ALL_AMBIGUITIES_ADJUDICATED":
        raise RuntimeError("Final v1 overlap audit has not passed")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if release.get("plan_lock_sha256") != sha256(LOCK):
        raise RuntimeError("Release points to a different locked analysis plan")
    for relative in ("data/protocols/trace_final_v1/ANALYSIS_PLAN.md", "data/manifests/trace_analysis_manifest.csv", "A. RFMiD_All_Classes_Dataset.zip"):
        if sha256(ROOT / relative) != lock["inputs_sha256"][relative]:
            raise RuntimeError(f"Locked source changed: {relative}")
    code_hashes = release.get("code_sha256", {})
    if set(code_hashes) != set(required_code_paths()):
        raise RuntimeError("Final code lock does not cover the complete evaluation pipeline")
    for relative in required_code_paths():
        if code_hashes.get(relative) != sha256(ROOT / relative):
            raise RuntimeError(f"Final evaluation code is not frozen: {relative}")
    excluded = release.get("excluded_v1_test_record_ids")
    if not isinstance(excluded, list) or len(excluded) != len(set(excluded)):
        raise RuntimeError("Release has an invalid exclusion list")
    if release.get("audit_decisions_sha256") != sha256(DECISIONS):
        raise RuntimeError("Final overlap decisions are missing or changed")
    if release.get("audit_review_queue_sha256") != sha256(QUEUE):
        raise RuntimeError("Human review queue changed after release")
    with DECISIONS.open(newline="", encoding="utf-8") as stream:
        decisions = list(csv.DictReader(stream))
    with QUEUE.open(newline="", encoding="utf-8") as stream:
        reviewed_queue = list(csv.DictReader(stream))
    if not decisions or any(row.get("decision") not in {"duplicate", "distinct"} or not row.get("reviewer") for row in decisions):
        raise RuntimeError("Overlap decisions are incomplete or have no reviewer")
    reviewed_pairs = {(row["query_id"], row["reference_id"]) for row in decisions}
    if not REQUIRED_REVIEW_PAIRS.issubset(reviewed_pairs) or len(reviewed_pairs) != len(decisions):
        raise RuntimeError("Required overlap pairs are absent or repeated in review")
    if {(row["query_id"], row["reference_id"], row["human_decision"], row["reviewer"].strip()) for row in reviewed_queue} != {(row["query_id"], row["reference_id"], row["decision"], row["reviewer"]) for row in decisions}:
        raise RuntimeError("Reviewed queue and frozen decisions disagree")
    duplicates = {row["query_id"] for row in decisions if row["decision"] == "duplicate"}
    if duplicates != set(excluded):
        raise RuntimeError("Release exclusions disagree with reviewed duplicate decisions")
    if release.get("unresolved_v1_candidate_pairs") != []:
        raise RuntimeError("Unresolved v1 overlap candidates remain")
    return excluded, release


def load_model(name: str, device: torch.device) -> torch.nn.Module:
    if name == "frozen_dinov2_s":
        return load_dinov2_vits14(ROOT, device=device)
    if name == "frozen_dinov2_l":
        return load_dinov2_vitl14(ROOT, device=device)
    if name == "frozen_retfound":
        return load_retfound_dinov2_meh(ROOT, device=device)
    mode = name.removesuffix("_epoch5")
    checkpoint = ROOT / "models/trace_ssl_v1" / mode / "last.pt"
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    relative = checkpoint.relative_to(ROOT).as_posix()
    if sha256(checkpoint) != lock["inputs_sha256"][relative]:
        raise RuntimeError(f"SSL checkpoint changed: {relative}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["config"]["mode"] != mode or int(state["epoch"]) != 5 or int(state["cursor"]) != 0:
        raise RuntimeError(f"Unexpected SSL checkpoint state: {mode}")
    encoder = load_dinov2_vits14(ROOT, device=device)
    add_lora(encoder)
    encoder.load_state_dict(state["student"], strict=True)
    return encoder.eval().requires_grad_(False).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-final-test", action="store_true", help="Explicitly permit test feature extraction after the locked gate")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not args.release_final_test:
        raise RuntimeError("Pass --release-final-test only after the audit and code lock pass")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    excluded, release = verify_release()
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"Final-test output already exists; refusing to overwrite: {OUTPUT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    class_order = official_class_order(ROOT)
    processed_images: list[np.ndarray] = []
    ids: list[str] = []
    labels: list[np.ndarray] = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    read_log_path = OUTPUT / "test_image_reads.jsonl"
    with ManifestImageLoader(ROOT, allow_test=True) as loader:
        all_ids = loader.record_ids(split="test", dataset="RFMiD")
        if len(all_ids) != 644 or not set(excluded).issubset(set(all_ids)):
            raise RuntimeError("RFMiD v1 test membership or exclusions changed")
        included = [record_id for record_id in all_ids if record_id not in set(excluded)]
        with read_log_path.open("x", encoding="utf-8", buffering=1) as read_log:
            for number, record_id in enumerate(included, 1):
                read_log.write(json.dumps({"record_id": record_id, "event": "read_attempt"}) + "\n")
                image, row = loader.load_image(record_id)
                read_log.write(json.dumps({"record_id": record_id, "decoded_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(), "event": "image_read"}) + "\n")
                if row["_effective_split"] != "test" or row["dataset"] != "RFMiD":
                    raise RuntimeError(f"Unexpected test record: {record_id}")
                result = pipeline.process(image, image_id=record_id)
                if result.status == ProcessingStatus.REJECT or result.processed_image is None:
                    raise RuntimeError(f"Geometry rejected locked test image: {record_id}")
                processed_images.append(result.processed_image.copy())
                ids.append(record_id)
                labels.append(label_vector(row["disease_labels_json"], class_order))
                if number % 100 == 0 or number == len(included):
                    print(f"test_preprocessing {number}/{len(included)}", flush=True)
    if len(ids) != len(processed_images) or len(ids) != len(labels):
        raise RuntimeError("Unpaired test inputs")
    label_array = np.stack(labels)
    for name, expected_dim in MODELS:
        output_path = OUTPUT / f"{name}_test_features.npz"
        if output_path.exists():
            raise RuntimeError(f"Refusing to overwrite model features: {name}")
        model = load_model(name, device)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(processed_images), args.batch_size):
                batch = torch.stack([prepare_dinov2_tensor(image) for image in processed_images[start:start + args.batch_size]]).to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    embedding = model(batch)
                if not torch.isfinite(embedding).all():
                    raise RuntimeError(f"Nonfinite test embedding from {name}")
                chunks.append(embedding.float().cpu().numpy())
                if (start // args.batch_size + 1) % 10 == 0:
                    print(f"{name} {min(start + args.batch_size, len(processed_images))}/{len(processed_images)}", flush=True)
        matrix = np.concatenate(chunks).astype(np.float32)
        if matrix.shape != (len(ids), expected_dim):
            raise RuntimeError(f"Unexpected embedding shape for {name}: {matrix.shape}")
        np.savez_compressed(output_path, record_ids=np.asarray(ids), embeddings=matrix, labels=label_array, class_order=np.asarray(class_order), split=np.asarray(["test"]))
        print(f"extracted {name} {len(ids)} test images", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    feature_hashes = {name: sha256(OUTPUT / f"{name}_test_features.npz") for name, _ in MODELS}
    (OUTPUT / "extraction_summary.json").write_text(json.dumps({"status": "TEST_FEATURES_EXTRACTED_NO_DISEASE_SCORES", "plan_lock_sha256": sha256(LOCK), "release_sha256": sha256(RELEASE), "test_records_included": len(ids), "excluded_record_ids": excluded, "feature_sha256": feature_hashes, "image_read_log_sha256": sha256(read_log_path), "device": str(device), "release_note": release.get("note", "")}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
