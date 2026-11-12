"""Hash the locally fixed TRACE final-analysis inputs without opening test members."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/protocols/trace_final_v1/plan_lock.json"
INPUTS = (
    "data/protocols/trace_final_v1/ANALYSIS_PLAN.md",
    "data/protocols/trace_fewshot_v1/protocol.json",
    "data/protocols/trace_fewshot_v1/episode_plan.csv",
    "data/manifests/trace_analysis_manifest.csv",
    "data/features/rfmid_train_dinov2_vits14_geometry_v1.npz",
    "data/features/rfmid_train_dinov2_vitl14_geometry448_v1.npz",
    "data/features/rfmid_train_retfound_dinov2_meh_geometry448_v1.npz",
    "data/features/rfmid_train_trace_ssl_plain_crop_matched_epoch5.npz",
    "data/features/rfmid_train_trace_ssl_full_field_epoch5.npz",
    "models/trace_ssl_v1/plain_crop_matched/last.pt",
    "models/trace_ssl_v1/full_field/last.pt",
    "A. RFMiD_All_Classes_Dataset.zip",
    "data/external/RFMiD2.0/RFMiD2_0.zip",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"Existing lock must not be overwritten: {OUT}")
    hashes = {}
    for relative in INPUTS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = sha256(path)
    lock = {
        "version": "trace-final-v1",
        "status": "LOCAL_ANALYSIS_PLAN_LOCKED_IMPLEMENTATION_AND_TEST_AUDIT_PENDING",
        "date_local": "2026-11-12",
        "inputs_sha256": hashes,
        "protected_test_members_opened_by_this_script": 0,
        "note": "Hashing the outer ZIP bytes is not decoding or inspecting test members. Final extraction and evaluation code must be hashed separately before scoring.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
