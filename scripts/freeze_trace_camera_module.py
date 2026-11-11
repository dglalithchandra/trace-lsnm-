"""Create the immutable file manifest for TRACE camera module v1 research release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
RELEASE = WORKSPACE / "release" / "trace_camera_module_v1"
FILES = (
    "trace_camera/__init__.py",
    "trace_camera/module.py",
    "trace_preprocessing/config.py",
    "trace_preprocessing/pipeline.py",
    "trace_preprocessing/geometry.py",
    "trace_preprocessing/measurements.py",
    "trace_preprocessing/quality.py",
    "trace_models/rfmid_scorer.py",
    "scripts/run_trace_camera_module.py",
    "models/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "models/rfmid_dinov2_linear_v1/rfmid_dinov2_linear_v1.pt",
    "outputs/camera_module_decision/current_decision.json",
    "outputs/palm_camera_transfer_v1/EXPERIMENT_2A_REPORT.md",
    "outputs/palm_trust_layer_v1/EXPERIMENT_3A_REPORT.md",
    "outputs/paraguay_external_shift_v1/EXPERIMENT_3B_REPORT.md",
    "outputs/rfmid_artifact_stress_v2/EXPERIMENT_4A_REPORT.md",
    "outputs/artifact_qc_v1/EXPERIMENT_4B_REPORT.md",
    "outputs/fives_frozen_artifact_qc_v1/EXPERIMENT_4C_REPORT.md",
    "outputs/final_camera_module_smoke/trace_result.json",
    "outputs/final_camera_module_smoke/trace_standardized.png",
    "outputs/final_camera_module_smoke/trace_retinal_mask.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = []
    for relative in FILES:
        path = WORKSPACE / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    RELEASE.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release": "trace-camera-module-v1.0-research",
        "date": "2026-11-11",
        "status": "frozen_research_prototype",
        "active_method": "camera-ID-agnostic geometry plus diagnostic quality measurements and frozen disease scoring",
        "disabled_methods": [
            "photometric harmonisation",
            "shift-consistency adapter",
            "learned artifact-QC detectors",
        ],
        "clinical_use": False,
        "universal_camera_claim": False,
        "protected_test_records_accessed_during_release_smoke": 0,
        "files": entries,
    }
    path = RELEASE / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
