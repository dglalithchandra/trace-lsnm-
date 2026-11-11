"""Run frozen TRACE camera module v1 on one colour-fundus photograph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from trace_camera import TraceCameraModule


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing TRACE output files in the exact output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not 1 <= args.top_k <= 45:
        raise ValueError("--top-k must be between 1 and 45")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "trace_result.json"
    standardized_path = output_dir / "trace_standardized.png"
    mask_path = output_dir / "trace_retinal_mask.png"
    targets = (result_path, standardized_path, mask_path)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "TRACE outputs already exist; choose a new --output-dir or pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    source_hash_before = sha256(image_path)
    with Image.open(image_path) as opened:
        image = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    module = TraceCameraModule(WORKSPACE, device=args.device)
    output = module.process(image, image_id=str(image_path))
    report = output.report
    report["source"] = {
        "path": str(image_path),
        "sha256_before": source_hash_before,
        "sha256_after": sha256(image_path),
        "original_unchanged": source_hash_before == sha256(image_path),
    }
    report["disease_model"]["scores"] = report["disease_model"]["scores"][
        : args.top_k
    ]
    written = []
    if output.standardized_image is not None and output.retinal_mask is not None:
        Image.fromarray(output.standardized_image, mode="RGB").save(standardized_path)
        Image.fromarray(output.retinal_mask, mode="L").save(mask_path)
        written.extend([standardized_path, mask_path])
    report["outputs"] = {
        "standardized_image": str(standardized_path) if standardized_path in written else None,
        "retinal_mask": str(mask_path) if mask_path in written else None,
        "result_json": str(result_path),
    }
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
