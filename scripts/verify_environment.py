"""Verify the Python environment and GPU needed for TRACE preprocessing."""

from __future__ import annotations

import importlib
import json
import sys


MODULES = [
    "albumentations",
    "cv2",
    "imagehash",
    "kornia",
    "matplotlib",
    "numpy",
    "openpyxl",
    "optuna",
    "pandas",
    "PIL",
    "scipy",
    "seaborn",
    "skimage",
    "sklearn",
    "timm",
    "torch",
    "torchmetrics",
    "torchvision",
    "tqdm",
    "umap",
]


def main() -> None:
    versions: dict[str, str] = {}
    failures: dict[str, str] = {}
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "installed"))
        except Exception as error:  # pragma: no cover - diagnostic script
            failures[name] = repr(error)

    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
        raise SystemExit(1)

    import torch

    result = {
        "status": "PASS",
        "python": sys.version.split()[0],
        "modules": versions,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
