"""Small platform preflight before assigning TRACE work to a Mac MPS device.

This checks only PyTorch installation and basic MPS forward/backward. It does
not establish that the CUDA-only TRACE trainer has been ported or benchmarked.
"""

from __future__ import annotations

import json
import platform
import sys
import time


def main() -> None:
    result: dict[str, object] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    try:
        import torch
    except ImportError as error:
        result.update({"torch_import": "FAILED", "error": str(error)})
        print(json.dumps(result, indent=2))
        return
    result.update({
        "torch": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    })
    if result["mps_available"]:
        try:
            start = time.perf_counter()
            tensor = torch.randn(64, 64, device="mps", requires_grad=True)
            loss = tensor.square().mean() + (tensor @ tensor).square().mean()
            loss.backward()
            torch.mps.synchronize()
            result["basic_mps_forward_backward"] = "PASS"
            result["basic_mps_seconds"] = round(time.perf_counter() - start, 4)
            result["gradient_finite"] = bool(torch.isfinite(tensor.grad).all().item())
        except Exception as error:  # Preserve the error for setup diagnosis.
            result["basic_mps_forward_backward"] = "FAILED"
            result["error"] = f"{type(error).__name__}: {error}"
    else:
        result["basic_mps_forward_backward"] = "NOT_RUN"
    result["trace_ssl_trainer_mps_ready"] = False
    result["note"] = "The existing TRACE SSL trainer uses CUDA-specific autocast, scaling and RNG APIs."
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
