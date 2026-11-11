"""Image-to-score interface for the research-only RFMiD baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor
from trace_preprocessing.adapters import prepare_dinov2_tensor
from trace_preprocessing.types import PreprocessingResult

from .dinov2 import load_dinov2_vits14
from .rfmid_dinov2 import (
    DECISION_TOLERANCE,
    load_rfmid_bundle,
    probabilities_from_embedding,
)


class TraceRFMiDScorer:
    """Apply geometry normalization, frozen DINOv2 and the RFMiD linear head.

    Returned values are development model scores, not clinical certainty. A
    binary decision is exposed only when the corresponding class had at least
    five positive validation examples for provisional threshold selection.
    """

    def __init__(
        self,
        workspace: Path | str,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder = load_dinov2_vits14(self.workspace, device=self.device)
        bundle_path = (
            self.workspace
            / "models"
            / "rfmid_dinov2_linear_v1"
            / "rfmid_dinov2_linear_v1.pt"
        )
        self.head, self.bundle = load_rfmid_bundle(bundle_path, device=self.device)
        if (
            self.bundle.get("preprocessing")
            != "geometry_only_trace-preprocessing-dev-v0.1"
        ):
            raise RuntimeError("RFMiD head preprocessing contract mismatch")
        self.preprocessor = TracePreprocessor(
            PreprocessingConfig(apply_photometric_correction=False)
        )

    def score_image(
        self,
        image: np.ndarray,
        *,
        image_id: str = "unidentified",
    ) -> dict[str, Any]:
        result = self.preprocessor.process(image, image_id=image_id)
        return self.score_preprocessed(result)

    def score_preprocessed(self, result: PreprocessingResult) -> dict[str, Any]:
        """Score one result from the scorer's frozen preprocessing contract."""

        image_id = result.image_id
        if result.status == ProcessingStatus.REJECT or result.processed_image is None:
            return {
                "image_id": image_id,
                "status": "rejected",
                "preprocessing_status": result.status.value,
                "reasons": list(result.reasons),
                "warnings": list(result.warnings),
                "scores": [],
                "score_interpretation": (
                    "No disease scores produced because preprocessing rejected the image."
                ),
                "research_only": True,
            }

        tensor = prepare_dinov2_tensor(result.processed_image).unsqueeze(0)
        tensor = tensor.to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            embedding = self.encoder(tensor)
        if embedding.shape != (1, int(self.bundle["embedding_width"])):
            raise RuntimeError(f"Unexpected DINOv2 embedding shape: {embedding.shape}")
        if not torch.isfinite(embedding).all():
            raise RuntimeError("Non-finite DINOv2 embedding")
        probabilities = probabilities_from_embedding(
            self.head, embedding.float(), self.bundle
        )[0].detach().cpu().numpy()

        rows: list[dict[str, Any]] = []
        for index, label in enumerate(self.bundle["class_order"]):
            threshold = self.bundle["thresholds"][index]
            rows.append(
                {
                    "label": label,
                    "score": float(probabilities[index]),
                    "threshold": float(threshold) if threshold is not None else None,
                    "positive": (
                        bool(
                            probabilities[index]
                            >= threshold - DECISION_TOLERANCE
                        )
                        if threshold is not None
                        else None
                    ),
                    "support_status": self.bundle["support_status"][label],
                    "training_positives": int(
                        self.bundle["training_positive_counts"][index]
                    ),
                    "validation_positives": int(
                        self.bundle["validation_positive_counts"][index]
                    ),
                }
            )
        rows.sort(key=lambda row: row["score"], reverse=True)
        return {
            "image_id": image_id,
            "status": "scored",
            "preprocessing_status": result.status.value,
            "reasons": list(result.reasons),
            "warnings": list(result.warnings),
            "score_interpretation": (
                "Development model score; not clinical certainty and not a reliably "
                "externally calibrated probability."
            ),
            "decision_interpretation": (
                "Positive/negative decisions use provisional RFMiD-validation thresholds "
                "that failed to transfer reliably to an external Paraguay dataset."
            ),
            "model_id": "frozen_dinov2_vits14_rfmid_linear_v1",
            "preprocessing_version": result.preprocessing_version,
            "scores": rows,
            "research_only": True,
        }
