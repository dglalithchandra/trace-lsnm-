"""Integrated, evidence-locked TRACE camera module v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trace_models import TraceRFMiDScorer
from trace_preprocessing.types import PreprocessingResult


MODULE_VERSION = "trace-camera-module-v1.0-research"


@dataclass(frozen=True)
class TraceCameraOutput:
    report: dict[str, Any]
    standardized_image: np.ndarray | None
    retinal_mask: np.ndarray | None


class TraceCameraModule:
    """Run the selected geometry, diagnostic QC and research disease scorer."""

    def __init__(
        self,
        workspace: Path | str,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.scorer = TraceRFMiDScorer(self.workspace, device=device)
        decision_path = (
            self.workspace / "outputs/camera_module_decision/current_decision.json"
        )
        self.decision = json.loads(decision_path.read_text(encoding="utf-8"))

    def process(
        self,
        image: np.ndarray,
        *,
        image_id: str = "unidentified",
    ) -> TraceCameraOutput:
        preprocessing = self.scorer.preprocessor.process(image, image_id=image_id)
        scoring = self.scorer.score_preprocessed(preprocessing)
        report = self._report(preprocessing, scoring)
        return TraceCameraOutput(
            report=report,
            standardized_image=(
                preprocessing.processed_image.copy()
                if preprocessing.processed_image is not None
                else None
            ),
            retinal_mask=(
                preprocessing.retinal_mask.copy()
                if preprocessing.retinal_mask is not None
                else None
            ),
        )

    def _report(
        self,
        preprocessing: PreprocessingResult,
        scoring: dict[str, Any],
    ) -> dict[str, Any]:
        audit = preprocessing.audit_dict()
        return {
            "module_version": MODULE_VERSION,
            "image_id": preprocessing.image_id,
            "status": scoring["status"],
            "preprocessing_status": preprocessing.status.value,
            "active_pipeline": self.decision["active_pipeline"],
            "disabled_experimental_components": self.decision[
                "disabled_by_default"
            ],
            "quality_control": {
                "status": preprocessing.status.value,
                "reasons": list(preprocessing.reasons),
                "warnings": list(preprocessing.warnings),
                "measurements": (
                    preprocessing.quality_before.to_dict()
                    if preprocessing.quality_before is not None
                    else None
                ),
                "threshold_status": (
                    "development diagnostics; not clinically validated acceptance limits"
                ),
            },
            "geometry": {
                "transform": audit["geometry_transform"],
                "metadata": audit["metadata"],
                "output_shape": (
                    list(preprocessing.processed_image.shape)
                    if preprocessing.processed_image is not None
                    else None
                ),
            },
            "disease_model": {
                "model_id": scoring.get("model_id"),
                "status": scoring["status"],
                "score_interpretation": scoring["score_interpretation"],
                "decision_interpretation": scoring.get("decision_interpretation"),
                "scores": scoring["scores"],
            },
            "current_claim": self.decision["current_claim"],
            "claims_not_supported": self.decision["claims_not_supported"],
            "research_only": True,
        }
