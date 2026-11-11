"""End-to-end model-independent TRACE preprocessing pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import PreprocessingConfig
from .geometry import normalize_geometry, transform_mask
from .measurements import measure_photometry, measure_quality
from .photometric import apply_corrections, derive_corrections
from .preservation import evaluate_annotation_preservation
from .quality import compare_images, decide_quality
from .types import (
    CorrectionParameters,
    PreprocessingResult,
    ProcessingStatus,
    ReferenceProfile,
)
from .validation import detect_fundus_field, validate_rgb_fundus_candidate


class TracePreprocessor:
    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def _rejected(
        self,
        image_id: str,
        reasons: list[str],
        warnings: list[str],
        original_image: np.ndarray | None,
        metadata: dict[str, Any],
    ) -> PreprocessingResult:
        return PreprocessingResult(
            image_id=image_id,
            status=ProcessingStatus.REJECT,
            reasons=reasons,
            warnings=warnings,
            original_image=original_image.copy() if original_image is not None else None,
            geometry_only_image=None,
            processed_image=None,
            retinal_mask=None,
            geometry_transform=None,
            stats_before=None,
            stats_after=None,
            quality_before=None,
            quality_after=None,
            corrections=None,
            comparison=None,
            annotation_preservation={},
            preprocessing_version=self.config.version,
            metadata=metadata,
        )

    def process(
        self,
        image: np.ndarray,
        *,
        image_id: str,
        reference: ReferenceProfile | None = None,
        annotation_masks: dict[str, np.ndarray] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PreprocessingResult:
        metadata = dict(metadata or {})
        metadata.update(
            {
                "modality": self.config.modality,
                "policy_status": self.config.policy_status,
                "reference_profile_id": reference.profile_id if reference else None,
            }
        )
        original = image.copy() if self.config.preserve_original else None
        validation = validate_rgb_fundus_candidate(image, self.config.geometry)
        if not validation.valid:
            return self._rejected(
                image_id,
                list(validation.reasons),
                list(validation.warnings),
                original,
                metadata,
            )

        try:
            detection = detect_fundus_field(image, self.config.geometry)
            geometry_only, retinal_mask, transform = normalize_geometry(
                image, detection, self.config.geometry
            )
        except (ValueError, RuntimeError) as error:
            return self._rejected(
                image_id,
                [str(error)],
                list(validation.warnings),
                original,
                metadata,
            )

        stats_before = measure_photometry(geometry_only, retinal_mask)
        quality_before = measure_quality(geometry_only, retinal_mask)
        corrections = CorrectionParameters()
        processed = geometry_only.copy()
        warnings = list(validation.warnings)
        if self.config.apply_photometric_correction:
            if reference is None:
                warnings.append("photometric_correction_skipped_without_reference")
            else:
                corrections = derive_corrections(
                    stats_before,
                    reference,
                    self.config.photometric_bounds,
                )
                processed = apply_corrections(geometry_only, retinal_mask, corrections)

        stats_after = measure_photometry(processed, retinal_mask)
        quality_after = measure_quality(processed, retinal_mask)
        comparison = compare_images(
            geometry_only,
            processed,
            retinal_mask,
            stats_before,
            stats_after,
        )
        annotation_preservation = {
            name: evaluate_annotation_preservation(
                geometry_only,
                processed,
                annotation,
                transform,
                retinal_mask,
                annotation_name=name,
            )
            for name, annotation in (annotation_masks or {}).items()
        }
        status, reasons, quality_warnings = decide_quality(
            comparison,
            quality_before,
            quality_after,
            self.config.quality,
        )
        warnings.extend(quality_warnings)
        if status == ProcessingStatus.ACCEPT and warnings:
            status = ProcessingStatus.WARNING

        metadata.update(
            {
                "fundus_detection_confidence": detection.confidence,
                "source_retinal_coverage": detection.retinal_coverage,
                "fundus_center_x": detection.center_x,
                "fundus_center_y": detection.center_y,
                "fundus_radius": detection.radius,
                "fundus_bbox_x": detection.bbox_x,
                "fundus_bbox_y": detection.bbox_y,
                "fundus_bbox_width": detection.bbox_width,
                "fundus_bbox_height": detection.bbox_height,
            }
        )
        source_valid = detection.mask > 0
        x0 = max(0, transform.crop_x)
        y0 = max(0, transform.crop_y)
        x1 = min(transform.source_width, transform.crop_x + transform.crop_size)
        y1 = min(transform.source_height, transform.crop_y + transform.crop_size)
        source_pixels = int(source_valid.sum())
        retained_pixels = int(source_valid[y0:y1, x0:x1].sum())
        transformed_detection = transform_mask(detection.mask, transform)
        transformed_valid = transformed_detection > 0
        transformed_pixels = int(transformed_valid.sum())
        common_mask_lost = int((transformed_valid & (retinal_mask == 0)).sum())
        intersection_area = max(0, x1 - x0) * max(0, y1 - y0)
        metadata.update(
            {
                "source_crop_retinal_loss_fraction": (
                    1.0 - retained_pixels / source_pixels if source_pixels else 1.0
                ),
                "common_mask_retinal_loss_fraction": (
                    common_mask_lost / transformed_pixels if transformed_pixels else 1.0
                ),
                "crop_padding_fraction": (
                    1.0 - intersection_area / (transform.crop_size * transform.crop_size)
                ),
                "normalized_detected_retina_fill_fraction": (
                    int((transformed_valid & (retinal_mask > 0)).sum())
                    / max(int((retinal_mask > 0).sum()), 1)
                ),
            }
        )
        return PreprocessingResult(
            image_id=image_id,
            status=status,
            reasons=reasons,
            warnings=sorted(set(warnings)),
            original_image=original,
            geometry_only_image=geometry_only,
            processed_image=processed,
            retinal_mask=retinal_mask,
            geometry_transform=transform,
            stats_before=stats_before,
            stats_after=stats_after,
            quality_before=quality_before,
            quality_after=quality_after,
            corrections=corrections,
            comparison=comparison,
            annotation_preservation=annotation_preservation,
            preprocessing_version=self.config.version,
            metadata=metadata,
        )
