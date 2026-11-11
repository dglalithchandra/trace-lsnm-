"""Annotation-aware checks for rare lesions, vessels and optic-disc regions."""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import transform_mask
from .types import AnnotationPreservation, GeometryTransform


def evaluate_annotation_preservation(
    geometry_only: np.ndarray,
    processed: np.ndarray,
    source_annotation: np.ndarray,
    transform: GeometryTransform,
    retinal_mask: np.ndarray,
    *,
    annotation_name: str,
) -> AnnotationPreservation:
    if source_annotation.ndim == 3:
        source_annotation = source_annotation[..., 0]
    if source_annotation.shape != (transform.source_height, transform.source_width):
        raise ValueError(f"annotation_shape_mismatch:{annotation_name}")
    source_foreground = source_annotation > 0
    source_count = int(source_foreground.sum())

    x0 = max(0, transform.crop_x)
    y0 = max(0, transform.crop_y)
    x1 = min(transform.source_width, transform.crop_x + transform.crop_size)
    y1 = min(transform.source_height, transform.crop_y + transform.crop_size)
    retained_source_count = int(source_foreground[y0:y1, x0:x1].sum())
    clipped_fraction = (
        1.0 - retained_source_count / source_count if source_count > 0 else 0.0
    )

    transformed = transform_mask(source_annotation, transform, retinal_mask)
    region = transformed > 0
    transformed_count = int(region.sum())
    if transformed_count == 0:
        return AnnotationPreservation(
            annotation_name=annotation_name,
            source_foreground_pixels=source_count,
            transformed_foreground_pixels=0,
            source_region_clipped_fraction=float(clipped_fraction),
            mean_absolute_change_in_region=1.0 if source_count else 0.0,
            p95_absolute_change_in_region=1.0 if source_count else 0.0,
            bright_clip_fraction_after=0.0,
            dark_clip_fraction_after=0.0,
        )

    difference = np.abs(processed.astype(np.float32) - geometry_only.astype(np.float32)) / 255.0
    region_difference = difference[region]
    gray_after = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return AnnotationPreservation(
        annotation_name=annotation_name,
        source_foreground_pixels=source_count,
        transformed_foreground_pixels=transformed_count,
        source_region_clipped_fraction=float(clipped_fraction),
        mean_absolute_change_in_region=float(region_difference.mean()),
        p95_absolute_change_in_region=float(np.percentile(region_difference, 95)),
        bright_clip_fraction_after=float(np.mean(gray_after[region] >= 0.99)),
        dark_clip_fraction_after=float(np.mean(gray_after[region] <= 0.01)),
    )
