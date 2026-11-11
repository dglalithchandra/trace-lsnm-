"""Geometric normalisation and annotation transformation."""

from __future__ import annotations

import cv2
import numpy as np

from .config import GeometryConfig
from .types import FundusDetection, GeometryTransform


def _square_crop_bounds(
    image: np.ndarray, detection: FundusDetection, config: GeometryConfig
) -> tuple[int, int, int]:
    margin = max(detection.radius * config.crop_margin_fraction, 1.0)
    half_size = detection.radius + margin
    size = max(int(np.ceil(2.0 * half_size)), 2)
    x = int(np.floor(detection.center_x - size / 2.0))
    y = int(np.floor(detection.center_y - size / 2.0))
    return x, y, size


def _crop_with_padding(
    array: np.ndarray, x: int, y: int, size: int, fill_value: int = 0
) -> tuple[np.ndarray, int, int]:
    height, width = array.shape[:2]
    source_x0 = max(0, x)
    source_y0 = max(0, y)
    source_x1 = min(width, x + size)
    source_y1 = min(height, y + size)
    pad_left = max(0, -x)
    pad_top = max(0, -y)
    pad_right = max(0, x + size - width)
    pad_bottom = max(0, y + size - height)
    crop = array[source_y0:source_y1, source_x0:source_x1]
    border_value: int | tuple[int, int, int]
    border_value = (fill_value, fill_value, fill_value) if array.ndim == 3 else fill_value
    crop = cv2.copyMakeBorder(
        crop,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )
    if crop.shape[0] != size or crop.shape[1] != size:
        raise RuntimeError("geometric_crop_size_mismatch")
    return crop, pad_left, pad_top


def common_circular_mask(output_size: int, radius_fraction: float) -> np.ndarray:
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    center = (output_size // 2, output_size // 2)
    radius = int(round(output_size * radius_fraction))
    cv2.circle(mask, center, radius, 255, thickness=cv2.FILLED)
    return mask


def normalize_geometry(
    image: np.ndarray,
    detection: FundusDetection,
    config: GeometryConfig,
) -> tuple[np.ndarray, np.ndarray, GeometryTransform]:
    x, y, size = _square_crop_bounds(image, detection, config)
    crop, pad_left, pad_top = _crop_with_padding(image, x, y, size)
    resized = cv2.resize(
        crop,
        (config.output_size, config.output_size),
        interpolation=cv2.INTER_AREA if size > config.output_size else cv2.INTER_CUBIC,
    )
    mask = common_circular_mask(config.output_size, config.common_mask_radius_fraction)
    normalized = resized.copy()
    normalized[mask == 0] = 0
    transform = GeometryTransform(
        source_width=image.shape[1],
        source_height=image.shape[0],
        crop_x=x,
        crop_y=y,
        crop_size=size,
        pad_left=pad_left,
        pad_top=pad_top,
        output_size=config.output_size,
    )
    return normalized, mask, transform


def transform_mask(
    source_mask: np.ndarray,
    transform: GeometryTransform,
    output_retinal_mask: np.ndarray | None = None,
) -> np.ndarray:
    if source_mask.ndim == 3:
        source_mask = source_mask[..., 0]
    crop, _, _ = _crop_with_padding(
        source_mask,
        transform.crop_x,
        transform.crop_y,
        transform.crop_size,
    )
    transformed = cv2.resize(
        crop,
        (transform.output_size, transform.output_size),
        interpolation=cv2.INTER_NEAREST,
    )
    if output_retinal_mask is not None:
        transformed = transformed.copy()
        transformed[output_retinal_mask == 0] = 0
    return transformed
