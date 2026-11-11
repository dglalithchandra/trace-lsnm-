"""Deterministic, bounded camera-style perturbations for robustness training.

These transforms create paired views for consistency learning. They are not
used as a claim that a real camera follows any one synthetic shift.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StressParameters:
    red_gain: float
    green_gain: float
    blue_gain: float
    saturation_gain: float
    brightness_shift: float
    contrast_gain: float
    gamma: float
    vignette_strength: float
    blur_sigma: float
    noise_sigma_255: float
    jpeg_quality: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactStressParameters:
    """One isolated artifact with a deterministic mild/moderate/severe level."""

    style: str
    severity: str
    glare_radius_fraction: float = 0.0
    glare_opacity: float = 0.0
    resolution_scale: float = 1.0
    field_of_view_zoom: float = 1.0

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _rng_for(image_id: str, seed_namespace: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed_namespace}:{image_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed)


def sample_stress_parameters(
    image_id: str,
    *,
    seed_namespace: str,
) -> StressParameters:
    rng = _rng_for(image_id, seed_namespace)
    style = int(rng.integers(0, 4))
    red_gain = blue_gain = saturation = contrast = gamma = 1.0
    brightness = vignette = blur = noise = 0.0
    jpeg_quality = 100
    if style == 0:  # colour/exposure response
        red_gain = float(rng.uniform(0.93, 1.07))
        blue_gain = float(rng.uniform(0.93, 1.07))
        saturation = float(rng.uniform(0.92, 1.08))
        brightness = float(rng.uniform(-0.035, 0.035))
        gamma = float(rng.uniform(0.93, 1.07))
    elif style == 1:  # tonal response and contrast
        brightness = float(rng.uniform(-0.045, 0.045))
        contrast = float(rng.uniform(0.90, 1.10))
        gamma = float(rng.uniform(0.90, 1.10))
    elif style == 2:  # illumination falloff
        vignette = float(rng.uniform(0.08, 0.22))
        brightness = float(rng.uniform(0.0, 0.025))
        saturation = float(rng.uniform(0.95, 1.05))
    else:  # optics, sensor and compression
        blur = float(rng.uniform(0.25, 0.75))
        noise = float(rng.uniform(0.5, 2.5))
        jpeg_quality = int(rng.integers(82, 96))
        contrast = float(rng.uniform(0.95, 1.05))
    return StressParameters(
        red_gain=red_gain,
        green_gain=1.0,
        blue_gain=blue_gain,
        saturation_gain=saturation,
        brightness_shift=brightness,
        contrast_gain=contrast,
        gamma=gamma,
        vignette_strength=vignette,
        blur_sigma=blur,
        noise_sigma_255=noise,
        jpeg_quality=jpeg_quality,
    )


def sample_artifact_stress_parameters(
    image_id: str,
    *,
    style: str,
    seed_namespace: str,
) -> ArtifactStressParameters:
    """Select a reproducible severity while keeping artifact types isolated."""

    if style not in {"glare", "resolution_loss", "narrow_field_of_view"}:
        raise ValueError(f"unsupported_artifact_stress_style:{style}")
    rng = _rng_for(image_id, f"{seed_namespace}:{style}:severity")
    index = int(rng.integers(0, 3))
    severity = ("mild", "moderate", "severe")[index]
    if style == "glare":
        return ArtifactStressParameters(
            style=style,
            severity=severity,
            glare_radius_fraction=(0.035, 0.055, 0.080)[index],
            glare_opacity=(0.45, 0.65, 0.85)[index],
        )
    if style == "resolution_loss":
        return ArtifactStressParameters(
            style=style,
            severity=severity,
            resolution_scale=(0.70, 0.50, 0.35)[index],
        )
    return ArtifactStressParameters(
        style=style,
        severity=severity,
        field_of_view_zoom=(1.08, 1.16, 1.25)[index],
    )


def apply_camera_stress(
    image: np.ndarray,
    mask: np.ndarray,
    parameters: StressParameters,
    *,
    image_id: str,
    seed_namespace: str,
) -> np.ndarray:
    """Apply a deterministic mild shift while preserving size and black mask."""

    valid = mask > 0
    rgb = image.astype(np.float32) / 255.0
    gains = np.asarray(
        [parameters.red_gain, parameters.green_gain, parameters.blue_gain],
        dtype=np.float32,
    )
    rgb *= gains.reshape(1, 1, 3)
    rgb = np.power(np.clip(rgb, 0.0, 1.0), parameters.gamma)
    rgb = (
        (rgb - 0.5) * parameters.contrast_gain
        + 0.5
        + parameters.brightness_shift
    )
    rgb = np.clip(rgb, 0.0, 1.0)

    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(
        np.float32
    )
    hsv[..., 1] = np.clip(hsv[..., 1] * parameters.saturation_gain, 0, 255)
    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    if parameters.vignette_strength > 0:
        height, width = mask.shape
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        center_x = float(xx[valid].mean())
        center_y = float(yy[valid].mean())
        radius = max(float(np.sqrt((xx[valid] - center_x) ** 2 + (yy[valid] - center_y) ** 2).max()), 1.0)
        radial = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2) / radius
        falloff = 1.0 - parameters.vignette_strength * np.clip(radial, 0, 1) ** 2
        shifted *= falloff[..., None]

    if parameters.blur_sigma > 0:
        shifted = cv2.GaussianBlur(
            shifted, (0, 0), sigmaX=parameters.blur_sigma, sigmaY=parameters.blur_sigma
        )
    if parameters.noise_sigma_255 > 0:
        rng = _rng_for(image_id, f"{seed_namespace}:noise")
        shifted += rng.normal(
            0.0, parameters.noise_sigma_255, size=shifted.shape
        ).astype(np.float32)
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    if parameters.jpeg_quality < 100:
        bgr = cv2.cvtColor(shifted, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, parameters.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("camera_stress_jpeg_encoding_failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        shifted = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    shifted[~valid] = 0
    return shifted


def apply_artifact_stress(
    image: np.ndarray,
    mask: np.ndarray,
    parameters: ArtifactStressParameters,
    *,
    image_id: str,
    seed_namespace: str,
) -> np.ndarray:
    """Apply one isolated camera artifact and preserve the circular mask."""

    valid = mask > 0
    height, width = mask.shape
    if image.shape[:2] != mask.shape:
        raise ValueError("artifact_stress_image_mask_shape_mismatch")
    output = image.copy()
    if parameters.style == "glare":
        rng = _rng_for(image_id, f"{seed_namespace}:glare:location")
        yy, xx = np.nonzero(valid)
        center_x = float(xx.mean())
        center_y = float(yy.mean())
        retina_radius = max(
            float(np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2).max()),
            1.0,
        )
        angle = float(rng.uniform(0, 2 * np.pi))
        distance = float(rng.uniform(0.15, 0.60)) * retina_radius
        glare_center = (
            int(round(center_x + np.cos(angle) * distance)),
            int(round(center_y + np.sin(angle) * distance)),
        )
        radius = max(2, int(round(parameters.glare_radius_fraction * width)))
        alpha = np.zeros((height, width), dtype=np.float32)
        cv2.ellipse(
            alpha,
            glare_center,
            (radius, max(2, int(round(radius * 0.65)))),
            float(rng.uniform(0, 180)),
            0,
            360,
            1.0,
            -1,
        )
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(radius * 0.30, 1.0))
        alpha *= float(parameters.glare_opacity)
        highlight = np.asarray([255.0, 250.0, 240.0], dtype=np.float32)
        output = np.clip(
            output.astype(np.float32) * (1.0 - alpha[..., None])
            + highlight * alpha[..., None],
            0,
            255,
        ).astype(np.uint8)
    elif parameters.style == "resolution_loss":
        scale = float(parameters.resolution_scale)
        reduced_width = max(16, int(round(width * scale)))
        reduced_height = max(16, int(round(height * scale)))
        reduced = cv2.resize(
            output, (reduced_width, reduced_height), interpolation=cv2.INTER_AREA
        )
        output = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)
    elif parameters.style == "narrow_field_of_view":
        zoom = float(parameters.field_of_view_zoom)
        crop_width = max(16, int(round(width / zoom)))
        crop_height = max(16, int(round(height / zoom)))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        cropped = output[top : top + crop_height, left : left + crop_width]
        output = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError(f"unsupported_artifact_stress_style:{parameters.style}")
    output[~valid] = 0
    return output
