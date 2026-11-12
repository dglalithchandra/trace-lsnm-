"""Compare fixed plain-crop and full-field SSL views before model training."""

from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trace_preprocessing import PreprocessingConfig, ProcessingStatus, TracePreprocessor

MANIFEST = ROOT / "data/protocols/trace_ssl_v1/unlabeled_train_manifest.csv"
ARCHIVE = ROOT / "A. RFMiD_All_Classes_Dataset.zip"
OUTPUT = ROOT / "data/protocols/trace_ssl_v1/view_preflight_summary.json"
SHEET = ROOT / "tmp/trace_ssl_view_preflight.jpg"
SEED = 20261111
SAMPLE = 60


def view_plain(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    side = len(image)
    scale = float(rng.uniform(0.70, 1.0))
    crop = max(1, int(round(side * scale**0.5)))
    x = int(rng.integers(0, side - crop + 1))
    y = int(rng.integers(0, side - crop + 1))
    kept = float((mask[y:y + crop, x:x + crop] > 0).sum() / max((mask > 0).sum(), 1))
    out = cv2.resize(image[y:y + crop, x:x + crop], (side, side), interpolation=cv2.INTER_LINEAR)
    channel_gain = rng.uniform(0.97, 1.03, size=(1, 1, 3))
    out = np.clip(out.astype(np.float32) * channel_gain, 0, 255).astype(np.uint8)
    return out, kept


def view_full_field(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    side = len(image)
    angle = float(rng.uniform(-5, 5))
    dx, dy = rng.integers(-4, 5, size=2).tolist()
    transform = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    transform[:, 2] += (dx, dy)
    out = cv2.warpAffine(image, transform, (side, side), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mapped_mask = cv2.warpAffine(mask.astype(np.uint8), transform, (side, side), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    kept = min(1.0, float((mapped_mask > 0).sum() / max((mask > 0).sum(), 1)))
    channel_gain = rng.uniform(0.97, 1.03, size=(1, 1, 3))
    out = np.clip(out.astype(np.float32) * channel_gain, 0, 255).astype(np.uint8)
    return out, kept


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1917 or set(rows[0]) != {"record_id", "archive_member", "image_sha256"}:
        raise RuntimeError("Unlabeled SSL manifest changed")
    rng = np.random.default_rng(SEED)
    selected = [rows[index] for index in np.linspace(0, len(rows) - 1, SAMPLE, dtype=int)]
    pipeline = TracePreprocessor(PreprocessingConfig(apply_photometric_correction=False))
    observations = []
    sheet = Image.new("RGB", (448 * 3, 448 * 8), "white")
    with zipfile.ZipFile(ARCHIVE) as archive:
        for index, row in enumerate(selected):
            bgr = cv2.imdecode(np.frombuffer(archive.read(row["archive_member"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("RFMiD training image decode failed")
            result = pipeline.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image_id=row["record_id"])
            if result.status == ProcessingStatus.REJECT or result.processed_image is None or result.retinal_mask is None:
                raise RuntimeError(f"RFMiD training image rejected: {row['record_id']}")
            image = result.processed_image
            mask = result.retinal_mask
            plain, plain_kept = view_plain(image, mask, rng)
            full, full_kept = view_full_field(image, mask, rng)
            observations.append((plain_kept, full_kept))
            if index < 8:
                for column, frame in enumerate((image, plain, full)):
                    sheet.paste(Image.fromarray(frame), (column * 448, index * 448))
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(SHEET, quality=90)
    observed = np.asarray(observations)
    summary = {"status": "GEOMETRIC_VIEW_PREFLIGHT_ONLY_NOT_LESION_PRESERVATION", "sampled_training_images": len(selected), "seed": SEED, "plain_crop_retinal_mask_retention_mean": float(observed[:, 0].mean()), "plain_crop_retinal_mask_retention_min": float(observed[:, 0].min()), "full_field_retinal_mask_retention_mean": float(observed[:, 1].mean()), "full_field_retinal_mask_retention_min": float(observed[:, 1].min()), "contact_sheet": str(SHEET.relative_to(ROOT)), "warning": "Retinal field retention is a geometric measure; it does not prove preservation of every lesion or medical finding.", "protected_test_images_opened": 0}
    OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
