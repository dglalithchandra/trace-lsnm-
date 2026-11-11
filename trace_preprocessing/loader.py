"""Manifest-driven image loading without extracting the original ZIP archives."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


class ManifestImageLoader:
    MASK_COLUMN_CANDIDATES = {
        "microaneurysms": ("mask_microaneurysms",),
        "haemorrhages": ("mask_haemorrhages",),
        "hard_exudates": ("mask_hard_exudates",),
        "soft_exudates": ("mask_soft_exudates",),
        "optic_disc": ("mask_optic_disc", "disc_mask"),
        "retinal_atrophy": ("atrophy_mask",),
        "retinal_detachment": ("detachment_mask",),
    }

    def __init__(
        self,
        workspace: Path,
        manifest_path: Path | None = None,
        *,
        allow_test: bool = False,
    ) -> None:
        self.workspace = workspace.resolve()
        self.manifest_path = manifest_path or (
            self.workspace / "data" / "manifests" / "trace_analysis_manifest.csv"
        )
        self.allow_test = allow_test
        self._archives: dict[Path, zipfile.ZipFile] = {}
        self.manifest = pd.read_csv(self.manifest_path, dtype=str, keep_default_na=False)
        if not self.manifest["record_id"].is_unique:
            raise ValueError("manifest_record_ids_not_unique")
        if "effective_split" in self.manifest.columns:
            final_split = self.manifest["effective_split"]
        elif "final_split" in self.manifest.columns:
            final_split = self.manifest["final_split"]
        else:
            final_split = pd.Series("", index=self.manifest.index, dtype=str)
        source_split = (
            self.manifest["source_split"]
            if "source_split" in self.manifest.columns
            else pd.Series("", index=self.manifest.index, dtype=str)
        ).replace({"training": "train", "testing": "test"})
        self.manifest["_effective_split"] = final_split.where(
            final_split.ne(""), source_split
        )
        self._rows = self.manifest.set_index("record_id", drop=False)

    def _archive(self, archive_name: str) -> zipfile.ZipFile:
        path = (self.workspace / archive_name).resolve()
        if path not in self._archives:
            self._archives[path] = zipfile.ZipFile(path)
        return self._archives[path]

    def close(self) -> None:
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def __enter__(self) -> "ManifestImageLoader":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def row(self, record_id: str) -> dict[str, Any]:
        if record_id not in self._rows.index:
            raise KeyError(record_id)
        row = self._rows.loc[record_id].to_dict()
        if row.get("_effective_split") == "test" and not self.allow_test:
            raise PermissionError("test_split_access_requires_explicit_allow_test")
        return row

    def load_image(self, record_id: str) -> tuple[np.ndarray, dict[str, Any]]:
        row = self.row(record_id)
        archive_name = row.get("archive_path", "")
        archive_member = row.get("archive_member", "")
        image_path = row.get("image_path", "")
        if archive_name and archive_member:
            payload = self._archive(archive_name).read(archive_member)
            with Image.open(io.BytesIO(payload)) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return array, row
        if image_path:
            with Image.open(self.workspace / image_path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            return array, row
        raise ValueError(f"record_has_no_supported_image_location:{record_id}")

    def load_annotation_masks(self, record_id: str) -> dict[str, np.ndarray]:
        row = self.row(record_id)
        masks: dict[str, np.ndarray] = {}
        archive_name = row.get("archive_path", "")
        members: dict[str, str] = {}
        for name, columns in self.MASK_COLUMN_CANDIDATES.items():
            member = next((row.get(column, "") for column in columns if row.get(column, "")), "")
            if member:
                members[name] = member
        if members:
            if not archive_name:
                raise ValueError("archive_mask_members_without_archive_path")
            archive = self._archive(archive_name)
            for name, member in members.items():
                with Image.open(io.BytesIO(archive.read(member))) as mask:
                    masks[name] = np.asarray(mask.convert("L"), dtype=np.uint8)

        vessel_path = row.get("vessel_mask_path", "")
        if vessel_path:
            with Image.open(self.workspace / vessel_path) as mask:
                masks["vessels"] = np.asarray(mask.convert("L"), dtype=np.uint8)
        return masks

    def record_ids(
        self,
        *,
        split: str | None = None,
        dataset: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        frame = self.manifest
        if split is not None:
            if split == "test" and not self.allow_test:
                raise PermissionError("test_split_access_requires_explicit_allow_test")
            frame = frame[frame["_effective_split"] == split]
        if dataset is not None:
            frame = frame[frame["dataset"] == dataset]
        values = frame["record_id"].tolist()
        return values[:limit] if limit is not None else values
