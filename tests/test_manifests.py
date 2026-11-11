from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trace_preprocessing.loader import ManifestImageLoader


WORKSPACE = Path(__file__).resolve().parents[1]
MANIFESTS = WORKSPACE / "data" / "manifests"


def test_source_manifest_validation_passed() -> None:
    report = json.loads((MANIFESTS / "manifest_validation.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["canonical_records"] == 3797
    assert report["confirmed_duplicate_groups"] == 56


def test_analysis_groups_never_cross_final_splits() -> None:
    manifest = pd.read_csv(MANIFESTS / "trace_analysis_manifest.csv")
    assert len(manifest) == 3797
    assert manifest["record_id"].is_unique
    assert set(manifest["final_split"]) == {"train", "validation", "test"}
    assert manifest.groupby("split_group_id")["final_split"].nunique().max() == 1


def test_camera_evidence_is_named_without_hiding_rfmid_mismatch() -> None:
    manifest = pd.read_csv(
        MANIFESTS / "master_manifest.csv", dtype=str, keep_default_na=False
    )
    rfmid = manifest[manifest["dataset"] == "RFMiD"]
    counts = rfmid["camera_model"].replace("", "unresolved").value_counts().to_dict()
    assert counts == {
        "TOPCON 3D OCT-2000": 2427,
        "unresolved": 497,
        "TOPCON TRC-NW300": 276,
    }
    unresolved = rfmid[rfmid["camera_model"] == ""]
    assert set(unresolved["camera_domain"]) == {
        "rfmid_resolution_4288x2848_unresolved"
    }
    idrid = manifest[manifest["dataset"].str.startswith("IDRiD")]
    assert set(idrid["camera_model"]) == {"Kowa VX-10 alpha"}


def test_fives_release_is_deduplicated_and_bad_masks_flagged() -> None:
    manifest = pd.read_csv(MANIFESTS / "fives_manifest.csv")
    summary = json.loads(
        (MANIFESTS / "fives_manifest_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS"
    assert len(manifest) == 800
    assert manifest["image_sha256"].nunique() == 794
    assert manifest["is_primary_analysis_record"].sum() == 794
    assert summary["exact_duplicate_groups"] == 6
    assert summary["empty_vessel_masks"] == 2
    assert manifest["mask_is_nonempty"].sum() == 798


def test_palm_manifest_is_checksum_verified_and_leakage_safe() -> None:
    manifest = pd.read_csv(
        MANIFESTS / "palm_manifest.csv", dtype=str, keep_default_na=False
    )
    summary = json.loads(
        (MANIFESTS / "palm_manifest_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS"
    assert summary["archive_md5"] == "28636601413706884ea89ec1be835e94"
    assert len(manifest) == 1200
    assert manifest["image_sha256"].nunique() == 1184
    primary = manifest[manifest["is_primary_analysis_record"] == "True"]
    assert len(primary) == 1184
    assert primary["image_sha256"].is_unique
    assert set(primary["effective_split"]) == {"train", "validation", "test"}
    excluded = manifest[manifest["is_primary_analysis_record"] == "False"]
    assert len(excluded) == 16
    assert set(excluded["effective_split"]) == {"excluded"}
    assert summary["cross_official_split_exact_duplicate_groups"] == 11
    assert summary["primary_records_by_effective_split"] == {
        "train": 391,
        "validation": 396,
        "test": 397,
    }
    assert summary["records_by_camera_model"] == {
        "ZEISS Visucam 500": 1047,
        "Canon CR-2": 153,
    }


def test_palm_loader_uses_effective_split_and_palm_masks() -> None:
    path = MANIFESTS / "palm_manifest.csv"
    with ManifestImageLoader(WORKSPACE, path) as loader:
        assert len(loader.record_ids(split="train")) == 391
        assert len(loader.record_ids(split="validation")) == 396
        assert len(loader.record_ids(split="excluded")) == 16
        try:
            loader.record_ids(split="test")
        except PermissionError:
            pass
        else:
            raise AssertionError("PALM test split was accessible without allow_test")

        eligible = loader.manifest[
            loader.manifest["_effective_split"].eq("train")
            & loader.manifest["disc_mask"].ne("")
        ]
        record_id = eligible.iloc[0]["record_id"]
        image, row = loader.load_image(record_id)
        masks = loader.load_annotation_masks(record_id)
        assert image.ndim == 3 and image.shape[2] == 3
        assert "optic_disc" in masks
        assert masks["optic_disc"].shape == image.shape[:2]
        assert row["camera_model"] in {"ZEISS Visucam 500", "Canon CR-2"}
