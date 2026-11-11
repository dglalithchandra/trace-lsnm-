"""Record visually reviewed near-copy pairs and freeze conservative EDID queries.

The contact sheets for the geometry-positive pairs must be inspected before
running this script. This script records that engineering review; it does not
substitute for clinical adjudication or patient IDs.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/edid_exploratory_manifest.csv"
DETAILS = (
    (ROOT / "data/manifests/edid_near_duplicate_detail.csv", "ordinary"),
    (ROOT / "data/manifests/edid_excluded_conflict_near_detail.csv", "excluded_conflict"),
)
REVIEW = ROOT / "data/manifests/edid_near_duplicate_visual_review.csv"
OUTPUT = ROOT / "data/manifests/edid_exploratory_manifest_v2.csv"
SUMMARY = ROOT / "data/manifests/edid_exploratory_manifest_v2_summary.json"
QUERY = {"Macular Scar", "Retinitis Pigmentosa", "Disc Edema", "Healthy"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    rows = read_csv(MANIFEST)
    by_member = {row["archive_member"]: row for row in rows}
    parent = {row["archive_member"]: row["archive_member"] for row in rows if row["analysis_eligible"] == "True"}

    def find(member: str) -> str:
        while parent[member] != member:
            parent[member] = parent[parent[member]]
            member = parent[member]
        return member

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)

    reviewed = []
    excluded_touches = set()
    for detail_path, kind in DETAILS:
        passes = [row for row in read_csv(detail_path) if row["candidate_source"] == "edid" and row["passes_local_geometry_screen"] == "True"]
        passes.sort(key=lambda row: (row["query_class"] == row["candidate_class"], int(row["ransac_inliers"])))
        for index, row in enumerate(passes):
            left = row["query_member"]
            right = row["candidate_id"].removeprefix("edid:")
            if left not in parent:
                raise RuntimeError(f"Reviewed query is ineligible: {left}")
            if kind == "ordinary":
                if right not in parent:
                    raise RuntimeError(f"Ordinary reviewed candidate is ineligible: {right}")
                union(left, right)
            else:
                if by_member[right]["status"] != "cross_class_exact_copy":
                    raise RuntimeError(f"Expected excluded class conflict: {right}")
                excluded_touches.add(left)
            reviewed.append({
                "query_member": left,
                "candidate_member": right,
                "review_kind": kind,
                "review_decision": "same_retina_engineering_visual_review",
                "review_basis": "matching_vessels_disc_and_retinal_detail_on_contact_sheet",
                "reviewer_type": "AI_engineering_visual_review_not_clinical",
                "contact_sheet": f"{'edid_excluded_conflict_overlap_review' if kind == 'excluded_conflict' else 'edid_overlap_review'}/review_{index // 8 + 1:02d}.jpg",
                "ransac_inliers": row["ransac_inliers"],
            })
    if Counter(item["review_kind"] for item in reviewed) != {"ordinary": 66, "excluded_conflict": 10}:
        raise RuntimeError("Reviewed-pair count changed; contact sheets need reinspection")

    groups: dict[str, list[str]] = defaultdict(list)
    for member in parent:
        groups[find(member)].append(member)
    excluded_roots = {find(member) for member in excluded_touches}
    updated = []
    for row in rows:
        item = dict(row)
        member = row["archive_member"]
        if member in parent:
            group = groups[find(member)]
            classes = {by_member[other]["class_folder"] for other in group}
            if find(member) in excluded_roots:
                item["status"] = "near_copy_of_excluded_exact_class_conflict"
                item["analysis_eligible"] = "False"
            elif len(classes) > 1:
                item["status"] = "near_duplicate_cross_class_conflict"
                item["analysis_eligible"] = "False"
            elif len(group) > 1 and member != min(group):
                item["status"] = "near_duplicate_copy_within_class"
                item["analysis_eligible"] = "False"
        updated.append(item)

    with REVIEW.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reviewed[0]))
        writer.writeheader()
        writer.writerows(reviewed)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(updated[0]))
        writer.writeheader()
        writer.writerows(updated)
    summary = {
        "status": "FROZEN_EXPLORATORY_QUERY_MANIFEST",
        "engineering_visual_reviewed_pairs": len(reviewed),
        "remaining_eligible_fundus_files": sum(item["analysis_eligible"] == "True" for item in updated),
        "query_counts": {name: sum(item["analysis_eligible"] == "True" and item["class_folder"] == name for item in updated) for name in sorted(QUERY)},
        "new_exclusion_counts": {name: sum(item["status"] == name for item in updated) for name in (
            "near_copy_of_excluded_exact_class_conflict", "near_duplicate_cross_class_conflict", "near_duplicate_copy_within_class"
        )},
        "caveats": ["Hash candidate screen is not exhaustive", "Patient IDs and per-image camera IDs unavailable", "Folder labels have not been clinically adjudicated", "RFMiD support screen found no confirmed overlap among priority pairs; residual overlap is possible"],
        "protected_test_images_opened": 0,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
