"""Create conservative leakage-aware TRACE splits from the source manifest.

The source archives do not provide patient IDs.  This script therefore does
not claim patient-independent splitting.  It only guarantees that confirmed
duplicate photographs and conservatively linked IDRiD grading/segmentation
records stay in the same final split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


SOURCE_TO_FINAL = {"training": "train", "validation": "validation", "testing": "test"}
HOLDOUT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assign(workspace: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    source_path = workspace / "data" / "manifests" / "master_manifest.csv"
    manifest = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    check(manifest["record_id"].is_unique, "record_id must be unique")
    check(set(manifest["source_split"]) <= set(SOURCE_TO_FINAL), "Unexpected source split")

    record_ids = manifest["record_id"].tolist()
    known_ids = set(record_ids)
    union_find = UnionFind(record_ids)
    edge_type_by_pair: dict[frozenset[str], str] = {}

    for _, group in manifest[manifest["duplicate_group_id"].ne("")].groupby("duplicate_group_id"):
        members = sorted(group["record_id"].tolist())
        for member in members[1:]:
            union_find.union(members[0], member)
            edge_type_by_pair[frozenset((members[0], member))] = "confirmed_perceptual_duplicate"

    related = manifest[
        manifest["related_status"].eq("possible_related_same_source_number")
        & manifest["related_record_id"].ne("")
    ]
    for row in related.itertuples(index=False):
        check(row.related_record_id in known_ids, f"Unknown related record: {row.related_record_id}")
        union_find.union(row.record_id, row.related_record_id)
        edge_type_by_pair[frozenset((row.record_id, row.related_record_id))] = (
            "conservative_idrid_same_number_relation"
        )

    component_members: dict[str, list[str]] = defaultdict(list)
    for record_id in record_ids:
        component_members[union_find.find(record_id)].append(record_id)

    row_by_id = manifest.set_index("record_id", drop=False)
    component_info: dict[str, dict[str, object]] = {}
    ordered_components = sorted(component_members.values(), key=lambda members: min(members))
    for number, members in enumerate(ordered_components, start=1):
        source_finals = [SOURCE_TO_FINAL[row_by_id.at[member, "source_split"]] for member in members]
        final_split = max(source_finals, key=HOLDOUT_PRIORITY.__getitem__)
        has_duplicate = any(
            row_by_id.at[member, "duplicate_group_id"] != "" for member in members
        )
        has_related = any(
            row_by_id.at[member, "related_status"] == "possible_related_same_source_number"
            for member in members
        )
        relation_type = (
            "mixed_duplicate_and_conservative_relation"
            if has_duplicate and has_related
            else "confirmed_duplicate_group"
            if has_duplicate
            else "conservative_related_group"
            if has_related
            else "singleton"
        )
        info = {
            "split_group_id": f"split_group_{number:04d}",
            "split_group_size": len(members),
            "split_group_relation": relation_type,
            "final_split": final_split,
        }
        for member in members:
            component_info[member] = info

    manifest["split_group_id"] = manifest["record_id"].map(
        lambda record_id: component_info[record_id]["split_group_id"]
    )
    manifest["split_group_size"] = manifest["record_id"].map(
        lambda record_id: component_info[record_id]["split_group_size"]
    )
    manifest["split_group_relation"] = manifest["record_id"].map(
        lambda record_id: component_info[record_id]["split_group_relation"]
    )
    manifest["final_split"] = manifest["record_id"].map(
        lambda record_id: component_info[record_id]["final_split"]
    )
    manifest["source_split_normalized"] = manifest["source_split"].map(SOURCE_TO_FINAL)
    manifest["split_changed_from_source"] = (
        manifest["final_split"] != manifest["source_split_normalized"]
    )
    manifest["final_split_status"] = (
        "duplicate_and_related_group_aware;patient_independence_unverifiable"
    )

    check(manifest["final_split"].isin(HOLDOUT_PRIORITY).all(), "Blank or invalid final split")
    group_split_counts = manifest.groupby("split_group_id")["final_split"].nunique()
    check((group_split_counts == 1).all(), "A leakage group spans multiple final splits")

    changed = manifest[manifest["split_changed_from_source"]]
    relation_components = [members for members in ordered_components if len(members) > 1]
    summary: dict[str, object] = {
        "status": "PASS",
        "records": len(manifest),
        "split_groups": len(ordered_components),
        "multi_record_split_groups": len(relation_components),
        "final_split_counts": manifest["final_split"].value_counts().sort_index().to_dict(),
        "final_split_counts_by_dataset": {
            dataset: group["final_split"].value_counts().sort_index().to_dict()
            for dataset, group in manifest.groupby("dataset")
        },
        "records_moved_to_more_conservative_holdout": len(changed),
        "moves": Counter(
            f"{source}->{final}"
            for source, final in zip(changed["source_split_normalized"], changed["final_split"])
        ),
        "confirmed_duplicate_groups": int(
            manifest.loc[manifest["duplicate_group_id"].ne(""), "duplicate_group_id"].nunique()
        ),
        "conservative_idrid_related_edges": len(related),
        "guarantee": (
            "Confirmed duplicate photographs and conservatively linked IDRiD records do not cross splits."
        ),
        "limitation": (
            "Patient independence cannot be guaranteed because patient identifiers are unavailable."
        ),
        "policy": "When a group crossed source splits, the entire group was moved to test, then validation, then train in that priority order.",
    }
    summary["moves"] = dict(sorted(summary["moves"].items()))
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    manifest, summary = assign(workspace)
    output_dir = workspace / "data" / "manifests"
    manifest.to_csv(output_dir / "trace_analysis_manifest.csv", index=False)
    (output_dir / "trace_split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
