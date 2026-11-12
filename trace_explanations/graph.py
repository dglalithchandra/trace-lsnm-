"""Validated, deterministic explanation lookup that never changes model scores."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class ExplanationGraphError(ValueError):
    """Raised for unsafe graph records or explanation requests."""


_MAPPING_PREDICATES = {"exact", "broad", "narrow", "related"}
_REVIEW_STATES = {"imported_unreviewed", "manually_reviewed", "pending"}
_ENTITY_TYPES = {"disease", "finding", "quality", "pending_definition"}


def load_graph(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    validate_graph(graph)
    return graph


def _object(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExplanationGraphError(f"{location} must be an object")
    return value


def _array(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExplanationGraphError(f"{location} must be an array")
    return value


def _provenance(value: Any, location: str) -> Mapping[str, Any]:
    provenance = _object(value, location)
    required = ("source_id", "source_version", "locator", "licence", "review_state")
    missing = [name for name in required if not provenance.get(name)]
    if missing:
        raise ExplanationGraphError(f"{location} missing provenance fields: {missing}")
    if provenance["review_state"] not in _REVIEW_STATES:
        raise ExplanationGraphError(f"{location}.review_state is invalid")
    return provenance


def validate_graph(graph: Mapping[str, Any]) -> None:
    """Validate source, mapping and assertion provenance without source inference."""

    root = _object(graph, "graph")
    if root.get("schema_version") != "trace_explanation_graph_v1":
        raise ExplanationGraphError("unsupported explanation graph schema")
    if root.get("purpose") != "explanation_only_preserve_model_ranking":
        raise ExplanationGraphError("graph purpose must be explanation-only")

    source_ids: set[str] = set()
    for index, raw_source in enumerate(_array(root.get("sources"), "sources")):
        source = _object(raw_source, f"sources[{index}]")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ExplanationGraphError(f"sources[{index}].source_id is required")
        if source_id in source_ids:
            raise ExplanationGraphError("source IDs must be unique")
        source_ids.add(source_id)
        expected_sha = source.get("expected_sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ExplanationGraphError(f"sources[{index}].expected_sha256 is invalid")

    nodes: dict[str, Mapping[str, Any]] = {}
    for index, raw_node in enumerate(_array(root.get("nodes"), "nodes")):
        node = _object(raw_node, f"nodes[{index}]")
        node_id = node.get("node_id")
        if not isinstance(node_id, str) or not node_id or node_id in nodes:
            raise ExplanationGraphError("node IDs must be non-empty and unique")
        if node.get("entity_type") not in _ENTITY_TYPES:
            raise ExplanationGraphError(f"nodes[{index}].entity_type is invalid")
        definition = node.get("definition")
        if definition is not None:
            _provenance(node.get("definition_provenance"), f"nodes[{index}].definition_provenance")
        nodes[node_id] = node

    for index, raw_mapping in enumerate(_array(root.get("mappings"), "mappings")):
        mapping = _object(raw_mapping, f"mappings[{index}]")
        predicate = mapping.get("predicate")
        if predicate not in _MAPPING_PREDICATES:
            raise ExplanationGraphError(f"mappings[{index}].predicate is invalid")
        source_node = nodes.get(str(mapping.get("source_node_id")))
        target_node = nodes.get(str(mapping.get("target_node_id")))
        if source_node is None or target_node is None:
            raise ExplanationGraphError(f"mappings[{index}] references an unknown node")
        provenance = _provenance(mapping.get("provenance"), f"mappings[{index}].provenance")
        if provenance["source_id"] not in source_ids:
            raise ExplanationGraphError(f"mappings[{index}] references an unknown source")
        if (
            predicate == "exact"
            and source_node.get("entity_type") in {"disease", "finding"}
            and target_node.get("entity_type") in {"disease", "finding"}
            and source_node.get("entity_type") != target_node.get("entity_type")
        ):
            raise ExplanationGraphError("disease and finding nodes cannot be exact mappings")
        target_id = str(mapping.get("target_node_id"))
        if target_node.get("entity_type") == "disease" and not target_id.startswith(
            ("MONDO:", "ORPHA:")
        ):
            raise ExplanationGraphError("disease mappings must target MONDO or ORPHA")
        if target_node.get("entity_type") == "finding" and not target_id.startswith("HP:"):
            raise ExplanationGraphError("finding mappings must target HPO")

    for index, raw_edge in enumerate(_array(root.get("knowledge_edges"), "knowledge_edges")):
        edge = _object(raw_edge, f"knowledge_edges[{index}]")
        if edge.get("subject_id") not in nodes or edge.get("object_id") not in nodes:
            raise ExplanationGraphError(f"knowledge_edges[{index}] references an unknown node")
        provenance = _provenance(edge.get("provenance"), f"knowledge_edges[{index}].provenance")
        if provenance["source_id"] not in source_ids:
            raise ExplanationGraphError(f"knowledge_edges[{index}] references an unknown source")

    queue = _array(root.get("curation_queue"), "curation_queue")
    queue_labels = [
        _object(item, f"curation_queue[{index}]").get("label_id")
        for index, item in enumerate(queue)
    ]
    if len(queue_labels) != len(set(queue_labels)):
        raise ExplanationGraphError("curation queue label IDs must be unique")


def _validate_outputs(model_outputs: Sequence[Mapping[str, Any]]) -> None:
    label_ids: list[str] = []
    for index, raw_output in enumerate(model_outputs):
        output = _object(raw_output, f"model_outputs[{index}]")
        label_id = output.get("label_id")
        score = output.get("score")
        if not isinstance(label_id, str) or not label_id:
            raise ExplanationGraphError(f"model_outputs[{index}].label_id is required")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ExplanationGraphError(f"model_outputs[{index}].score must be numeric")
        if not math.isfinite(float(score)):
            raise ExplanationGraphError(f"model_outputs[{index}].score must be finite")
        label_ids.append(label_id)
    if len(label_ids) != len(set(label_ids)):
        raise ExplanationGraphError("model output label IDs must be unique")


def lookup_explanations(
    graph: Mapping[str, Any],
    model_outputs: Sequence[Mapping[str, Any]],
    *,
    qc_flags: Sequence[Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach cited knowledge while preserving supplied output order and scores."""

    validate_graph(graph)
    _validate_outputs(model_outputs)
    if provenance and provenance.get("data_scope") not in {None, "development", "synthetic"}:
        raise ExplanationGraphError("protected or non-development model outputs are not permitted")
    original_outputs = copy.deepcopy(list(model_outputs))
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    mappings_by_source: dict[str, list[Mapping[str, Any]]] = {}
    for mapping in graph["mappings"]:
        mappings_by_source.setdefault(mapping["source_node_id"], []).append(mapping)

    explanations = []
    missing = []
    for output in original_outputs:
        trace_node_id = f"TRACE_LABEL:{output['label_id']}"
        node = nodes.get(trace_node_id)
        mappings = mappings_by_source.get(trace_node_id, [])
        if node is None or (node.get("definition") is None and not mappings):
            missing.append(
                {
                    "label_id": output["label_id"],
                    "reason": "no_source_backed_definition_or_mapping_available",
                }
            )
        explanations.append(
            {
                "label_id": output["label_id"],
                "definition": node.get("definition") if node else None,
                "definition_provenance": (
                    copy.deepcopy(node.get("definition_provenance")) if node else None
                ),
                "mappings": copy.deepcopy(mappings),
                "general_knowledge_edges": copy.deepcopy(
                    [
                        edge
                        for edge in graph["knowledge_edges"]
                        if edge["subject_id"] in {trace_node_id, *(item["target_node_id"] for item in mappings)}
                    ]
                ),
                "case_evidence_status": "unknown_no_image_finding_inference_from_graph",
            }
        )

    measured_qc = copy.deepcopy(list(qc_flags or []))
    return {
        "schema_version": "trace_explanation_lookup_v1",
        "purpose": "explanation_only",
        "original_model_outputs": original_outputs,
        "original_ranking_and_scores_preserved": True,
        "explanations": explanations,
        "missing_knowledge": missing,
        "measured_qc_flags": measured_qc,
        "input_provenance": copy.deepcopy(provenance),
        "technical_quality_limitations": [
            "QC values describe measured technical properties only.",
            "The graph does not infer image findings, diagnosis, confidence, causality or abstention.",
            "General knowledge does not establish visibility in the supplied photograph.",
        ],
    }
