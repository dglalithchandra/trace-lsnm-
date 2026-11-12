import copy
import json
from pathlib import Path

import pytest

from trace_explanations import (
    ExplanationGraphError,
    load_graph,
    lookup_explanations,
    validate_graph,
)


ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = (
    ROOT / "data" / "knowledge_sources" / "curated" / "trace_explanation_graph_v1.json"
)


def test_checked_in_graph_is_conservative_and_has_a_20_label_queue():
    graph = load_graph(GRAPH_PATH)
    assert graph["mappings"] == []
    assert graph["knowledge_edges"] == []
    assert 20 <= len(graph["curation_queue"]) <= 40
    assert all(source["available_in_clone"] is False for source in graph["sources"])
    assert graph["clinical_verification_status"] == "not_clinician_verified"


def synthetic_mapped_graph():
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    graph["sources"][0]["available_in_clone"] = True
    graph["nodes"].extend(
        [
            {
                "node_id": "MONDO:synthetic",
                "label": "Synthetic disease",
                "entity_type": "disease",
                "definition": "A synthetic test definition.",
                "definition_provenance": {
                    "source_id": "MONDO_RARE",
                    "source_version": "2026-11-03",
                    "locator": "synthetic:test",
                    "licence": "CC-BY-4.0",
                    "review_state": "imported_unreviewed",
                },
            },
            {
                "node_id": "TRACE_LABEL:SYN",
                "label": "SYN",
                "entity_type": "disease",
                "definition": None,
            },
        ]
    )
    graph["mappings"].append(
        {
            "source_node_id": "TRACE_LABEL:SYN",
            "target_node_id": "MONDO:synthetic",
            "predicate": "exact",
            "justification": "Synthetic validator fixture only.",
            "provenance": {
                "source_id": "MONDO_RARE",
                "source_version": "2026-11-03",
                "locator": "synthetic:test",
                "licence": "CC-BY-4.0",
                "review_state": "imported_unreviewed",
            },
        }
    )
    return graph


@pytest.mark.parametrize("predicate", ["exact", "broad", "narrow", "related"])
def test_mapping_predicates_remain_distinct(predicate):
    graph = synthetic_mapped_graph()
    graph["mappings"][0]["predicate"] = predicate
    validate_graph(graph)
    result = lookup_explanations(graph, [{"label_id": "SYN", "score": 0.7}])
    assert result["explanations"][0]["mappings"][0]["predicate"] == predicate


def test_graph_requires_complete_provenance():
    graph = synthetic_mapped_graph()
    del graph["mappings"][0]["provenance"]["locator"]
    with pytest.raises(ExplanationGraphError, match="missing provenance"):
        validate_graph(graph)


def test_disease_finding_exact_equivalence_is_rejected():
    graph = synthetic_mapped_graph()
    graph["nodes"].append(
        {
            "node_id": "HP:synthetic",
            "label": "Synthetic finding",
            "entity_type": "finding",
            "definition": None,
        }
    )
    graph["mappings"][0]["target_node_id"] = "HP:synthetic"
    with pytest.raises(ExplanationGraphError, match="cannot be exact mappings"):
        validate_graph(graph)


def test_lookup_preserves_exact_score_values_and_input_order():
    graph = load_graph(GRAPH_PATH)
    outputs = [
        {"label_id": "RT", "score": 0.42},
        {"label_id": "MS", "score": -1.25},
        {"label_id": "UNKNOWN", "score": 9},
    ]
    result = lookup_explanations(
        graph,
        outputs,
        qc_flags=[{"name": "blur", "value": 0.2, "source": "measured"}],
        provenance={"data_scope": "synthetic", "model_id": "synthetic"},
    )
    assert result["original_model_outputs"] == outputs
    assert [item["label_id"] for item in result["original_model_outputs"]] == [
        "RT",
        "MS",
        "UNKNOWN",
    ]
    assert [item["score"] for item in result["original_model_outputs"]] == [0.42, -1.25, 9]
    assert len(result["missing_knowledge"]) == 3
    assert all(
        item["case_evidence_status"] == "unknown_no_image_finding_inference_from_graph"
        for item in result["explanations"]
    )
    assert "confidence" not in result
    assert "diagnosis" not in result


def test_lookup_rejects_protected_data_scope():
    graph = load_graph(GRAPH_PATH)
    with pytest.raises(ExplanationGraphError, match="protected"):
        lookup_explanations(
            graph,
            [{"label_id": "RT", "score": 0.1}],
            provenance={"data_scope": "protected_final_test"},
        )
