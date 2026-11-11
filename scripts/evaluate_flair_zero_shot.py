"""Base-tuned prompt-style selection and novel-label FLAIR zero-shot test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flair import FLAIRModel
from sklearn.metrics import average_precision_score


WORKSPACE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE / "outputs" / "flair_zero_shot_v1"
FLAIR_REVISION = "5f6bdd0a068353dc41a896ba3abdd7c0f6d35938"
BOOTSTRAP_REPLICATES = 2000
SEED = 20261111


def compute_text_embeddings(
    model: FLAIRModel, categories: list[str], domain_knowledge: bool
) -> np.ndarray:
    _, embeddings = model.compute_text_embeddings(
        categories, domain_knowledge=domain_knowledge
    )
    embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
    return embeddings.detach().cpu().numpy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol_dir = WORKSPACE / "data" / "protocols" / "trace_fewshot_v1"
    protocol = json.loads((protocol_dir / "protocol.json").read_text(encoding="utf-8"))
    prompt_map = json.loads(
        (WORKSPACE / "data" / "labels" / "rfmid_flair_prompt_map.json").read_text(
            encoding="utf-8"
        )
    )
    base_labels = protocol["base_labels"]
    novel_labels = protocol["novel_labels"]
    all_labels = base_labels + novel_labels
    categories = [prompt_map["labels"][label]["flair_category"] for label in all_labels]

    feature_path = WORKSPACE / "data" / "features" / "rfmid_validation_flair_geometry_v1.npz"
    with np.load(feature_path, allow_pickle=False) as payload:
        image_embeddings = payload["embeddings"].astype(np.float32)
        labels = payload["labels"].astype(np.uint8)
        class_order = payload["class_order"].astype(str).tolist()
        split = str(payload["split"][0])
    if split != "validation":
        raise RuntimeError("Zero-shot evaluation must use the frozen validation query set")
    image_embeddings /= np.maximum(
        np.linalg.norm(image_embeddings, axis=1, keepdims=True), 1e-12
    )
    class_index = {label: index for index, label in enumerate(class_order)}

    model = FLAIRModel.from_pretrained(
        "jusiro2/FLAIR",
        revision=FLAIR_REVISION,
        cache_dir=WORKSPACE / "models" / "huggingface",
        local_files_only=True,
        vision_pretrained=False,
    ).eval().requires_grad_(False)
    name_embeddings = compute_text_embeddings(model, categories + ["normal"], False)
    knowledge_embeddings = compute_text_embeddings(model, categories + ["normal"], True)

    variants: dict[str, np.ndarray] = {}
    for prefix, text_embeddings in (
        ("name", name_embeddings),
        ("domain_knowledge", knowledge_embeddings),
    ):
        positive_scores = image_embeddings @ text_embeddings[:-1].T
        normal_scores = image_embeddings @ text_embeddings[-1]
        variants[f"{prefix}_positive_only"] = positive_scores
        variants[f"{prefix}_minus_normal"] = positive_scores - normal_scores[:, None]

    rows: list[dict[str, object]] = []
    for variant, score_matrix in variants.items():
        for prompt_index, label in enumerate(all_labels):
            y = labels[:, class_index[label]]
            prevalence = float(y.mean())
            auprc = float(average_precision_score(y, score_matrix[:, prompt_index]))
            rows.append(
                {
                    "role": "base" if label in base_labels else "novel",
                    "label": label,
                    "rfmid_name": prompt_map["labels"][label]["rfmid_name"],
                    "flair_category": prompt_map["labels"][label]["flair_category"],
                    "variant": variant,
                    "query_prevalence": prevalence,
                    "auprc": auprc,
                    "auprc_lift_over_prevalence": auprc - prevalence,
                }
            )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "zero_shot_per_label.csv", index=False)
    base_macro = (
        results[results["role"].eq("base")]
        .groupby("variant", as_index=False)[
            ["query_prevalence", "auprc", "auprc_lift_over_prevalence"]
        ]
        .mean()
        .sort_values(["auprc_lift_over_prevalence", "variant"], ascending=[False, True])
    )
    base_macro.to_csv(OUTPUT_DIR / "base_prompt_selection.csv", index=False)
    selected_variant = str(base_macro.iloc[0]["variant"])
    novel_selected = results[
        results["role"].eq("novel") & results["variant"].eq(selected_variant)
    ].copy()
    novel_selected.to_csv(OUTPUT_DIR / "novel_selected_prompt_results.csv", index=False)

    selected_scores = variants[selected_variant]
    np.savez_compressed(
        OUTPUT_DIR / "selected_zero_shot_scores.npz",
        labels=np.asarray(all_labels, dtype=np.str_),
        scores=selected_scores.astype(np.float32),
        validation_labels=labels,
        class_order=np.asarray(class_order, dtype=np.str_),
        selected_variant=np.asarray([selected_variant], dtype=np.str_),
    )
    novel_prompt_index = {label: all_labels.index(label) for label in novel_labels}
    rng = np.random.default_rng(SEED)
    distribution = np.empty(BOOTSTRAP_REPLICATES)
    for bootstrap_index in range(BOOTSTRAP_REPLICATES):
        sampled_labels = rng.choice(novel_labels, size=len(novel_labels), replace=True)
        values: list[float] = []
        for label in sampled_labels:
            y = labels[:, class_index[label]]
            positive = np.flatnonzero(y == 1)
            negative = np.flatnonzero(y == 0)
            query = np.concatenate(
                [
                    rng.choice(positive, size=len(positive), replace=True),
                    rng.choice(negative, size=len(negative), replace=True),
                ]
            )
            values.append(
                average_precision_score(
                    y[query], selected_scores[query, novel_prompt_index[label]]
                )
            )
        distribution[bootstrap_index] = np.mean(values)

    summary = {
        "status": "PASS",
        "selected_prompt_variant_from_base_labels": selected_variant,
        "base_labels": base_labels,
        "novel_labels": novel_labels,
        "novel_macro_query_prevalence": float(novel_selected["query_prevalence"].mean()),
        "novel_macro_auprc": float(novel_selected["auprc"].mean()),
        "novel_macro_auprc_lift_over_prevalence": float(
            novel_selected["auprc_lift_over_prevalence"].mean()
        ),
        "bootstrap_macro_auprc_mean": float(distribution.mean()),
        "bootstrap_ci95_lower": float(np.quantile(distribution, 0.025)),
        "bootstrap_ci95_upper": float(np.quantile(distribution, 0.975)),
        "contamination_tag": "possibly_seen_rfmid_listed_in_pretraining_assembly",
        "interpretation": (
            "Zero-shot means no RFMiD support image was used for the novel label. "
            "Prompt style was selected on base labels only. Because FLAIR lists "
            "RFMiD in its pretraining assembly, this is not a clean unseen-dataset test."
        ),
        "test_records_accessed": 0,
    }
    (OUTPUT_DIR / "zero_shot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nBase prompt selection:\n" + base_macro.to_string(index=False))
    print("\nNovel selected prompt results:\n" + novel_selected.to_string(index=False))


if __name__ == "__main__":
    main()

