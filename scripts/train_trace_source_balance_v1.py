"""Fixed-update TRACE SSL trainer for natural versus source-balanced sampling."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight_trace_ssl_views import view_plain
from scripts.train_trace_ssl_v1 import (
    MOMENTUM,
    PROJECTION,
    QUEUE_SIZE,
    TEMPERATURE,
    add_lora,
    ema_update,
    projector,
    save_checkpoint,
)
from trace_models.dinov2 import load_dinov2_vits14
from trace_preprocessing.adapters import prepare_dinov2_tensor


REQUIRED_FIELDS = {
    "record_id",
    "source_id",
    "cache_path",
    "source_image_sha256",
    "cache_image_sha256",
    "preprocessing_status",
    "admission_status",
    "final_split",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pool(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if not REQUIRED_FIELDS.issubset(fields):
            raise RuntimeError(f"Pool manifest missing fields: {sorted(REQUIRED_FIELDS - fields)}")
        rows = [row for row in reader if row["admission_status"] == "admitted"]
    if len(rows) < 2:
        raise RuntimeError("Pool must contain at least two admitted records")
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate record IDs in admitted pool")
    hashes = [row["source_image_sha256"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Duplicate source-image hashes in admitted pool")
    if any(row["final_split"] != "train" for row in rows):
        raise RuntimeError("Non-training split record admitted to SSL pool")
    if any(not row["source_id"] for row in rows):
        raise RuntimeError("Every admitted record requires a source_id")
    return sorted(rows, key=lambda row: row["record_id"])


def sample_batch_indices(
    rows: list[dict[str, str]],
    *,
    step: int,
    batch_size: int,
    sampling: str,
    seed: int,
) -> list[int]:
    """Return a deterministic batch; balanced mode equalises source slots."""
    if step < 0 or batch_size < 2:
        raise ValueError("Invalid step or batch size")
    rng = np.random.default_rng(seed + step * 1_000_003)
    if sampling == "natural":
        return rng.choice(len(rows), size=batch_size, replace=len(rows) < batch_size).tolist()
    if sampling != "source_balanced":
        raise ValueError(f"Unknown sampling mode: {sampling}")

    source_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        source_to_indices[row["source_id"]].append(index)
    sources = sorted(source_to_indices)
    assigned: dict[str, int] = Counter(
        sources[(step * batch_size + position) % len(sources)]
        for position in range(batch_size)
    )
    selected: list[int] = []
    for source in sources:
        count = assigned.get(source, 0)
        if not count:
            continue
        candidates = source_to_indices[source]
        selected.extend(
            rng.choice(candidates, size=count, replace=len(candidates) < count).tolist()
        )
    rng.shuffle(selected)
    return selected


def build_views(
    rows: list[dict[str, str]], indices: list[int], *, step: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    first: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    mask = np.ones((448, 448), dtype=np.uint8)
    for index in indices:
        row = rows[index]
        bgr = cv2.imread(str(ROOT / row["cache_path"]), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (448, 448, 3):
            raise RuntimeError(f"Corrupt cached training image: {row['cache_path']}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        view_seed = seed + step * 10_000_019 + index * 20_011
        a, _ = view_plain(image, mask, np.random.default_rng(view_seed))
        b, _ = view_plain(image, mask, np.random.default_rng(view_seed + 503))
        first.append(prepare_dinov2_tensor(a))
        second.append(prepare_dinov2_tensor(b))
    return torch.stack(first), torch.stack(second)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/trace_ssl_pool_v2.csv")
    parser.add_argument("--sampling", choices=("natural", "source_balanced"), required=True)
    parser.add_argument("--updates", type=int, default=9580)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--run-dir", required=True, help="Checkpoint directory under models/")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop at this total step for a smoke run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        raise ValueError("Invalid training configuration")
    if args.max_steps < 0:
        raise ValueError("max-steps cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("TRACE source-balance training requires CUDA")

    manifest = (ROOT / args.manifest).resolve()
    if not manifest.is_relative_to((ROOT / "data/manifests").resolve()):
        raise RuntimeError("Pool manifest must be under data/manifests/")
    rows = read_pool(manifest)
    sources = sorted({row["source_id"] for row in rows})
    if args.sampling == "source_balanced" and len(sources) < 2:
        raise RuntimeError("Source-balanced sampling requires at least two admitted sources")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    student = load_dinov2_vits14(ROOT, device=device)
    trainable_backbone = add_lora(student)
    student.to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    student_head = projector().to(device)
    teacher_head = copy.deepcopy(student_head).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad]
        + list(student_head.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda")
    queue = torch.zeros((QUEUE_SIZE, PROJECTION), device=device, dtype=torch.float32)
    queue_ids = torch.full((QUEUE_SIZE,), -1, device=device, dtype=torch.long)
    pointer = 0
    count = 0
    step = 0
    source_draw_counts: Counter[str] = Counter()

    run_dir = (ROOT / args.run_dir).resolve()
    if not run_dir.is_relative_to((ROOT / "models").resolve()):
        raise RuntimeError("Run directory must be under models/")
    checkpoint = run_dir / "last.pt"
    history_path = run_dir / "training_history.csv"
    summary_path = run_dir / "training_summary.json"
    config = {
        "version": "trace-source-balance-v1",
        "manifest": str(manifest.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256(manifest),
        "record_count": len(rows),
        "sources": sources,
        "sampling": args.sampling,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "view": "plain_crop_matched",
        "rank": 8,
        "alpha": 8.0,
        "queue_size": QUEUE_SIZE,
        "temperature": TEMPERATURE,
        "momentum": MOMENTUM,
        "backbone_trainable_parameters": trainable_backbone,
    }

    if args.resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint missing: {checkpoint}")
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["config"] != config:
            raise RuntimeError("Checkpoint config differs from current configuration")
        student.load_state_dict(saved["student"])
        teacher.load_state_dict(saved["teacher"])
        student_head.load_state_dict(saved["student_head"])
        teacher_head.load_state_dict(saved["teacher_head"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        queue.copy_(saved["queue"])
        queue_ids.copy_(saved["queue_ids"])
        pointer = int(saved["pointer"])
        count = int(saved["count"])
        step = int(saved["step"])
        source_draw_counts.update(saved["source_draw_counts"])
        torch.set_rng_state(saved["torch_rng_cpu"].cpu())
        torch.cuda.set_rng_state(saved["torch_rng_cuda"].cpu())
    elif checkpoint.exists():
        raise RuntimeError("Run checkpoint already exists; use --resume or a new run-dir")

    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream, lineterminator="\n").writerow(
                (
                    "step",
                    "batch_size",
                    "loss",
                    "positive_cosine",
                    "negative_cosine",
                    "embedding_std",
                    "queue_count",
                    "source_draw_counts_json",
                    "wall_seconds",
                )
            )

    def checkpoint_state() -> dict[str, object]:
        return {
            "config": config,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "student_head": student_head.state_dict(),
            "teacher_head": teacher_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "queue": queue,
            "queue_ids": queue_ids,
            "pointer": pointer,
            "count": count,
            "step": step,
            "source_draw_counts": dict(source_draw_counts),
            "torch_rng_cpu": torch.get_rng_state(),
            "torch_rng_cuda": torch.cuda.get_rng_state(),
        }

    stop_at = min(args.updates, args.max_steps) if args.max_steps else args.updates
    if step >= stop_at:
        raise RuntimeError(f"Checkpoint is already at step {step}, requested stop is {stop_at}")
    student.train()
    student_head.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    while step < stop_at:
        indices = sample_batch_indices(
            rows,
            step=step,
            batch_size=args.batch_size,
            sampling=args.sampling,
            seed=args.seed,
        )
        for index in indices:
            source_draw_counts[rows[index]["source_id"]] += 1
        query_cpu, key_cpu = build_views(rows, indices, step=step, seed=args.seed)
        query_input = query_cpu.to(device)
        key_input = key_cpu.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            query_embedding = F.normalize(student_head(student(query_input)).float(), dim=1)
            with torch.no_grad():
                key_embedding = F.normalize(teacher_head(teacher(key_input)).float(), dim=1)
            logits_batch = query_embedding @ key_embedding.T / TEMPERATURE
            if count:
                logits_queue = query_embedding @ queue[:count].T / TEMPERATURE
                repeated = (
                    torch.as_tensor(indices, device=device).unsqueeze(1)
                    == queue_ids[:count].unsqueeze(0)
                )
                logits_queue = logits_queue.masked_fill(repeated, -1e4)
                logits = torch.cat((logits_batch, logits_queue), dim=1)
            else:
                logits = logits_batch
            loss = F.cross_entropy(logits, torch.arange(len(indices), device=device))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite SSL loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in student.parameters() if parameter.requires_grad]
            + list(student_head.parameters()),
            1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        ema_update(student, teacher, student_head, teacher_head)
        with torch.no_grad():
            for position, image_index in enumerate(indices):
                queue[pointer] = key_embedding[position]
                queue_ids[pointer] = image_index
                pointer = (pointer + 1) % QUEUE_SIZE
                count = min(QUEUE_SIZE, count + 1)
            positive_cosine = float((query_embedding * key_embedding).sum(dim=1).mean())
            off_diagonal = (query_embedding @ key_embedding.T)[
                ~torch.eye(len(indices), dtype=torch.bool, device=device)
            ]
            negative_cosine = float(off_diagonal.mean())
            embedding_std = float(query_embedding.std(dim=0).mean())
        step += 1
        if step % 10 == 0 or step == 1:
            with history_path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream, lineterminator="\n").writerow(
                    (
                        step,
                        len(indices),
                        float(loss.detach()),
                        positive_cosine,
                        negative_cosine,
                        embedding_std,
                        count,
                        json.dumps(dict(sorted(source_draw_counts.items())), sort_keys=True),
                        round(time.monotonic() - started, 3),
                    )
                )
            print(
                f"{args.sampling} step {step}/{stop_at} loss {float(loss.detach()):.4f} "
                f"std {embedding_std:.4f}",
                flush=True,
            )
        if step % 100 == 0 or step == stop_at:
            save_checkpoint(checkpoint, checkpoint_state())

    elapsed = time.monotonic() - started
    summary = {
        "status": "TRAINING_COMPLETE" if step == args.updates else "SMOKE_STOP_COMPLETE",
        "config": config,
        "step": step,
        "process_training_seconds": round(elapsed, 3),
        "steps_per_second_this_process": round((stop_at - (0 if not args.resume else int(saved["step"]))) / elapsed, 5),
        "source_draw_counts": dict(sorted(source_draw_counts.items())),
        "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256(checkpoint),
        "protected_test_images_opened": 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
