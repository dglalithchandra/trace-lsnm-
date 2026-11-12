"""Label-hidden, checkpointable MoCo pilot on RFMiD-v1 training images only."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight_trace_ssl_views import view_full_field, view_plain
from trace_models.dinov2 import load_dinov2_vits14
from trace_preprocessing.adapters import prepare_dinov2_tensor

PROTOCOL = ROOT / "data/protocols/trace_ssl_v1"
INDEX = PROTOCOL / "cached_image_index.csv"
SEED = 20261111
RANK = 8
ALPHA = 8.0
EMBEDDING = 384
PROJECTION = 128
QUEUE_SIZE = 4096
TEMPERATURE = 0.2
MOMENTUM = 0.996


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = RANK, alpha: float = ALPHA) -> None:
        super().__init__()
        self.base = linear
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_b = nn.Parameter(torch.zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale


def add_lora(model: nn.Module) -> int:
    for block in model.blocks[-4:]:
        block.attn.qkv = LoRALinear(block.attn.qkv)
        block.attn.proj = LoRALinear(block.attn.proj)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if not 0 < trainable < 1_000_000:
        raise RuntimeError(f"Unexpected LoRA parameter count {trainable}")
    return trainable


def projector() -> nn.Module:
    return nn.Sequential(nn.Linear(EMBEDDING, EMBEDDING), nn.GELU(), nn.Linear(EMBEDDING, PROJECTION))


def read_index() -> list[dict[str, str]]:
    with INDEX.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1917 or set(rows[0]) != {"record_id", "cache_path", "source_image_sha256", "cache_image_sha256", "preprocessing_status"}:
        raise RuntimeError("Frozen SSL cache index changed")
    return rows


def views(rows: list[dict[str, str]], indices: list[int], epoch: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    first, second = [], []
    mask = np.ones((448, 448), dtype=np.uint8)
    transform = view_plain if mode in {"plain_crop", "plain_crop_matched"} else view_full_field
    for index in indices:
        bgr = cv2.imread(str(ROOT / rows[index]["cache_path"]), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (448, 448, 3):
            raise RuntimeError(f"Corrupt cached training image: {rows[index]['cache_path']}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        a_rng = np.random.default_rng(SEED + epoch * 10000019 + index * 20011)
        b_rng = np.random.default_rng(SEED + epoch * 10000019 + index * 20011 + 503)
        a, _ = transform(image, mask, a_rng)
        b, _ = transform(image, mask, b_rng)
        first.append(prepare_dinov2_tensor(a))
        second.append(prepare_dinov2_tensor(b))
    return torch.stack(first), torch.stack(second)


def ema_update(student: nn.Module, teacher: nn.Module, student_head: nn.Module, teacher_head: nn.Module) -> None:
    student_parameters = dict(student.named_parameters())
    teacher_parameters = dict(teacher.named_parameters())
    with torch.no_grad():
        for name, parameter in student_parameters.items():
            if parameter.requires_grad:
                teacher_parameters[name].lerp_(parameter, 1 - MOMENTUM)
        for current, target in zip(student_head.parameters(), teacher_head.parameters(), strict=True):
            target.lerp_(current, 1 - MOMENTUM)


def save_checkpoint(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plain_crop", "plain_crop_matched", "full_field"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke-test stop after this many global steps; 0 runs to epoch target")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-dir", default="", help="Checkpoint directory under models/; default is the original trace_ssl_v1 mode directory")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0 or args.max_steps < 0:
        raise ValueError("Invalid training configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("TRACE SSL training requires a CUDA GPU in this implementation")
    rows = read_index()
    index_hash = hashlib.sha256(INDEX.read_bytes()).hexdigest()
    seed = SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    student = load_dinov2_vits14(ROOT, device=device)
    trainable_backbone = add_lora(student)
    student.to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    student_head = projector().to(device)
    teacher_head = copy.deepcopy(student_head).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW([parameter for parameter in student.parameters() if parameter.requires_grad] + list(student_head.parameters()), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    queue = torch.zeros((QUEUE_SIZE, PROJECTION), device=device, dtype=torch.float32)
    queue_ids = torch.full((QUEUE_SIZE,), -1, device=device, dtype=torch.long)
    pointer = 0
    count = 0
    epoch = 0
    cursor = 0
    step = 0
    run_dir = (ROOT / args.run_dir if args.run_dir else ROOT / "models/trace_ssl_v1" / args.mode).resolve()
    if not run_dir.is_relative_to((ROOT / "models").resolve()):
        raise RuntimeError("Training run directory must be under models/")
    checkpoint = run_dir / "last.pt"
    config = {"mode": args.mode, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "seed": seed, "cache_index_sha256": index_hash, "rank": RANK, "alpha": ALPHA, "queue_size": QUEUE_SIZE, "temperature": TEMPERATURE, "momentum": MOMENTUM, "backbone_trainable_parameters": trainable_backbone}
    if args.resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint missing: {checkpoint}")
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["config"] != config:
            raise RuntimeError("Checkpoint config differs from current training config")
        student.load_state_dict(saved["student"])
        teacher.load_state_dict(saved["teacher"])
        student_head.load_state_dict(saved["student_head"])
        teacher_head.load_state_dict(saved["teacher_head"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        queue.copy_(saved["queue"])
        queue_ids.copy_(saved["queue_ids"])
        pointer, count, epoch, cursor, step = (int(saved[name]) for name in ("pointer", "count", "epoch", "cursor", "step"))
        torch.set_rng_state(saved["torch_rng_cpu"].cpu())
        torch.cuda.set_rng_state(saved["torch_rng_cuda"].cpu())
    elif checkpoint.exists():
        raise RuntimeError(f"Checkpoint already exists; pass --resume or use another mode: {checkpoint}")

    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training_history.csv"
    if not args.resume:
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(("epoch", "step", "batch_size", "loss", "positive_cosine", "negative_cosine", "embedding_std", "queue_count", "wall_seconds"))

    def checkpoint_state() -> dict[str, object]:
        return {"config": config, "student": student.state_dict(), "teacher": teacher.state_dict(), "student_head": student_head.state_dict(), "teacher_head": teacher_head.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "queue": queue, "queue_ids": queue_ids, "pointer": pointer, "count": count, "epoch": epoch, "cursor": cursor, "step": step, "torch_rng_cpu": torch.get_rng_state(), "torch_rng_cuda": torch.cuda.get_rng_state()}

    student.train()
    student_head.train()
    start = time.monotonic()
    while epoch < args.epochs:
        order = np.random.default_rng(SEED + epoch).permutation(len(rows)).tolist()
        if cursor > len(order):
            raise RuntimeError("Checkpoint cursor exceeds epoch length")
        while cursor < len(order):
            indices = order[cursor:cursor + args.batch_size]
            if len(indices) < 2:
                cursor = len(order)
                break
            query_cpu, key_cpu = views(rows, indices, epoch, args.mode)
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
                    repeat = torch.as_tensor(indices, device=device).unsqueeze(1) == queue_ids[:count].unsqueeze(0)
                    logits_queue = logits_queue.masked_fill(repeat, -1e4)
                    logits = torch.cat((logits_batch, logits_queue), dim=1)
                else:
                    logits = logits_batch
                loss = F.cross_entropy(logits, torch.arange(len(indices), device=device))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite SSL loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad] + list(student_head.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema_update(student, teacher, student_head, teacher_head)
            with torch.no_grad():
                for position, image_id in enumerate(indices):
                    queue[pointer] = key_embedding[position]
                    queue_ids[pointer] = image_id
                    pointer = (pointer + 1) % QUEUE_SIZE
                    count = min(QUEUE_SIZE, count + 1)
                positive_cosine = float((query_embedding * key_embedding).sum(dim=1).mean())
                off_diagonal = (query_embedding @ key_embedding.T)[~torch.eye(len(indices), dtype=torch.bool, device=device)]
                negative_cosine = float(off_diagonal.mean())
                embedding_std = float(query_embedding.std(dim=0).mean())
            step += 1
            cursor += len(indices)
            if step % 10 == 0 or step == 1:
                with history_path.open("a", newline="", encoding="utf-8") as stream:
                    csv.writer(stream).writerow((epoch, step, len(indices), float(loss.detach()), positive_cosine, negative_cosine, embedding_std, count, round(time.monotonic() - start, 2)))
                print(f"{args.mode} epoch {epoch + 1}/{args.epochs} step {step} loss {float(loss.detach()):.4f} std {embedding_std:.4f}", flush=True)
            if step % 100 == 0 or (args.max_steps and step >= args.max_steps):
                save_checkpoint(checkpoint, checkpoint_state())
            if args.max_steps and step >= args.max_steps:
                print(f"Stopped at requested smoke-test step {step}; resume with --resume", flush=True)
                return
        epoch += 1
        cursor = 0
        save_checkpoint(checkpoint, checkpoint_state())
        print(f"{args.mode} completed epoch {epoch}/{args.epochs}", flush=True)
    summary = {"status": "TRAINING_COMPLETE_NOT_EVALUATED", "config": config, "epochs": epoch, "steps": step, "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "protected_test_images_opened": 0}
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
