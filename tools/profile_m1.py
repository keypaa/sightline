#!/usr/bin/env python3
"""Profile where a training step spends its time.

Answers one question with numbers: of the seconds in a step, how much is the
frozen reader, how much is the vocabulary projection, and how much is everything
else? Also reports how per-example cost scales with micro-batch, which is what
decides whether a bigger batch is worth it.

    PYTHONPATH=src python tools/profile_m1.py --n-examples 32 --micro-batch 4

The A10G profile is structural, not predictive: it shows the shape of the cost.
Absolute step times must be re-measured on the card that will do the training.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

from sightline.backbone import splice
from sightline.data.m1_synthetic import generate_m1_dataset
from sightline.objective import IGNORE_INDEX, answer_only_loss
from sightline.projector import Projector
from sightline.train import Batch, Trainer
from sightline.vision_k3_encoder import K3VisionEncoder

QWEN3_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
IMAGE_PAD = 151655
K3_WEIGHTS = "artifacts/k3_vision_f831ab66.safetensors"


def kernel_rows(prof, steps: int) -> list[list]:
    """Top kernels by CUDA time, robust to the torch 2.13 attribute rename."""

    rows = []
    for event in prof.key_averages():
        total = float(
            getattr(event, "device_time_total", 0.0) or getattr(event, "cuda_time_total", 0.0)
        )
        if total > 0:
            rows.append([round(total / steps / 1e3, 3), event.key])
    rows.sort(reverse=True)
    return rows


def run_profile(n_examples: int = 32, micro_batch: int = 4, profile_steps: int = 5) -> dict:
    device = "cuda"
    out: dict = {"gpu": torch.cuda.get_device_name(0), "n_examples": n_examples}

    t = time.perf_counter()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=QWEN3_REVISION)
    out["setup_tokenizer_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    encoder = K3VisionEncoder(
        weights=K3_WEIGHTS,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    out["setup_encoder_s"] = round(time.perf_counter() - t, 2)

    examples = generate_m1_dataset(seed=0).train[:n_examples]

    t = time.perf_counter()
    features = torch.cat([encoder.encode([e.image]) for e in examples], dim=0)
    feature_s = time.perf_counter() - t
    out["features_total_s"] = round(feature_s, 2)
    out["features_per_image_ms"] = round(1000 * feature_s / n_examples, 1)

    def build(attn: str, host_answer_len: bool):
        t = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B",
            revision=QWEN3_REVISION,
            dtype=torch.bfloat16,
            attn_implementation=attn,
        ).to(device)
        model.eval()
        load_s = time.perf_counter() - t

        t = time.perf_counter()
        singles = []
        for example in examples:
            text = tok.apply_chat_template(
                [{"role": "user", "content": "<|image_pad|>" * 256 + example.question}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            ids = tok(text, return_tensors="pt")["input_ids"].to(device)
            answer = tok(
                example.answer + tok.eos_token, add_special_tokens=False, return_tensors="pt"
            )["input_ids"].to(device)
            train_ids = torch.cat([ids, answer], dim=1)
            labels = torch.full_like(train_ids, IGNORE_INDEX)
            labels[:, ids.shape[1] :] = answer
            singles.append(Batch(train_ids, None, labels))
        build_s = time.perf_counter() - t

        batches = []
        for i in range(0, len(singles), micro_batch):
            part = singles[i : i + micro_batch]
            ids = torch.cat([b.input_ids for b in part], dim=0)
            labels = torch.cat([b.labels for b in part], dim=0)
            batches.append(
                Batch(
                    ids,
                    features[i : i + micro_batch].to(device),
                    labels,
                    n_supervised=(
                        int((labels != IGNORE_INDEX).sum(dim=1).max()) if host_answer_len else None
                    ),
                )
            )
        return model, batches, singles, round(load_s, 2), round(build_s, 2)

    def make_trainer(model, batches, trim: bool) -> Trainer:
        torch.manual_seed(123)
        return Trainer(
            model,
            Projector(input_width=4096, reader_width=2048),
            batches,
            micro_batches_per_step=max(1, 128 // micro_batch // 4),
            base_lr=3e-4,
            warmup_steps=50,
            max_steps=64,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            grad_clip=1.0,
            placeholder_id=IMAGE_PAD,
            device=device,
            dtype=torch.bfloat16,
            cycle=True,
            trim_logits=trim,
        )

    def phases(model, trainer, n_micro: int, n_steps: int) -> dict:
        """Attribution with explicit syncs, over a replica of one step."""

        acc = {
            "projector": 0.0,
            "splice": 0.0,
            "reader_forward": 0.0,
            "loss": 0.0,
            "backward": 0.0,
            "clip_and_step": 0.0,
        }
        embed = model.get_input_embeddings()
        params = list(trainer.projector.parameters())
        for _ in range(n_steps):
            trainer.optimizer.zero_grad()
            start = (trainer.step * n_micro) % len(trainer.batches)
            idxs = [(start + i) % len(trainer.batches) for i in range(n_micro)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for idx in idxs:
                    batch = trainer.batches[idx]
                    ids = batch.input_ids.to(device)
                    feats = batch.features.to(device)
                    labels = batch.labels.to(device)

                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    visual = trainer.projector(feats)
                    torch.cuda.synchronize()
                    acc["projector"] += time.perf_counter() - t

                    t = time.perf_counter()
                    spliced = splice(ids, visual, IMAGE_PAD, embed.weight)
                    torch.cuda.synchronize()
                    acc["splice"] += time.perf_counter() - t

                    t = time.perf_counter()
                    logits = trainer._logits(
                        spliced, batch.attention_mask, labels, batch.n_supervised
                    )
                    torch.cuda.synchronize()
                    acc["reader_forward"] += time.perf_counter() - t

                    t = time.perf_counter()
                    loss = answer_only_loss(
                        logits[:, :-1], labels[:, labels.shape[1] - (logits.shape[1] - 1) :]
                    )
                    torch.cuda.synchronize()
                    acc["loss"] += time.perf_counter() - t

                    t = time.perf_counter()
                    (loss / n_micro).backward()
                    torch.cuda.synchronize()
                    acc["backward"] += time.perf_counter() - t

            t = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(params, trainer.grad_clip)
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.step += 1
            torch.cuda.synchronize()
            acc["clip_and_step"] += time.perf_counter() - t
        return {k: round(v / n_steps, 4) for k, v in acc.items()}

    def batch_scaling(model, singles, sizes=(1, 2, 4)) -> list:
        """Reader forward+backward cost against micro-batch size."""

        torch.manual_seed(123)
        projector = Projector(input_width=4096, reader_width=2048).to(device)
        embed = model.get_input_embeddings()
        rows = []
        for m in sizes:
            ids = torch.cat([singles[i].input_ids for i in range(m)], dim=0)
            feats = features[:m].to(device)
            labels = torch.cat([singles[i].labels for i in range(m)], dim=0)
            supervised = int((labels != IGNORE_INDEX).sum(dim=1).max())

            def one(ids=ids, feats=feats, labels=labels, supervised=supervised) -> None:
                projector.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    visual = projector(feats)
                    spliced = splice(ids, visual, IMAGE_PAD, embed.weight)
                    logits = model(
                        inputs_embeds=spliced, logits_to_keep=min(supervised + 1, labels.shape[1])
                    ).logits
                    loss = answer_only_loss(
                        logits[:, :-1], labels[:, labels.shape[1] - (logits.shape[1] - 1) :]
                    )
                loss.backward()

            for _ in range(3):
                one()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(5):
                one()
            torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - t) / 5
            rows.append(
                {"m": m, "ms_per_fwd_bwd": round(ms, 1), "ms_per_example": round(ms / m, 1)}
            )
        return rows

    def run_variant(attn: str, trim: bool, host_answer_len: bool = True) -> dict:
        model, batches, singles, load_s, build_s = build(attn, host_answer_len)
        trainer = make_trainer(model, batches, trim)
        n_micro = trainer.micro_batches_per_step

        for _ in range(3):
            trainer.train_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t = time.perf_counter()
        for _ in range(profile_steps):
            trainer.train_step()
        torch.cuda.synchronize()
        step_s = (time.perf_counter() - t) / profile_steps

        phase = phases(model, trainer, n_micro, 3)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                trainer.train_step()
            torch.cuda.synchronize()
        kernels = kernel_rows(prof, 3)

        return {
            "attn": attn,
            "trim_logits": trim,
            "host_answer_len": host_answer_len,
            "n_micro_per_step": n_micro,
            "load_s": load_s,
            "build_s": build_s,
            "step_s": round(step_s, 4),
            "ms_per_micro_batch": round(1000 * step_s / n_micro, 1),
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
            "phase_s": phase,
            "top_kernels_ms_per_step": kernels[:8],
            "kernel_total_ms_per_step": round(sum(k[0] for k in kernels), 2),
        }

    variants = []
    for label, attn, trim, host in (
        ("sdpa_trim_hostlen", "sdpa", True, True),
        ("sdpa_notrim_hostlen", "sdpa", False, True),
        ("sdpa_trim_devicelen", "sdpa", True, False),
        ("eager_trim_hostlen", "eager", True, True),
    ):
        try:
            variants.append({"label": label, **run_variant(attn, trim, host)})
        except torch.OutOfMemoryError:
            variants.append({"label": label, "oom": True})

    built = build("sdpa", True)
    out["batch_scaling"] = batch_scaling(built[0], built[2], (1, 2, 4))
    out["variants"] = variants
    out["torch"] = torch.__version__
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=32)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--profile-steps", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    result = run_profile(args.n_examples, args.micro_batch, args.profile_steps)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
