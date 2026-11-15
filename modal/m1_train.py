"""Modal M1 training harness (test-only, not production).

Trains the WP7 projector on the M1 synthetic data with the frozen Qwen3-1.7B
reader and the frozen K3 MoonViT-V2 tower. The recipe is the Baseten reference:
global batch 128, AdamW (0.9, 0.95), LR 3e-4, warmup 50, 2048 steps.

Run a 5-step calibration:

    python3 modal/m1_train.py --seed 0 --calibrate

Run the full 2048 steps:

    python3 modal/m1_train.py --seed 0

Modal is authenticated with the `modal` CLI profile; no token is stored here.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("sightline-m1")
volume = modal.Volume.from_name("sightline-m1", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "torchvision==0.28.0",
        "transformers==5.12.1",
        "tokenizers==0.22.2",
        "safetensors==0.8.0",
        "huggingface-hub==1.25.1",
        "numpy==2.5.1",
        "pillow==12.3.0",
        "einops",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
    .add_local_dir("src", "/root/src")
    .add_local_file(
        "artifacts/k3_vision_f831ab66.safetensors",
        "/root/k3_vision_f831ab66.safetensors",
    )
)

QWEN3_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
IMAGE_PAD = 151655
GLOBAL_BATCH = 128


@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=16384,
    timeout=50400,
    volumes={"/ckpt": volume},
)
def m1_train(
    seed: int,
    max_steps: int = 2048,
    lr: float = 3e-4,
    micro_batch: int = 8,
    calibrate_only: bool = False,
    n_calibrate: int = 8,
) -> dict:
    import resource
    import sys
    import time

    sys.path.insert(0, "/root/src")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sightline.data.m1_synthetic import generate_m1_dataset
    from sightline.projector import Projector
    from sightline.train import Batch, Trainer
    from sightline.vision_k3_encoder import K3VisionEncoder

    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=QWEN3_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", revision=QWEN3_REVISION, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    encoder = K3VisionEncoder(
        weights="/root/k3_vision_f831ab66.safetensors",
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    dataset = generate_m1_dataset(seed=0)
    train = dataset.train[:n_calibrate] if calibrate_only else dataset.train
    features = torch.cat([encoder.encode([e.image]) for e in train], dim=0)

    def make_batch(i: int) -> Batch:
        example = train[i]
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": "<|image_pad|>" * 256 + example.question}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        prompt_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        answer_ids = tok(
            example.answer + tok.eos_token,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        train_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        labels = torch.full_like(train_ids, -100)
        labels[:, prompt_ids.shape[1] :] = answer_ids
        return Batch(train_ids, features[i : i + 1], labels)

    singles = [make_batch(i) for i in range(len(train))]

    def merge(part: list[Batch]) -> Batch:
        return Batch(
            torch.cat([b.input_ids for b in part], dim=0),
            torch.cat([b.features for b in part], dim=0),
            torch.cat([b.labels for b in part], dim=0),
        )

    base = [merge(singles[i : i + micro_batch]) for i in range(0, len(singles), micro_batch)]
    del singles, features

    torch.manual_seed(123)
    projector = Projector(input_width=4096, reader_width=2048)
    trainer = Trainer(
        model,
        projector,
        base,
        micro_batches_per_step=min(GLOBAL_BATCH // micro_batch, len(base)),
        base_lr=lr,
        warmup_steps=50,
        max_steps=max_steps,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        placeholder_id=IMAGE_PAD,
        device=device,
        dtype=torch.bfloat16,
        cycle=True,
    )

    if calibrate_only:
        t = time.perf_counter()
        trainer.train(max_steps=5)
        step_s = (time.perf_counter() - t) / 5
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "calibrate": True,
            "seed": seed,
            "n_examples": len(train),
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "step_s": round(step_s, 3),
            "peak_rss_mb": round(usage.ru_maxrss / 1024, 1),
            "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 1),
            "max_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
            if torch.cuda.is_available()
            else None,
            "elapsed_s": round(time.perf_counter() - started, 2),
        }

    trainer.train(max_steps=max_steps)
    checkpoint_dir = f"/ckpt/seed{seed}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector": trainer.projector.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
            "step": trainer.step,
        },
        f"{checkpoint_dir}/projector.pt",
    )
    volume.commit()

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "calibrate": False,
        "seed": seed,
        "max_steps": max_steps,
        "final_loss": trainer.losses[-1],
        "loss_every_100": trainer.losses[::100],
        "checkpoint": f"{checkpoint_dir}/projector.pt",
        "elapsed_s": round(time.perf_counter() - started, 2),
        "peak_rss_mb": round(usage.ru_maxrss / 1024, 1),
        "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 1),
        "max_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        if torch.cuda.is_available()
        else None,
    }


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--n-calibrate", type=int, default=8)
    args = parser.parse_args()

    async def run() -> dict:
        async with app.run():
            return await m1_train.remote.aio(
                seed=args.seed,
                max_steps=args.max_steps,
                lr=args.lr,
                micro_batch=args.micro_batch,
                calibrate_only=args.calibrate,
                n_calibrate=args.n_calibrate,
            )

    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    suffix = "_calibrate" if args.calibrate else ""
    Path(f"/tmp/opencode/m1b_seed{args.seed}{suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
