"""Modal WP10 overfit harness (test-only, not production).

Trains the WP7 projector on the 32 WP9 scenes against the frozen Qwen3-1.7B
reader and the frozen K3 MoonViT-V2 tower. Evaluates exact match and the blank,
shuffled and swapped controls.

Runs on an A10G (24 GB): the T4 OOMs at micro-batch 8 (13.75 GiB allocated)
and still OOMs at micro-batch 4 (13.24 GiB). The A10G fits the original
micro-batch 8.

Run one seed:

    python3 modal/wp10_overfit.py --seed 0

Modal is authenticated with the `modal` CLI profile; no token is stored here.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("sightline-wp10")

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
PAD = 151643


@app.function(image=image, gpu="A10G", cpu=4.0, memory=16384, timeout=1800)
def overfit(
    seed: int,
    max_steps: int = 500,
    lr: float = 1e-3,
    micro_batch: int = 8,
) -> dict:
    import resource
    import sys
    import time

    sys.path.insert(0, "/root/src")

    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sightline.backbone import splice
    from sightline.data.synthetic import generate_dataset
    from sightline.evaluate import EvalExample, evaluate_controls
    from sightline.generate import generate_new_tokens
    from sightline.objective import answer_only_loss
    from sightline.projector import Projector
    from sightline.train import Batch, Trainer
    from sightline.vision_k3_encoder import K3VisionEncoder

    started = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Frozen reader.
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=QWEN3_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", revision=QWEN3_REVISION, dtype=torch.bfloat16
    ).to(device)
    model.eval()

    # Frozen K3 tower.
    encoder = K3VisionEncoder(
        weights="/root/k3_vision_f831ab66.safetensors",
        device=device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # Data and precomputed features.
    examples = generate_dataset(seed=0)
    blank_image = Image.new("RGB", (448, 448), (255, 255, 255))
    features_real = torch.cat([encoder.encode([e.image]) for e in examples], dim=0)
    feature_blank = encoder.encode([blank_image])

    # Prompt: 256 image pads plus the constant question.
    prompt_text = tok.apply_chat_template(
        [{"role": "user", "content": "<|image_pad|>" * 256 + "What color is the shape?"}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    prompt_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)

    # Training batches: one per example.
    def make_batch(i: int) -> Batch:
        example = examples[i]
        answer_ids = tok(
            example.answer + tok.eos_token,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(device)
        train_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        labels = torch.full_like(train_ids, -100)
        labels[:, prompt_ids.shape[1] :] = answer_ids
        return Batch(train_ids, features_real[i : i + 1], labels)

    singles = [make_batch(i) for i in range(len(examples))]

    def merge(part: list[Batch]) -> Batch:
        max_len = max(b.input_ids.shape[1] for b in part)
        input_ids = torch.full((len(part), max_len), PAD, dtype=torch.long)
        labels = torch.full((len(part), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(part), max_len), dtype=torch.long)
        for i, batch in enumerate(part):
            length = batch.input_ids.shape[1]
            input_ids[i, :length] = batch.input_ids[0]
            labels[i, :length] = batch.labels[0]
            attention_mask[i, :length] = 1
        return Batch(
            input_ids,
            torch.cat([batch.features for batch in part], dim=0),
            labels,
            attention_mask,
        )

    base = [merge(singles[i : i + micro_batch]) for i in range(0, len(singles), micro_batch)]
    trainer_batches = base * max_steps

    torch.manual_seed(123)
    projector = Projector(input_width=4096, reader_width=2048)
    trainer = Trainer(
        model,
        projector,
        trainer_batches,
        micro_batches_per_step=len(base),
        base_lr=lr,
        warmup_steps=0,
        max_steps=max_steps,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        placeholder_id=IMAGE_PAD,
        device=device,
        dtype=torch.bfloat16,
    )

    # Evaluation.
    def generate_answer(features: torch.Tensor) -> str:
        with torch.no_grad():
            visual = projector(features.float())
            inputs_embeds = splice(
                prompt_ids, visual, IMAGE_PAD, model.get_input_embeddings().weight
            )
            generated = generate_new_tokens(
                model,
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=8,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        return tok.decode(generated[0], skip_special_tokens=True).strip().lower().strip(" .!")

    def feature_for(example, condition: str) -> torch.Tensor:
        n = len(examples)
        if condition == "real":
            return features_real[example.index : example.index + 1]
        if condition == "blank":
            return feature_blank
        if condition == "shuffled":
            j = (example.index + 1) % n
            return features_real[j : j + 1]
        if condition == "swapped":
            j = (example.index + 16) % n
            return features_real[j : j + 1]
        raise ValueError(condition)

    def evaluate_exact(condition: str) -> float:
        correct = sum(
            generate_answer(feature_for(example, condition)) == example.answer
            for example in examples
        )
        return correct / len(examples)

    def image_for(example, condition: str):
        n = len(examples)
        if condition == "real":
            return example.image
        if condition == "blank":
            return blank_image
        if condition == "shuffled":
            return examples[(example.index + 1) % n].image
        if condition == "swapped":
            return examples[(example.index + 16) % n].image
        raise ValueError(condition)

    feature_by_image = {id(e.image): features_real[i : i + 1] for i, e in enumerate(examples)}
    feature_by_image[id(blank_image)] = feature_blank

    def generate_fn(image, condition: str) -> str:
        return generate_answer(feature_by_image[id(image)])

    eval_examples = [EvalExample(e.index, e.answer, e.image) for e in examples]

    # Train in chunks; evaluate every 25 steps.
    eval_every = 25
    history = []
    steps_to_99 = None
    for step in range(0, max_steps, eval_every):
        trainer.train(max_steps=min(step + eval_every, max_steps))
        real = evaluate_exact("real")
        history.append({"step": trainer.step, "loss": trainer.losses[-1], "real_exact_match": real})
        if steps_to_99 is None and real >= 0.99:
            steps_to_99 = trainer.step

    controls = evaluate_controls(generate_fn, eval_examples, image_for)

    # Teacher-forced answer loss gap: blank minus real.
    def loss_for(example, condition: str) -> float:
        batch = singles[example.index]
        features = feature_for(example, condition)
        with torch.no_grad():
            visual = projector(features.float())
            inputs_embeds = splice(
                batch.input_ids, visual, IMAGE_PAD, model.get_input_embeddings().weight
            )
            logits = model(inputs_embeds=inputs_embeds).logits
            return answer_only_loss(logits[:, :-1], batch.labels[:, 1:]).item()

    with torch.no_grad():
        gaps = [loss_for(e, "blank") - loss_for(e, "real") for e in examples]
    answer_loss_gap = sum(gaps) / len(gaps)

    return {
        "seed": seed,
        "max_steps": max_steps,
        "micro_batch": micro_batch,
        "lr": lr,
        "steps_to_99": steps_to_99,
        "history": history,
        "controls": controls,
        "answer_loss_gap_real_vs_blank": answer_loss_gap,
        "per_step_loss": trainer.losses,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "max_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        if torch.cuda.is_available()
        else None,
    }


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--micro-batch", type=int, default=8)
    args = parser.parse_args()

    async def run() -> dict:
        async with app.run():
            return await overfit.remote.aio(
                seed=args.seed,
                max_steps=args.max_steps,
                lr=args.lr,
                micro_batch=args.micro_batch,
            )

    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    Path(f"/tmp/opencode/wp10_seed{args.seed}.json").write_text(json.dumps(result, indent=2) + "\n")
