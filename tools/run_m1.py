#!/usr/bin/env python3
"""Run the M1 training and G1 evaluation on the local GPU.

Usage:
    python tools/run_m1.py --seed 0 --max-steps 2048 --micro-batch 16
    python tools/run_m1.py --seed 0 --probe-steps 20 --micro-batch 16
    python tools/run_m1.py --seed 0 --eval-only

Settings are the Baseten reference: global batch 128, AdamW (0.9, 0.95),
weight decay 0.1, LR 3e-4, warmup 50, 2048 steps, bf16 autocast, fp32
projector and optimizer state.

Outputs:
    experiments/m1b_train/seed<seed>/run.json
    experiments/m1b_train/seed<seed>/metrics.jsonl
    artifacts/m1b/seed<seed>/projector.pt
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from sightline.backbone import splice
from sightline.config import Config, resolve_config
from sightline.data.m1_synthetic import generate_m1_dataset
from sightline.evaluate import g1_pass, g1_report
from sightline.generate import generate_new_tokens
from sightline.projector import Projector
from sightline.record import RunRecord
from sightline.train import Batch, Trainer
from sightline.vision_k3_encoder import K3VisionEncoder

QWEN3_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
IMAGE_PAD = 151655
MODEL_ROOT = Path("artifacts/models/Qwen3-1.7B")
K3_WEIGHTS = Path("artifacts/k3_vision_f831ab66.safetensors")
CHAT_TEMPLATE = {
    "tokenizer_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "rendered_sha256": "1d91a8fec65d03716b986d047a50d68711028dd97f742f687ef69bee7379c454",
    "enable_thinking": False,
}


def load_reader():
    if (MODEL_ROOT / "config.json").exists():
        tok = AutoTokenizer.from_pretrained(str(MODEL_ROOT))
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_ROOT), dtype=torch.bfloat16)
    else:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", revision=QWEN3_REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B", revision=QWEN3_REVISION, dtype=torch.bfloat16
        )
    return tok, model.eval()


def build_batches(examples, features, tok, device, micro_batch):
    singles = []
    for i, example in enumerate(examples):
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
        singles.append(Batch(train_ids, features[i : i + 1], labels))

    def merge(part):
        return Batch(
            torch.cat([b.input_ids for b in part], dim=0),
            torch.cat([b.features for b in part], dim=0),
            torch.cat([b.labels for b in part], dim=0),
        )

    return [merge(singles[i : i + micro_batch]) for i in range(0, len(singles), micro_batch)]


def generate_answer(model, tok, projector, prompt_ids, features, device):
    with torch.no_grad():
        visual = projector(features.float())
        inputs_embeds = splice(prompt_ids, visual, IMAGE_PAD, model.get_input_embeddings().weight)
        generated = generate_new_tokens(
            model,
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=8,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(generated[0], skip_special_tokens=True).strip().lower().strip(" .!")


def chance_rates(examples) -> dict[str, float]:
    """Guessing rate per task, from the distinct answers the test split contains."""

    answers: dict[str, set[str]] = {}
    for example in examples:
        answers.setdefault(example.task, set()).add(example.answer)
    return {task: 1.0 / len(options) for task, options in answers.items()}


def evaluate(model, tok, encoder, projector, examples, device):
    features = torch.cat([encoder.encode([e.image]) for e in examples], dim=0)
    blank = Image.new("RGB", (448, 448), (255, 255, 255))
    blank_features = encoder.encode([blank])

    prompts = []
    text_prompts = []
    for example in examples:
        text = tok.apply_chat_template(
            [{"role": "user", "content": "<|image_pad|>" * 256 + example.question}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        prompts.append(tok(text, return_tensors="pt")["input_ids"].to(device))
        text = tok.apply_chat_template(
            [{"role": "user", "content": example.question}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        text_prompts.append(tok(text, return_tensors="pt")["input_ids"].to(device))

    untrained = Projector(input_width=4096, reader_width=2048).to(device)
    per_task = {}
    for task in ("color", "shape", "count"):
        indices = [i for i, e in enumerate(examples) if e.task == task]
        correct, blank_c, mismatched_c, no_tokens_c, untrained_c = [], [], [], [], []
        for i in indices:
            example = examples[i]
            answer = generate_answer(model, tok, projector, prompts[i], features[i : i + 1], device)
            correct.append(answer == example.answer)

            answer = generate_answer(model, tok, projector, prompts[i], blank_features, device)
            blank_c.append(answer == example.answer)

            j = next(
                (k for k in indices if k != i and examples[k].answer != example.answer),
                i,
            )
            answer = generate_answer(model, tok, projector, prompts[i], features[j : j + 1], device)
            mismatched_c.append(answer == example.answer)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=text_prompts[i],
                    max_new_tokens=8,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id,
                )
            answer = tok.decode(generated[0], skip_special_tokens=True).strip().lower().strip(" .!")
            no_tokens_c.append(answer == example.answer)

            answer = generate_answer(model, tok, untrained, prompts[i], features[i : i + 1], device)
            untrained_c.append(answer == example.answer)

        per_task[task] = {
            "correct": correct,
            "blank": blank_c,
            "mismatched": mismatched_c,
            "no_tokens": no_tokens_c,
            "untrained": untrained_c,
        }

    report = g1_report(per_task, chance_rates(examples))
    return {"report": report, "g1_pass": g1_pass(report), "per_task": per_task}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=None,
        help="time N steps (plus 2 untimed warmup) and exit; writes no checkpoint",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    print("arch_list:", torch.cuda.get_arch_list())
    print(
        "device:",
        torch.cuda.get_device_name(0),
        "memory_gb:",
        round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    )

    tok, model = load_reader()
    model = model.to("cuda")
    encoder = K3VisionEncoder(
        weights=str(K3_WEIGHTS),
        device="cuda",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    dataset = generate_m1_dataset(seed=0)
    train_features = torch.cat([encoder.encode([e.image]) for e in dataset.train], dim=0)
    base = build_batches(dataset.train, train_features, tok, "cuda", args.micro_batch)
    del train_features

    torch.manual_seed(123)
    projector = Projector(input_width=4096, reader_width=2048)
    trainer = Trainer(
        model,
        projector,
        base,
        micro_batches_per_step=min(128 // args.micro_batch, len(base)),
        base_lr=args.lr,
        warmup_steps=50,
        max_steps=args.max_steps,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        grad_clip=1.0,
        placeholder_id=IMAGE_PAD,
        device="cuda",
        dtype=torch.bfloat16,
        cycle=True,
    )

    started = time.perf_counter()
    checkpoint = Path(f"artifacts/m1b/seed{args.seed}/projector.pt")
    if args.probe_steps is not None:
        probe = trainer.probe(args.probe_steps)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        probe["peak_rss_mb"] = round(usage.ru_maxrss / 1024, 1)
        probe["elapsed_s"] = round(time.perf_counter() - started, 2)
        probe["seed"] = args.seed
        probe["micro_batch"] = args.micro_batch
        record = RunRecord(
            Path(f"experiments/m1b_probe/micro{args.micro_batch}"),
            dry_run=False,
            config=resolve_config(Config()),
            runtime={
                "n_visual_tokens": 256,
                "reader_hidden_size": 2048,
                "projector_hidden_width": 4096,
                "k3_tower_width": 4096,
                "derived_at_runtime": True,
            },
            chat_template=CHAT_TEMPLATE,
            outcome=f"M1b probe: micro batch {args.micro_batch}, {args.probe_steps} timed steps",
        )
        record.write()
        record.append_metric({"event": "m1b_probe", **probe})
        print(json.dumps(probe, indent=2))
        return 0

    if args.eval_only:
        state = torch.load(checkpoint, weights_only=False)
        trainer.projector.load_state_dict(state["projector"])
        trainer.step = state["step"]
    else:
        trainer.train(max_steps=args.max_steps)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "projector": trainer.projector.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": trainer.scheduler.state_dict(),
                "step": trainer.step,
            },
            checkpoint,
        )

    eval_result = evaluate(model, tok, encoder, trainer.projector, dataset.test, "cuda")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "micro_batch": args.micro_batch,
        "final_loss": trainer.losses[-1] if trainer.losses else None,
        "loss_every_100": trainer.losses[::100] if trainer.losses else [],
        "eval": eval_result,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "peak_rss_mb": round(usage.ru_maxrss / 1024, 1),
        "max_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }

    record = RunRecord(
        Path(f"experiments/m1b_train/seed{args.seed}"),
        dry_run=False,
        config=resolve_config(Config()),
        runtime={
            "n_visual_tokens": 256,
            "reader_hidden_size": 2048,
            "projector_hidden_width": 4096,
            "k3_tower_width": 4096,
            "derived_at_runtime": True,
        },
        chat_template=CHAT_TEMPLATE,
        outcome="M1b: training + G1 evaluation",
    )
    record.write()
    record.append_metric(
        {
            "event": "m1b",
            "seed": args.seed,
            "max_steps": args.max_steps,
            "micro_batch": args.micro_batch,
            "final_loss": summary["final_loss"],
            "g1_pass": eval_result["g1_pass"],
            "elapsed_s": summary["elapsed_s"],
            "peak_rss_mb": summary["peak_rss_mb"],
            "max_gpu_memory_mb": summary["max_gpu_memory_mb"],
        }
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
