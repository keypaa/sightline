#!/usr/bin/env python3
"""Run the M1 training and G1 evaluation on the local GPU.

Usage:
    python tools/run_m1.py --seed 0 --max-steps 2048 --micro-batch 16
    python tools/run_m1.py --seed 0 --probe-steps 20 --micro-batch 16
    python tools/run_m1.py --seed 0 --resume
    python tools/run_m1.py --seed 0 --eval-only

A progress line (step, percentage, loss, median step time, elapsed, ETA) is
printed every --progress-every steps. --resume continues from the checkpoint on
disk, so an interrupted run picks up where it stopped; the checkpoint holds the
projector, optimizer, scheduler, step and RNG states.

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
import contextlib
import io
import json
import random
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from sightline.backbone import splice
from sightline.config import Config, resolve_config
from sightline.data.m1_synthetic import generate_m1_dataset
from sightline.evaluate import g1_pass, g1_report
from sightline.generate import generate_new_tokens
from sightline.progress import ProgressPrinter
from sightline.projector import Projector, calibrate_output_scale
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


@dataclass(frozen=True)
class VisualSpec:
    """How K3 features are presented to the reader.

    The shipped design uses all 256 merged patches. The single-token mode is a
    test-only ablation (DECISIONS 0036): it mean-pools the same features into one
    vector, exactly as the linear probe did. Nothing here is hard-coded in the
    prompt or the splice; every consumer derives its placeholder count from spec.
    """

    n_tokens: int
    pooling: str  # "patches" or "mean"

    @property
    def placeholder(self) -> str:
        return "<|image_pad|>" * self.n_tokens

    def prepare(self, features: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return features.float().mean(dim=1).unsqueeze(1).to(features.dtype)
        if self.pooling == "patches":
            return features
        raise ValueError(f"unknown visual pooling {self.pooling!r}")


VISUAL_SPECS = {
    256: VisualSpec(256, "patches"),
    1: VisualSpec(1, "mean"),
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


def build_batches(examples, features, tok, device, micro_batch, spec: VisualSpec):
    singles = []
    for i, example in enumerate(examples):
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": spec.placeholder + example.question}],
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
        input_ids = torch.cat([b.input_ids for b in part], dim=0)
        features = torch.cat([b.features for b in part], dim=0)
        labels = torch.cat([b.labels for b in part], dim=0)
        return Batch(
            input_ids,
            features,
            labels,
            n_supervised=int((labels != -100).sum(dim=1).max()) if len(labels) else 0,
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


def sample_subset(items: list, n: int | None, seed: int = 0) -> list:
    """Take n items at random, deterministically.

    The M1 splits are ordered by answer, so ``items[:n]`` is a biased sample and
    a fixed stride can alias onto one class when the stride matches the ordering
    period. A seeded shuffle is unbiased and reproducible.
    """

    if n is None or n >= len(items):
        return list(items)
    return random.Random(seed).sample(list(items), n)


def build_fixed_eval(tok, encoder, examples, device, spec: VisualSpec, n: int = 64):
    """A fixed evaluation subset: prompts and features encoded once."""

    examples = sample_subset(list(examples), n, seed=0)
    features = spec.prepare(torch.cat([encoder.encode([e.image]) for e in examples], dim=0))
    prompts = [build_prompt(tok, example, device, spec) for example in examples]
    return examples, features, prompts


def exact_match(model, tok, projector, examples, features, prompts, device) -> dict:
    """Greedy exact match per task over a fixed subset."""

    hits: dict[str, list[bool]] = {}
    for i, example in enumerate(examples):
        got = generate_answer(model, tok, projector, prompts[i], features[i : i + 1], device)
        hits.setdefault(example.task, []).append(got == example.answer)
    return {task: round(sum(v) / len(v), 4) for task, v in hits.items()}


def build_prompt(tok, example, device, spec: VisualSpec):
    text = tok.apply_chat_template(
        [{"role": "user", "content": spec.placeholder + example.question}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    return tok(text, return_tensors="pt")["input_ids"].to(device)


def debug_generations(
    model, tok, encoder, projector, examples, device, n, spec: VisualSpec
) -> None:
    """Print what the model actually says, to tell a decode bug from a mapping bug.

    The examples are sampled with a seeded shuffle (see ``sample_subset``): the
    first n of a split are all one answer, which flatters a model with a colour
    prior.
    """

    examples = sample_subset(list(examples), n)
    features = spec.prepare(torch.cat([encoder.encode([e.image]) for e in examples], dim=0))
    for i, example in enumerate(examples):
        prompt_ids = build_prompt(tok, example, device, spec)
        answer = generate_answer(model, tok, projector, prompt_ids, features[i : i + 1], device)
        print(
            f"[{example.task}] expected={example.answer!r} got={answer!r} "
            f"q={example.question[:70]!r}"
        )


def debug_logits(model, tok, encoder, projector, examples, device, n, spec: VisualSpec) -> None:
    """Compare teacher-forced top-k with greedy decoding on the same examples.

    This separates two failures that look identical from the outside:

    - teacher-forced top-1 is the answer but greedy is not -> the model learned
      and the generation path is wrong (cache, masks, or prompt handling);
    - teacher-forced top-1 is also wrong -> the model never learned, and the
      problem is optimisation or capacity rather than decoding.

    The examples are sampled evenly; the splits are ordered by answer.
    """

    examples = sample_subset(list(examples), n)
    features = spec.prepare(torch.cat([encoder.encode([e.image]) for e in examples], dim=0))
    embed = model.get_input_embeddings()
    teacher_forced = 0
    for i, example in enumerate(examples):
        prompt_ids = build_prompt(tok, example, device, spec)
        answer_ids = tok(
            example.answer + tok.eos_token, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)
        ids = torch.cat([prompt_ids, answer_ids], dim=1)
        with torch.no_grad():
            visual = projector(features[i : i + 1].float())
            spliced = splice(ids, visual, IMAGE_PAD, embed.weight)
            logits = model(inputs_embeds=spliced, attention_mask=torch.ones_like(ids)).logits
        position = prompt_ids.shape[1] - 1
        probabilities = logits[0, position].float().softmax(dim=-1)
        top = torch.topk(probabilities, 5)
        top_tokens = [
            (tok.decode([int(token)]), round(float(p), 3))
            for token, p in zip(top.indices, top.values, strict=True)
        ]
        greedy = generate_answer(model, tok, projector, prompt_ids, features[i : i + 1], device)
        top_answer = str(top_tokens[0][0]).strip().lower().strip(" .!")
        teacher_forced += int(top_answer == example.answer)
        print(
            f"[{example.task}] expected={example.answer!r} "
            f"teacher_forced_top1={top_tokens[0][0]!r} greedy={greedy!r} "
            f"top5={top_tokens}"
        )
    print(f"teacher-forced top-1 correct: {teacher_forced}/{len(examples)}", flush=True)


def chance_rates(examples) -> dict[str, float]:
    """Guessing rate per task, from the distinct answers the test split contains."""

    answers: dict[str, set[str]] = {}
    for example in examples:
        answers.setdefault(example.task, set()).add(example.answer)
    return {task: 1.0 / len(options) for task, options in answers.items()}


def answer_breakdown(examples, indices, correct) -> dict[str, float]:
    """Accuracy of the correct condition, split by answer value.

    The count task is the reason this exists: train and val use counts 1-4 while
    test uses 1-5, so an aggregate score cannot distinguish "the model cannot
    count" from "the model has never seen five".
    """

    buckets: dict[str, list[bool]] = {}
    for i, hit in zip(indices, correct, strict=True):
        buckets.setdefault(examples[i].answer, []).append(bool(hit))
    return {answer: round(sum(v) / len(v), 4) for answer, v in sorted(buckets.items())}


def evaluate(model, tok, encoder, projector, examples, device, spec: VisualSpec):
    features = spec.prepare(torch.cat([encoder.encode([e.image]) for e in examples], dim=0))
    blank = Image.new("RGB", (448, 448), (255, 255, 255))
    blank_features = spec.prepare(encoder.encode([blank]))

    prompts = []
    text_prompts = []
    for example in examples:
        text = tok.apply_chat_template(
            [{"role": "user", "content": spec.placeholder + example.question}],
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
    breakdown = {}
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

        breakdown[task] = answer_breakdown(examples, indices, correct)
        per_task[task] = {
            "correct": correct,
            "blank": blank_c,
            "mismatched": mismatched_c,
            "no_tokens": no_tokens_c,
            "untrained": untrained_c,
        }

    report = g1_report(per_task, chance_rates(examples))
    return {
        "report": report,
        "g1_pass": g1_pass(report),
        "per_task": per_task,
        "answer_breakdown": breakdown,
    }


def sweep_summary(trainer, window: int = 25) -> dict:
    """Loss and gradient-norm summary for one sweep point.

    The gradient norm is the diagnostic: if it stays near zero the projector is
    not receiving signal through the frozen reader, and no learning rate will
    fix that.
    """

    losses = trainer.losses[-window:]
    norms = [float(n) for n in list(trainer.grad_norms)[-window:]]
    return {
        "final_loss": trainer.losses[-1] if trainer.losses else None,
        "mean_loss_last_window": sum(losses) / len(losses) if losses else None,
        "mean_grad_norm": sum(norms) / len(norms) if norms else None,
        "max_grad_norm": max(norms) if norms else None,
    }


def build_record(seed: int, outcome: str, spec: VisualSpec) -> RunRecord:
    return RunRecord(
        Path(f"experiments/m1b_train/seed{seed}"),
        dry_run=False,
        config=resolve_config(Config()),
        runtime={
            "n_visual_tokens": spec.n_tokens,
            "reader_hidden_size": 2048,
            "projector_hidden_width": 4096,
            "k3_tower_width": 4096,
            "derived_at_runtime": True,
        },
        chat_template=CHAT_TEMPLATE,
        outcome=outcome,
    )


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from artifacts/m1b/seed<seed>/projector.pt instead of restarting",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="print a progress line every N steps (0 silences)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=200,
        help="write a checkpoint every N steps so --resume has a recent state (0 disables)",
    )
    parser.add_argument(
        "--no-trim-logits",
        action="store_true",
        help="project the vocabulary at every position; baseline for the logits-trim optimization",
    )
    parser.add_argument(
        "--eval-split",
        choices=("train", "val", "test"),
        default="test",
        help="which split to evaluate; train is the diagnostic for memorisation",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="evaluate only the first N examples of the split",
    )
    parser.add_argument(
        "--debug-generations",
        type=int,
        default=0,
        help="print N expected-vs-generated answers and exit; no full evaluation",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="append loss, gradient norm and step time to metrics.jsonl every N steps",
    )
    parser.add_argument(
        "--track-grad-norm",
        action="store_true",
        help="log the projector's gradient norm (costs one device sync per step)",
    )
    parser.add_argument(
        "--lr-sweep",
        type=str,
        default=None,
        help="comma-separated learning rates; run each for --max-steps and compare",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="train and print the loss/gradient summary, but skip the G1 evaluation",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="score greedy exact match on a fixed 64-example subset every N steps",
    )
    parser.add_argument(
        "--no-output-scale",
        action="store_true",
        help="skip matching the projector output RMS to the reader's embedding RMS",
    )
    parser.add_argument(
        "--visual-tokens",
        type=int,
        choices=(256, 1),
        default=256,
        help="256 merged patches (shipped) or 1 mean-pooled token (test-only ablation)",
    )
    parser.add_argument(
        "--debug-logits",
        type=int,
        default=0,
        help="print teacher-forced top-5 against greedy decoding for N examples",
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

    spec = VISUAL_SPECS[args.visual_tokens]
    dataset = generate_m1_dataset(seed=0)
    train_features = spec.prepare(
        torch.cat([encoder.encode([e.image]) for e in dataset.train], dim=0)
    )
    base = build_batches(dataset.train, train_features, tok, "cuda", args.micro_batch, spec)
    del train_features

    def make_trainer(lr: float) -> Trainer:
        torch.manual_seed(123)
        projector = Projector(input_width=4096, reader_width=2048)
        trainer = Trainer(
            model,
            projector,
            base,
            micro_batches_per_step=min(128 // args.micro_batch, len(base)),
            base_lr=lr,
            warmup_steps=50,
            max_steps=args.max_steps,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            grad_clip=1.0,
            placeholder_id=IMAGE_PAD,
            device="cuda",
            dtype=torch.bfloat16,
            cycle=True,
            trim_logits=not args.no_trim_logits,
            track_grad_norm=args.track_grad_norm,
        )
        if not args.no_output_scale:
            scale = calibrate_output_scale(
                projector, model.get_input_embeddings().weight, base[0].features
            )
            print(f"output_scale: {scale:.6g}", flush=True)
        return trainer

    if args.lr_sweep:
        rows = []
        for lr in (float(x) for x in args.lr_sweep.split(",")):
            trainer = make_trainer(lr)
            started = time.perf_counter()
            trainer.train(max_steps=args.max_steps)
            row = {
                "lr": lr,
                "elapsed_s": round(time.perf_counter() - started, 1),
                **sweep_summary(trainer),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        record = build_record(args.seed, f"M1b: LR sweep over {len(rows)} values", spec)
        record.write()
        record.append_metric({"event": "m1b_lr_sweep", "seed": args.seed, "rows": rows})
        print(f"{'lr':>10} {'final':>10} {'mean last 25':>13} {'mean gnorm':>12}")
        for row in rows:
            gnorm = "n/a" if row["mean_grad_norm"] is None else f"{row['mean_grad_norm']:.3e}"
            print(
                f"{row['lr']:>10.0e} {row['final_loss']:>10.4f} "
                f"{row['mean_loss_last_window']:>13.4f} {gnorm:>12}"
            )
        return 0

    trainer = make_trainer(args.lr)

    if args.debug_logits or args.debug_generations:
        # Fail fast: the diagnostics run after training, so a wiring error there
        # would otherwise surface only after the whole run. Rehearse them once on
        # the untrained projector, with a single example, and say so.
        with contextlib.redirect_stdout(io.StringIO()):
            if args.debug_logits:
                debug_logits(
                    model, tok, encoder, trainer.projector, dataset.train[:1], "cuda", 1, spec
                )
            else:
                debug_generations(
                    model, tok, encoder, trainer.projector, dataset.train[:1], "cuda", 1, spec
                )
        print("diagnostic self-check ok", flush=True)

    fixed = None
    if args.eval_every > 0:
        fixed = build_fixed_eval(tok, encoder, dataset.train, "cuda", spec)
        print(
            "fixed eval subset: "
            + json.dumps(
                {t: sum(1 for e in fixed[0] if e.task == t) for t in ("color", "shape", "count")}
            ),
            flush=True,
        )

    started = time.perf_counter()
    checkpoint = Path(f"artifacts/m1b/seed{args.seed}/projector.pt")
    progress = ProgressPrinter(
        total=args.max_steps, every=args.progress_every, label=f"[seed {args.seed}]"
    )
    train_started = time.perf_counter()
    record = build_record(args.seed, "M1b: training + G1 evaluation", spec)
    record.write()

    def on_step(info: dict) -> None:
        progress({**info, "elapsed_s": time.perf_counter() - train_started})
        if args.log_every > 0 and info["step"] % args.log_every == 0:
            record.append_metric(
                {
                    "event": "m1b_step",
                    "seed": args.seed,
                    "step": info["step"],
                    "loss": info["loss"],
                    "grad_norm": info["grad_norm"],
                    "step_s": info["step_s"],
                    "lr": args.lr,
                }
            )
        if fixed is not None and info["step"] % args.eval_every == 0:
            accuracy = exact_match(model, tok, trainer.projector, *fixed, "cuda")
            record.append_metric(
                {
                    "event": "m1b_acc",
                    "seed": args.seed,
                    "step": info["step"],
                    "accuracy": accuracy,
                    "loss": info["loss"],
                }
            )
            print(
                f"[seed {args.seed}]  acc {json.dumps(accuracy)} at step {info['step']}", flush=True
            )
        if (
            args.save_every > 0
            and info["step"] % args.save_every == 0
            and info["step"] != args.max_steps
        ):
            trainer.save_checkpoint(checkpoint)
            print(
                f"[seed {args.seed}]  checkpoint at step {info['step']}/{args.max_steps}",
                flush=True,
            )

    if args.probe_steps is not None:
        probe = trainer.probe(args.probe_steps)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        probe["peak_rss_mb"] = round(usage.ru_maxrss / 1024, 1)
        probe["elapsed_s"] = round(time.perf_counter() - started, 2)
        probe["seed"] = args.seed
        probe["micro_batch"] = args.micro_batch
        probe["trim_logits"] = not args.no_trim_logits
        record = RunRecord(
            Path(f"experiments/m1b_probe/micro{args.micro_batch}"),
            dry_run=False,
            config=resolve_config(Config()),
            runtime={
                "n_visual_tokens": spec.n_tokens,
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

    trainer.on_step = on_step
    if args.eval_only:
        state = torch.load(checkpoint, weights_only=False)
        projector_state = state["projector"]
        if "output_scale" not in projector_state:
            print(
                "warning: checkpoint predates output scaling; loading with output_scale = 1.0",
                flush=True,
            )
            trainer.projector.load_state_dict(projector_state, strict=False)
        else:
            trainer.projector.load_state_dict(projector_state)
        trainer.step = state["step"]
    elif args.resume:
        if not checkpoint.exists():
            raise SystemExit(f"--resume but no checkpoint at {checkpoint}")
        trainer.load_state_dict(torch.load(checkpoint, weights_only=False))
        print(f"resumed at step {trainer.step}/{args.max_steps}", flush=True)
        trainer.train(max_steps=args.max_steps)
        trainer.save_checkpoint(checkpoint)
    else:
        trainer.train(max_steps=args.max_steps)
        trainer.save_checkpoint(checkpoint)

    split = sample_subset(list(getattr(dataset, args.eval_split)), args.eval_limit)

    if args.no_eval:
        norms = [float(n) for n in list(trainer.grad_norms)[-25:]]
        losses = trainer.losses[-25:]
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "steps": trainer.step,
                    "final_loss": trainer.losses[-1],
                    "mean_loss_last_25": sum(losses) / len(losses) if losses else None,
                    "mean_grad_norm_last_25": sum(norms) / len(norms) if norms else None,
                    "max_grad_norm_last_25": max(norms) if norms else None,
                    "median_step_s": float(median(trainer.step_times))
                    if trainer.step_times
                    else None,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    if args.debug_logits:
        debug_logits(model, tok, encoder, trainer.projector, split, "cuda", args.debug_logits, spec)
        return 0

    if args.debug_generations:
        debug_generations(
            model, tok, encoder, trainer.projector, split, "cuda", args.debug_generations, spec
        )
        return 0

    eval_result = evaluate(model, tok, encoder, trainer.projector, split, "cuda", spec)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = {
        "seed": args.seed,
        "max_steps": args.max_steps,
        "micro_batch": args.micro_batch,
        "trim_logits": not args.no_trim_logits,
        "eval_split": args.eval_split,
        "eval_limit": args.eval_limit,
        "final_loss": trainer.losses[-1] if trainer.losses else None,
        "loss_every_100": trainer.losses[::100] if trainer.losses else [],
        "eval": eval_result,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "peak_rss_mb": round(usage.ru_maxrss / 1024, 1),
        "max_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
    }

    # run.json and metrics.jsonl were created before training so step metrics
    # could be appended as the run progressed; nothing to write again here.
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
    record.append_metric(
        {
            "event": "g1",
            "seed": args.seed,
            "g1_pass": eval_result["g1_pass"],
            "report": eval_result["report"],
        }
    )
    record.append_metric(
        {"event": "g1_raw", "seed": args.seed, "per_task": eval_result["per_task"]}
    )

    print(
        json.dumps(
            {
                **summary,
                "eval": {
                    "report": eval_result["report"],
                    "g1_pass": eval_result["g1_pass"],
                    "answer_breakdown": eval_result["answer_breakdown"],
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
