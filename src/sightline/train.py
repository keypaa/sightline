"""Training loop, checkpoints and resume.

Only the projector trains. The loop uses gradient accumulation, answer-only
cross-entropy, AdamW, a warmup-then-cosine schedule, gradient clipping and
bf16 autocast on CUDA. Checkpoints hold the projector, optimizer, scheduler,
step and RNG states so a run can resume exactly.
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import torch

from sightline.backbone import splice
from sightline.objective import answer_only_loss
from sightline.scheduler import warmup_cosine_lr


@dataclass(frozen=True)
class Batch:
    input_ids: torch.Tensor
    features: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor | None = None


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        projector: torch.nn.Module,
        batches: list[Batch],
        *,
        micro_batches_per_step: int,
        base_lr: float,
        warmup_steps: int,
        max_steps: int,
        weight_decay: float,
        betas: tuple[float, float],
        grad_clip: float,
        placeholder_id: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        cycle: bool = False,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.projector = projector
        self.batches = batches
        self.micro_batches_per_step = micro_batches_per_step
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.placeholder_id = placeholder_id
        self.device = torch.device(device)
        self.dtype = dtype
        self.cycle = cycle
        self.on_step = on_step
        self.step_times: deque[float] = deque(maxlen=50)
        self.projector = projector.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=base_lr,
            betas=betas,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: warmup_cosine_lr(step, max_steps, warmup_steps, base_lr),
        )
        self.step = 0
        self.losses: list[float] = []

    def train_step(self) -> None:
        started = time.perf_counter()
        self.model.eval()
        self.projector.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        start = self.step * self.micro_batches_per_step
        if self.cycle:
            start = start % len(self.batches)
            indices = [(start + i) % len(self.batches) for i in range(self.micro_batches_per_step)]
        else:
            indices = range(start, start + self.micro_batches_per_step)
        autocast_enabled = self.dtype == torch.bfloat16 and self.device.type == "cuda"
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            for idx in indices:
                batch = self.batches[idx]
                input_ids = batch.input_ids.to(self.device)
                features = batch.features.to(self.device)
                labels = batch.labels.to(self.device)
                embed = self.model.get_input_embeddings()
                visual = self.projector(features)
                spliced = splice(input_ids, visual, self.placeholder_id, embed.weight)
                logits = self.model(
                    inputs_embeds=spliced, attention_mask=batch.attention_mask
                ).logits
                loss = answer_only_loss(logits[:, :-1], labels[:, 1:])
                (loss / self.micro_batches_per_step).backward()
                total_loss += loss.item() / self.micro_batches_per_step

        torch.nn.utils.clip_grad_norm_(self.projector.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.losses.append(total_loss)
        self.step += 1
        self.step_times.append(time.perf_counter() - started)
        if self.on_step is not None:
            self.on_step(
                {
                    "step": self.step,
                    "loss": total_loss,
                    "step_s": self.step_times[-1],
                    "median_step_s": float(median(self.step_times)),
                    "step_times": len(self.step_times),
                }
            )

    def train(self, max_steps: int | None = None) -> list[float]:
        if max_steps is None:
            max_steps = self.max_steps
        while self.step < max_steps:
            self.train_step()
        return self.losses

    def probe(self, steps: int, warmup: int = 2) -> dict[str, Any]:
        """Time ``steps`` optimizer steps after ``warmup`` untimed ones.

        Used to record a baseline before an optimization (project rule 9):
        the warmup steps absorb cudnn autotuning and allocator growth, so the
        reported ``step_s`` is steady-state. No checkpoint is written; the
        caller discards the probe run.
        """

        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        if warmup:
            self.train(max_steps=self.step + warmup)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        first = len(self.losses)
        started = time.perf_counter()
        self.train(max_steps=self.step + steps)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        return {
            "steps": steps,
            "warmup": warmup,
            "step_s": round(elapsed / steps, 4),
            "loss_first": self.losses[first],
            "loss_last": self.losses[-1],
            "peak_gpu_memory_mb": round(
                torch.cuda.max_memory_allocated(self.device) / 1024 / 1024, 1
            )
            if self.device.type == "cuda"
            else None,
        }

    def state_dict(self) -> dict[str, Any]:
        cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        return {
            "projector": self.projector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": cuda_rng,
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.projector.load_state_dict(state["projector"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]

        rng = state.get("rng") if isinstance(state, dict) else None
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["random"])

    def save_checkpoint(self, path: str | Path) -> None:
        """Write a checkpoint atomically.

        The state goes to a temporary file first and is then renamed, so an
        interrupted save cannot leave a half-written checkpoint behind. This is
        what makes periodic checkpointing safe to run during training.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        torch.save(self.state_dict(), tmp)
        tmp.replace(path)

    def load_checkpoint(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, weights_only=False))
