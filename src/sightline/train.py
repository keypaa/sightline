"""Training loop, checkpoints and resume.

Only the projector trains. The loop uses gradient accumulation, answer-only
cross-entropy, AdamW, a warmup-then-cosine schedule, gradient clipping and
bf16 autocast on CUDA. Checkpoints hold the projector, optimizer, scheduler,
step and RNG states so a run can resume exactly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
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
        self.optimizer = torch.optim.AdamW(
            projector.parameters(),
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
        self.model.eval()
        self.projector.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        start = self.step * self.micro_batches_per_step
        autocast_enabled = self.dtype == torch.bfloat16 and self.device.type == "cuda"
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            for batch in self.batches[start : start + self.micro_batches_per_step]:
                input_ids = batch.input_ids.to(self.device)
                features = batch.features.to(self.device)
                labels = batch.labels.to(self.device)
                embed = self.model.get_input_embeddings()
                visual = self.projector(features)
                spliced = splice(input_ids, visual, self.placeholder_id, embed.weight)
                logits = self.model(inputs_embeds=spliced).logits
                loss = answer_only_loss(logits[:, :-1], labels[:, 1:])
                (loss / self.micro_batches_per_step).backward()
                total_loss += loss.item() / self.micro_batches_per_step

        torch.nn.utils.clip_grad_norm_(self.projector.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.losses.append(total_loss)
        self.step += 1

    def train(self, max_steps: int | None = None) -> list[float]:
        if max_steps is None:
            max_steps = self.max_steps
        while self.step < max_steps:
            self.train_step()
        return self.losses

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

        rng = state["rng"]
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["random"])

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, weights_only=False))
