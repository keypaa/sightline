"""MoonViT-V2 encoder wrapper.

Loads the K3 vision tower from the pinned configuration and the extracted
tower weights, preprocesses images with the official K3 processor, and returns
merged patch features of shape ``(batch, tokens, 4096)``.

The token count is derived from ``grid_thws`` and the merge kernel size; it is
never hard-coded. The tower is frozen and runs under ``torch.no_grad()``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from sightline.vision_k3 import MoonViT3dPretrainedModel, VisionTowerConfig
from sightline.vision_k3.configuration_kimi_k3 import KimiK3Config
from sightline.vision_k3.kimi_k3_vision_processing import KimiK3VisionProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_K3_DIR = REPO_ROOT / "artifacts" / "k3_ref"
DEFAULT_WEIGHTS = REPO_ROOT / "artifacts" / "k3_vision_f831ab66.safetensors"
ALLOWED_ATTENTION = ("eager", "flash_attention_2")


class K3VisionEncoder:
    def __init__(
        self,
        k3_dir: str | Path = DEFAULT_K3_DIR,
        weights: str | Path = DEFAULT_WEIGHTS,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        attn_implementation: str = "eager",
    ) -> None:
        if attn_implementation not in ALLOWED_ATTENTION:
            raise ValueError(
                f"attn_implementation must be one of {ALLOWED_ATTENTION}, "
                f"got {attn_implementation!r}"
            )

        k3_dir = Path(k3_dir)
        config = KimiK3Config.from_pretrained(str(k3_dir))
        vt_config = VisionTowerConfig(config.vision_config)
        vt_config._attn_implementation = attn_implementation

        model = MoonViT3dPretrainedModel(vt_config)
        state = load_file(str(weights))
        tower_state = {
            key[len("vision_tower.") :]: value
            for key, value in state.items()
            if key.startswith("vision_tower.")
        }
        missing, unexpected = model.load_state_dict(tower_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"tower weight mismatch: {len(missing)} missing, {len(unexpected)} unexpected"
            )

        self.device = torch.device(device)
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=dtype)
        self.model.eval()
        self.merge_kernel_size = vt_config.merge_kernel_size
        self.patch_size = vt_config.patch_size

        preprocessor = json.loads((k3_dir / "preprocessor_config.json").read_text())
        self.processor = KimiK3VisionProcessor.from_dict(preprocessor)

    def n_visual_tokens(self, grid_thws: torch.Tensor) -> int:
        """Return the merged token count for the first image in ``grid_thws``."""

        _, height, width = (int(v) for v in grid_thws[0].tolist())
        kernel_h, kernel_w = self.merge_kernel_size
        return (height // kernel_h) * (width // kernel_w)

    def preprocess(self, images: Sequence[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        medias = [{"type": "image", "image": image.convert("RGB")} for image in images]
        batch = self.processor.preprocess(medias, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(device=self.device, dtype=self.dtype)
        grid_thws = batch["grid_thws"].to(device=self.device)
        return pixel_values, grid_thws

    def encode(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Encode images one at a time and stack the merged features.

        The vendored eager attention builds a full ``(L, L)`` mask over the
        packed token sequence, so a multi-image batch would be quadratic in the
        total token count. One image per forward keeps the path within memory;
        the output is still ``(batch, tokens, 4096)``.
        """

        outputs = []
        for image in images:
            pixel_values, grid_thws = self.preprocess([image])
            with torch.no_grad():
                features = self.model(pixel_values, grid_thws)
            outputs.append(features[0].reshape(features[0].shape[0], -1))
        return torch.stack(outputs, dim=0)
