"""MoonViT-V2 (K3) encoder wrapper.

Loads the K3 vision tower from the pinned configuration and the extracted
tower weights, preprocesses images with the official K3 processor, and returns
merged patch features of shape ``(batch, tokens, 4096)``.

The token count is derived from ``grid_thws`` and the merge kernel size; it is
never hard-coded. The tower is frozen and runs under ``torch.no_grad()``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from sightline.vision_encoder import MoonViTEncoder
from sightline.vision_k3 import MoonViT3dPretrainedModel, VisionTowerConfig
from sightline.vision_k3.configuration_kimi_k3 import KimiK3Config
from sightline.vision_k3.kimi_k3_vision_processing import KimiK3VisionProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_K3_DIR = Path(__file__).resolve().parent / "vision_k3"
DEFAULT_WEIGHTS = REPO_ROOT / "artifacts" / "k3_vision_f831ab66.safetensors"
ALLOWED_ATTENTION = ("eager", "flash_attention_2")


class K3VisionEncoder(MoonViTEncoder):
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

        preprocessor = json.loads((k3_dir / "preprocessor_config.json").read_text())
        processor = KimiK3VisionProcessor.from_dict(preprocessor)
        super().__init__(
            model,
            processor,
            vt_config.merge_kernel_size,
            vt_config.patch_size,
            device=device,
            dtype=dtype,
        )
