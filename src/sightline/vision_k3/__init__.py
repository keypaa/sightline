"""Vendored Kimi-K3 vision code.

Source: moonshotai/Kimi-K3 at revision f831ab66814297da540d832a5235f8e904f29d06.
Only the MoonViT-V2 tower is vendored (modeling_k3_vision.py); the K3 language
model is out of scope and its module targets an older transformers API.
The K3 LICENSE file is kept alongside this code.
"""

from sightline.vision_k3.modeling_k3_vision import (
    MoonViT3dPretrainedModel,
    VisionTowerConfig,
)

__all__ = ["MoonViT3dPretrainedModel", "VisionTowerConfig"]
