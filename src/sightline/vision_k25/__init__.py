"""Vendored Kimi-K2.6 MoonViT vision code.

Source: moonshotai/Kimi-K2.6 at revision 7eb5002f6aadc958aed6a9177b7ed26bb94011bb.
Only the MoonViT tower is vendored; the K2.6 language model is out of scope.
The K2.6 LICENSE (Modified MIT) is kept alongside this code.
"""

from sightline.vision_k25.modeling_k25_vision import (
    MoonViT3dPretrainedModel,
    VisionTowerConfig,
)

__all__ = ["MoonViT3dPretrainedModel", "VisionTowerConfig"]
