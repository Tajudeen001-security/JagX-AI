from __future__ import annotations

import torch
from torch import nn


class PatchVisionEncoder(nn.Module):
    """Trainable lightweight image-to-token encoder.

    This is a real neural encoder for multimodal experiments, not a claim of
    frontier vision quality. It can be scaled by changing image_size,
    patch_size, width and depth while keeping the language interface stable.
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16, width: int = 384, depth: int = 6, heads: int = 6):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size)
        n = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, width))
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(width)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        x = self.patch(images).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        if x.size(1) != self.pos.size(1):
            raise ValueError("image resolution does not match configured image_size")
        return self.norm(self.encoder(x + self.pos))


class VisionProjector(nn.Module):
    """Projects vision tokens into the JagX language-model hidden space."""

    def __init__(self, vision_width: int, language_width: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(vision_width), nn.Linear(vision_width, language_width), nn.GELU(), nn.Linear(language_width, language_width))

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(vision_tokens)
