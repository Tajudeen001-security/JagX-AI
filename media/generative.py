from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GenerationShape:
    channels: int
    height: int
    width: int
    frames: int = 1


class LatentDiffusionUNet(nn.Module):
    """Compact native diffusion backbone for image/video latent generation.

    This is an actual trainable PyTorch network; quality depends on trained
    JagX weights and a real media dataset.
    """

    def __init__(self, channels: int = 4, hidden: int = 64):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.in_proj = nn.Conv3d(channels, hidden, 3, padding=1)
        self.mid = nn.Sequential(nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv3d(hidden, hidden, 3, padding=1))
        self.out_proj = nn.Conv3d(hidden, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_4d = x.ndim == 4
        if original_4d:
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise ValueError("latent must have shape [B,C,H,W] or [B,C,T,H,W]")
        t = self.time(timestep.float().reshape(-1, 1)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = self.in_proj(x) + t
        h = h + self.mid(h)
        output = self.out_proj(h)
        return output.squeeze(2) if original_4d else output


class ImageGenerator(nn.Module):
    """Native latent image denoiser used by the image diffusion trainer."""

    def __init__(self, channels: int = 4, hidden: int = 64):
        super().__init__()
        self.backbone = LatentDiffusionUNet(channels, hidden)

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.backbone(latents, timestep)


class AudioGenerator(nn.Module):
    """Native causal waveform-token generator backbone."""

    def __init__(self, vocab_size: int = 1024, hidden: int = 256, layers: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(hidden, 8, hidden * 4, batch_first=True) for _ in range(layers)
        )
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        length = tokens.shape[1]
        causal_mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask)
        return self.lm_head(x)


class VideoGenerator(nn.Module):
    """Native spatiotemporal diffusion backbone preserving frame dimension."""

    def __init__(self, channels: int = 4, hidden: int = 64):
        super().__init__()
        self.backbone = LatentDiffusionUNet(channels, hidden)

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.backbone(latents, timestep)


def diffusion_loss(model: nn.Module, clean: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    """Standard epsilon-prediction objective for native image/video training."""
    alpha = 1.0 - timestep.float().reshape(-1, 1, 1, 1, 1).clamp(0, 1)
    if clean.ndim == 4:
        alpha = alpha.squeeze(2)
    noisy = clean * alpha.sqrt() + noise * (1.0 - alpha).sqrt()
    predicted = model(noisy, timestep)
    return torch.mean((predicted - noise) ** 2)
