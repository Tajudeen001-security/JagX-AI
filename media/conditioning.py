from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ModalityProjector(nn.Module):
    """Project arbitrary modality features into the JagX language space."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        hidden = hidden_dim or max(input_dim, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2:
            raise ValueError("features must have at least 2 dimensions")
        return self.net(features)


class ModalityRouter(nn.Module):
    """Fuse text and optional modality sequences with learned modality gates."""

    def __init__(self, d_model: int, modalities: int = 3):
        super().__init__()
        if d_model <= 0 or modalities <= 0:
            raise ValueError("d_model and modalities must be positive")
        self.gates = nn.Parameter(torch.zeros(modalities))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, text: torch.Tensor, *modalities: Optional[torch.Tensor]) -> torch.Tensor:
        if text.ndim != 3:
            raise ValueError("text must have shape [B,T,D]")
        active = [m for m in modalities if m is not None]
        if not active:
            return text
        for m in active:
            if m.ndim != 3 or m.shape[0] != text.shape[0] or m.shape[2] != text.shape[2]:
                raise ValueError("modality tensors must have shape [B,T,D] matching text batch and width")
        fused = [text]
        for i, modality in enumerate(modalities):
            if modality is not None:
                gate = torch.sigmoid(self.gates[min(i, self.gates.numel() - 1)])
                fused.append(modality * gate)
        return self.norm(torch.cat(fused, dim=1))
