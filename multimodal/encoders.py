from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import nn

from media.multimodal_contract import MediaTensor, Modality


class ModalityEncoder(ABC, nn.Module):
    """Provider-independent modality encoder interface.

    Concrete encoders may be trained later. This defines the contract used by
    training and inference so JagX does not depend on external multimodal APIs.
    """

    modality: Modality

    @abstractmethod
    def forward(self, batch: MediaTensor) -> torch.Tensor:
        """Return (B, T, D) features compatible with JagX hidden size."""


class TextEncoder(ModalityEncoder):
    """Identity-style text path: expects token embedding inputs already formed."""

    modality = Modality.TEXT

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, batch: MediaTensor) -> torch.Tensor:
        x = batch.values
        if not torch.is_tensor(x):
            raise TypeError("text encoder expects tensor values")
        if x.dim() != 3 or x.size(-1) != self.d_model:
            raise ValueError(f"expected (B,T,{self.d_model}) got {tuple(x.shape)}")
        return x


class ImagePatchEncoder(ModalityEncoder):
    """Minimal patch embedder for smoke tests (not a trained vision model)."""

    modality = Modality.IMAGE

    def __init__(self, d_model: int, patch_dim: int = 48):
        super().__init__()
        self.proj = nn.Linear(patch_dim, d_model)

    def forward(self, batch: MediaTensor) -> torch.Tensor:
        x = batch.values  # expected (B, N_patches, patch_dim)
        if not torch.is_tensor(x) or x.dim() != 3:
            raise ValueError("image encoder expects (B, N, patch_dim) tensor")
        return self.proj(x)


class AudioFrameEncoder(ModalityEncoder):
    """Minimal frame embedder for smoke tests."""

    modality = Modality.AUDIO

    def __init__(self, d_model: int, frame_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(frame_dim, d_model)

    def forward(self, batch: MediaTensor) -> torch.Tensor:
        x = batch.values  # (B, T, frame_dim)
        if not torch.is_tensor(x) or x.dim() != 3:
            raise ValueError("audio encoder expects (B, T, frame_dim) tensor")
        return self.proj(x)


class VideoFrameEncoder(ModalityEncoder):
    """Treat video as sequence of image-like patches for interface testing."""

    modality = Modality.VIDEO

    def __init__(self, d_model: int, frame_dim: int = 48):
        super().__init__()
        self.proj = nn.Linear(frame_dim, d_model)

    def forward(self, batch: MediaTensor) -> torch.Tensor:
        x = batch.values  # (B, T, frame_dim)
        if not torch.is_tensor(x) or x.dim() != 3:
            raise ValueError("video encoder expects (B, T, frame_dim) tensor")
        return self.proj(x)


class MultimodalProjector(nn.Module):
    """Project modality features into shared JagX space."""

    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))
