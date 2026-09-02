from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ModalityCodec(ABC):
    """Interface for converting raw media tensors to trainable latent/token spaces."""

    @abstractmethod
    def encode(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IdentityCodec(ModalityCodec):
    """Lossless codec useful for native experiments and tests."""

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("value must be a torch.Tensor")
        return value

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if not isinstance(latent, torch.Tensor):
            raise TypeError("latent must be a torch.Tensor")
        return latent
