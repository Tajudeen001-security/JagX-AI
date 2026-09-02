from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    """Linear beta schedule used by the native media generators."""

    steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self) -> None:
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("beta_start and beta_end must satisfy 0 < start < end < 1")

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        betas = torch.linspace(self.beta_start, self.beta_end, self.steps, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars


def _extract(values: torch.Tensor, timestep: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.gather(0, timestep.long()).reshape(-1, *([1] * (ndim - 1)))


@torch.no_grad()
def ddpm_sample(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shape: tuple[int, ...],
    *,
    schedule: DiffusionSchedule | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a latent tensor with the native epsilon-prediction backbone."""
    schedule = schedule or DiffusionSchedule()
    device = torch.device(device)
    betas, alphas, alpha_bars = schedule.tensors(device)
    x = torch.randn(shape, device=device, generator=generator)
    batch = shape[0]
    for step in range(schedule.steps - 1, -1, -1):
        timestep = torch.full((batch,), step, device=device, dtype=torch.long)
        predicted_noise = model(x, timestep)
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        mean = (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        if step:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = mean + torch.sqrt(betas[step]) * noise
        else:
            x = mean
    return x


@torch.no_grad()
def ddim_sample(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shape: tuple[int, ...],
    *,
    schedule: DiffusionSchedule | None = None,
    inference_steps: int = 50,
    eta: float = 0.0,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Fast DDIM sampler; ``eta=0`` is deterministic for fixed initial noise."""
    schedule = schedule or DiffusionSchedule()
    if inference_steps < 2 or inference_steps > schedule.steps:
        raise ValueError("inference_steps must be between 2 and schedule.steps")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    device = torch.device(device)
    _, _, alpha_bars = schedule.tensors(device)
    indices = torch.linspace(schedule.steps - 1, 0, inference_steps, device=device).round().long()
    indices = torch.unique_consecutive(indices)
    x = torch.randn(shape, device=device, generator=generator)
    batch = shape[0]
    for i, step in enumerate(indices):
        timestep = torch.full((batch,), int(step), device=device, dtype=torch.long)
        eps = model(x, timestep)
        a_bar = alpha_bars[step]
        pred_x0 = (x - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar)
        if i == len(indices) - 1:
            x = pred_x0
            break
        next_step = indices[i + 1]
        next_bar = alpha_bars[next_step]
        sigma = eta * torch.sqrt((1 - next_bar) / (1 - a_bar)) * torch.sqrt(1 - a_bar / next_bar)
        direction = torch.sqrt((1 - next_bar - sigma.square()).clamp_min(0)) * eps
        if eta:
            noise = torch.randn(x.shape, device=device, generator=generator)
        else:
            noise = torch.zeros_like(x)
        x = torch.sqrt(next_bar) * pred_x0 + direction + sigma * noise
    return x
