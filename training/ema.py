from __future__ import annotations

from collections.abc import Iterable

import torch


class ExponentialMovingAverage:
    """Shadow weights for more stable evaluation/checkpoint selection."""

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be between 0 and 1")
        self.decay = decay
        self._parameter_refs = tuple(p for p in parameters if p.requires_grad)
        self.shadow = {id(p): p.detach().clone() for p in self._parameter_refs}

    @torch.no_grad()
    def update(self, model_or_params):
        """Accept either a module or an iterable of parameters."""
        if isinstance(model_or_params, torch.nn.Module):
            parameters = model_or_params.parameters()
        else:
            parameters = model_or_params
        for p in parameters:
            if p.requires_grad and id(p) in self.shadow:
                self.shadow[id(p)].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model_or_params):
        if isinstance(model_or_params, torch.nn.Module):
            parameters = model_or_params.parameters()
        else:
            parameters = model_or_params
        for p in parameters:
            if p.requires_grad and id(p) in self.shadow:
                p.copy_(self.shadow[id(p)])

    def state_dict(self) -> dict:
        """Return an ID-independent representation suitable for checkpoints."""
        return {
            "decay": self.decay,
            "shadow": [self.shadow[id(p)].detach().cpu().clone() for p in self._parameter_refs],
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        """Restore EMA shadows onto the current model's parameter ordering."""
        if not isinstance(state, dict) or "shadow" not in state:
            raise ValueError("invalid EMA state: missing shadow weights")
        shadows = list(state["shadow"])
        if len(shadows) != len(self._parameter_refs):
            raise ValueError("EMA state parameter count does not match the model")
        if "decay" in state:
            decay = float(state["decay"])
            if not 0.0 < decay < 1.0:
                raise ValueError("invalid EMA decay")
            self.decay = decay
        self.shadow = {
            id(p): shadow.detach().to(device=p.device, dtype=p.dtype).clone()
            for p, shadow in zip(self._parameter_refs, shadows)
        }


# Backward-compatible alias used by CausalLMTrainer
EMA = ExponentialMovingAverage
