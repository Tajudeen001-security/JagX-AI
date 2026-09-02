from __future__ import annotations

from typing import Callable

import torch


def apply_eos_freeze(token: torch.Tensor, finished: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Keep already-finished batch rows at EOS."""
    if token.ndim != 2 or token.size(1) != 1:
        raise ValueError("token must have shape [B,1]")
    if finished.ndim != 1 or finished.size(0) != token.size(0):
        raise ValueError("finished must have shape [B]")
    return torch.where(finished.unsqueeze(1), torch.full_like(token, eos_token_id), token)


def update_finished(token: torch.Tensor, finished: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Return the updated EOS-completion mask."""
    return finished | token[:, 0].eq(eos_token_id)
