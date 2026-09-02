from __future__ import annotations

import torch


def freeze_finished(token: torch.Tensor, finished: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """Keep completed rows at EOS while preserving rectangular batches."""
    if token.ndim != 2 or token.size(1) != 1:
        raise ValueError("token must have shape [B,1]")
    if finished.ndim != 1 or finished.size(0) != token.size(0):
        raise ValueError("finished must have shape [B]")
    if eos_token_id < 0:
        raise ValueError("eos_token_id must be non-negative")
    return torch.where(finished[:, None], torch.full_like(token, eos_token_id), token)
