from __future__ import annotations
import torch
import torch.nn.functional as F


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy for next-token prediction."""
    if logits.ndim != 3 or targets.ndim != 2:
        raise ValueError("logits must be [B,T,V] and targets must be [B,T]")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)


def perplexity(loss: torch.Tensor) -> float:
    return float(torch.exp(loss.detach().float()).cpu())
