from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DPOConfig:
    beta: float = 0.1
    label_smoothing: float = 0.0


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean log-probability of non-ignored target tokens."""
    logp = F.log_softmax(logits[:, :-1], dim=-1)
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    safe = targets.masked_fill(~valid, 0)
    token_lp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return (token_lp * valid).sum(-1) / valid.sum(-1).clamp_min(1)


def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             reference_chosen: torch.Tensor, reference_rejected: torch.Tensor,
             config: DPOConfig = DPOConfig()) -> torch.Tensor:
    """Direct Preference Optimization loss; reference scores are detached."""
    margin = (policy_chosen - policy_rejected) - (
        reference_chosen.detach() - reference_rejected.detach()
    )
    targets = torch.ones_like(margin)
    return F.binary_cross_entropy_with_logits(config.beta * margin, targets)
