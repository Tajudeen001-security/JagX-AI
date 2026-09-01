from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DPOConfig:
    beta: float = 0.1
    label_pad_id: int = -100


def _logprob_sum(logits: torch.Tensor, labels: torch.Tensor, pad_id: int = -100) -> torch.Tensor:
    """Sum of log-probs of label tokens (shifted causal LM style)."""
    # logits: [B, T, V], labels: [B, T]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = shift_labels.ne(pad_id).float()
    return (gathered * mask).sum(dim=-1)  # [B]


def dpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    config: Optional[DPOConfig] = None,
) -> torch.Tensor:
    """Direct Preference Optimization loss (Rafailov et al.).

    All logits are from causal LMs over the respective sequences.
    Reference model should be frozen. This is the training objective scaffold;
    it does not imply a preference dataset has been collected.
    """
    config = config or DPOConfig()
    pi_c = _logprob_sum(policy_chosen_logits, chosen_labels, config.label_pad_id)
    pi_r = _logprob_sum(policy_rejected_logits, rejected_labels, config.label_pad_id)
    ref_c = _logprob_sum(ref_chosen_logits, chosen_labels, config.label_pad_id)
    ref_r = _logprob_sum(ref_rejected_logits, rejected_labels, config.label_pad_id)

    logits = config.beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss = -F.logsigmoid(logits).mean()
    return loss
