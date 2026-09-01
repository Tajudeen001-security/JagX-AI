from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DPOConfig:
    beta: float = 0.1
    label_smoothing: float = 0.0
    reference_free: bool = False


def _logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Per-sequence mean log-prob of labels under logits (shifted causal LM style)."""
    # logits: (B, T, V), labels: (B, T)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = shift_labels.ne(ignore_index).float()
    # avoid div by zero
    denom = mask.sum(dim=-1).clamp(min=1.0)
    return (gathered * mask).sum(dim=-1) / denom


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: Optional[torch.Tensor],
    ref_rejected_logps: Optional[torch.Tensor],
    config: Optional[DPOConfig] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Standard DPO loss on precomputed sequence log-probs.

    If reference_free or ref_* is None, reference log-ratios are treated as zero.
    """
    config = config or DPOConfig()
    if config.reference_free or ref_chosen_logps is None or ref_rejected_logps is None:
        ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
        ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    if config.label_smoothing > 0:
        loss = (
            -F.logsigmoid(config.beta * logits) * (1 - config.label_smoothing)
            - F.logsigmoid(-config.beta * logits) * config.label_smoothing
        ).mean()
    else:
        loss = -F.logsigmoid(config.beta * logits).mean()

    with torch.no_grad():
        chosen_rewards = config.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = config.beta * (policy_rejected_logps - ref_rejected_logps)
        metrics = {
            "loss": float(loss.detach()),
            "chosen_reward": float(chosen_rewards.mean()),
            "rejected_reward": float(rejected_rewards.mean()),
            "reward_margin": float((chosen_rewards - rejected_rewards).mean()),
        }
    return loss, metrics


class DPOTrainerStep(nn.Module):
    """One DPO step helper: runs policy (and optional ref) forward to get logps.

    Expects batches with keys chosen_input_ids / rejected_input_ids (labels optional;
    defaults to input_ids).
    """

    def __init__(self, policy: nn.Module, reference: Optional[nn.Module] = None, config: Optional[DPOConfig] = None):
        super().__init__()
        self.policy = policy
        self.reference = reference
        self.config = config or DPOConfig()

    def _seq_logps(self, model: nn.Module, input_ids: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        labels = labels if labels is not None else input_ids
        out = model(input_ids, labels=None)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        return _logprobs_from_logits(logits, labels)

    def forward(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_labels: Optional[torch.Tensor] = None,
        rejected_labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        policy_chosen = self._seq_logps(self.policy, chosen_input_ids, chosen_labels)
        policy_rejected = self._seq_logps(self.policy, rejected_input_ids, rejected_labels)

        ref_chosen = ref_rejected = None
        if self.reference is not None and not self.config.reference_free:
            with torch.no_grad():
                ref_chosen = self._seq_logps(self.reference, chosen_input_ids, chosen_labels)
                ref_rejected = self._seq_logps(self.reference, rejected_input_ids, rejected_labels)

        return dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, self.config)
