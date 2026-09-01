from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    stop_token_ids: Optional[list[int]] = None


@torch.no_grad()
def generate(model, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
    """Thin wrapper that prefers the model's native generate when available."""
    if hasattr(model, "generate") and callable(model.generate):
        return model.generate(
            input_ids,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            stop_token_ids=config.stop_token_ids,
        )

    # Fallback for models that only expose forward
    model.eval()
    ids = input_ids
    stop_set = set(config.stop_token_ids or [])
    for _ in range(config.max_new_tokens):
        context = ids[:, -getattr(model, "cfg", type("C", (), {"max_seq_len": 2048})).max_seq_len :]
        out = model(context)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits[:, -1, :] / max(config.temperature, 1e-5)

        if config.repetition_penalty != 1.0:
            for token in ids[0].unique():
                value = logits[0, int(token)]
                logits[0, int(token)] = (
                    value / config.repetition_penalty if value > 0 else value * config.repetition_penalty
                )

        if config.top_k:
            v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = float("-inf")

        if 0.0 < config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative > config.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            indices_to_remove = mask.scatter(1, sorted_indices, mask)
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
        if stop_set and int(next_id[0, 0]) in stop_set:
            break
    return ids
