from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from training.unified_multimodal import MultimodalBatch, UnifiedMultimodalModel


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0


def _filter_logits(logits: torch.Tensor, cfg: GenerationConfig) -> torch.Tensor:
    if cfg.temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = logits / cfg.temperature
    if cfg.top_k is not None:
        if cfg.top_k <= 0:
            raise ValueError("top_k must be positive")
        k = min(cfg.top_k, logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if cfg.top_p is not None:
        if not 0 < cfg.top_p <= 1:
            raise ValueError("top_p must be in (0,1]")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative > cfg.top_p
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    return logits


@torch.no_grad()
def generate_multimodal(
    model: UnifiedMultimodalModel,
    batch: MultimodalBatch,
    config: GenerationConfig | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    cfg = config or GenerationConfig()
    if cfg.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if cfg.repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    model.eval()
    ids = batch.input_ids.clone()
    labels = batch.labels
    for _ in range(cfg.max_new_tokens):
        current = MultimodalBatch(
            input_ids=ids,
            labels=labels[:, : ids.shape[1]] if labels is not None else ids,
            images=batch.images,
            audio=batch.audio,
            video=batch.video,
        )
        logits, _ = model(current)
        next_logits = logits[:, -1, :]
        if cfg.repetition_penalty != 1.0:
            for token in ids.unique().tolist():
                if next_logits[:, token].numel():
                    next_logits[:, token] = torch.where(
                        next_logits[:, token] < 0,
                        next_logits[:, token] * cfg.repetition_penalty,
                        next_logits[:, token] / cfg.repetition_penalty,
                    )
        filtered = _filter_logits(next_logits, cfg)
        if cfg.temperature == 0:
            next_token = filtered.argmax(dim=-1, keepdim=True)
        else:
            probs = torch.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_token], dim=1)
        if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
            break
    return ids
