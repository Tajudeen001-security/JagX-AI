from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from training.unified_multimodal import MultimodalBatch, UnifiedMultimodalModel


@dataclass(frozen=True)
class CachedGenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    repetition_penalty: float = 1.0


def _next_token(logits: torch.Tensor, generated: torch.Tensor, cfg: CachedGenerationConfig) -> torch.Tensor:
    if cfg.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if cfg.repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    scores = logits.clone()
    if cfg.repetition_penalty != 1.0:
        for row in range(generated.size(0)):
            for token_id in set(generated[row].tolist()):
                value = scores[row, token_id]
                scores[row, token_id] = value / cfg.repetition_penalty if value > 0 else value * cfg.repetition_penalty
    if cfg.temperature == 0:
        return scores.argmax(dim=-1, keepdim=True)
    scores = scores / cfg.temperature
    if cfg.top_k is not None:
        if cfg.top_k <= 0:
            raise ValueError("top_k must be positive")
        values = torch.topk(scores, min(cfg.top_k, scores.size(-1)), dim=-1).values
        scores = scores.masked_fill(scores < values[:, [-1]], float("-inf"))
    if cfg.top_p is not None:
        if not 0 < cfg.top_p <= 1:
            raise ValueError("top_p must be in (0,1]")
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_scores, dim=-1), dim=-1)
        remove = cumulative > cfg.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, sorted_indices, sorted_scores)
    return torch.multinomial(F.softmax(scores, dim=-1), 1)


@torch.no_grad()
def generate_multimodal_cached(
    model: UnifiedMultimodalModel,
    batch: MultimodalBatch,
    config: CachedGenerationConfig | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate text from any supported modality using one prefix-cache pass."""
    cfg = config or CachedGenerationConfig()
    if cfg.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    model.eval()
    device = next(model.parameters()).device
    ids = batch.input_ids.to(device)
    if ids.ndim != 2 or ids.size(1) == 0:
        raise ValueError("input_ids must have shape [B,T] with T > 0")
    if cfg.max_new_tokens == 0:
        return ids.clone()

    routed = MultimodalBatch(
        input_ids=ids,
        labels=None,
        images=None if batch.images is None else batch.images.to(device),
        audio=None if batch.audio is None else batch.audio.to(device),
        video=None if batch.video is None else batch.video.to(device),
    )
    prefixes = model._prefixes(routed)
    text = model.language_model.token_embedding(ids)
    embeddings = torch.cat([*prefixes, text], dim=1) if prefixes else text
    if embeddings.size(1) + cfg.max_new_tokens > model.language_model.cfg.max_seq_len:
        raise ValueError("prompt and modality prefixes leave insufficient max_seq_len")

    logits, _, past = model._forward_cached_embeddings(embeddings)
    generated = ids.clone()
    finished = torch.zeros(ids.size(0), dtype=torch.bool, device=device)
    for _ in range(cfg.max_new_tokens):
        token = _next_token(logits[:, -1, :], generated, cfg)
        generated = torch.cat([generated, token], dim=1)
        if eos_token_id is not None:
            finished |= token[:, 0].eq(eos_token_id)
            if bool(finished.all()):
                break
        logits, _, past = model.language_model(token, past_key_values=past, use_cache=True)
    return generated
