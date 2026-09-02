from __future__ import annotations

import torch

from media.generation import GenerationConfig, _filter_logits
from training.unified_multimodal import MultimodalBatch, UnifiedMultimodalModel


@torch.no_grad()
def generate_cached(
    model: UnifiedMultimodalModel,
    batch: MultimodalBatch,
    config: GenerationConfig | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate conditioned text while reusing Transformer KV state."""
    cfg = config or GenerationConfig()
    if cfg.max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if cfg.repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")

    model.eval()
    lm = model.language_model
    text = lm.token_embedding(batch.input_ids)
    prefixes = model._prefixes(batch)
    embeddings = torch.cat([*prefixes, text], dim=1) if prefixes else text
    if embeddings.shape[1] > lm.cfg.max_seq_len:
        raise ValueError("conditioning plus prompt exceeds max_seq_len")

    cos, sin = lm.rope(embeddings.shape[1])
    x = embeddings
    past = []
    for block in lm.blocks:
        x, present = block(x, cos.to(x.device), sin.to(x.device), use_cache=True)
        past.append(present)

    generated = batch.input_ids.clone()
    logits = lm.lm_head(lm.norm(x))[:, -1, :]
    for _ in range(cfg.max_new_tokens):
        next_logits = logits / cfg.temperature
        if cfg.repetition_penalty != 1.0:
            for b in range(generated.size(0)):
                for token_id in set(generated[b].tolist()):
                    value = next_logits[b, token_id]
                    next_logits[b, token_id] = value / cfg.repetition_penalty if value > 0 else value * cfg.repetition_penalty
        filtered = _filter_logits(next_logits, GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=1.0,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            repetition_penalty=1.0,
        ))
        next_token = filtered.argmax(dim=-1, keepdim=True) if cfg.temperature <= 0 else torch.multinomial(torch.softmax(filtered, -1), 1)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
            break
        x = lm.token_embedding(next_token)
        pos = sum(k.shape[2] for k, _ in [past[0]]) if False else past[0][0].shape[2]
        cos, sin = lm.rope(pos + 1)
        cos, sin = cos[:, :, -1:, :], sin[:, :, -1:, :]
        new_past = []
        for i, block in enumerate(lm.blocks):
            x, present = block(x, cos.to(x.device), sin.to(x.device), past_kv=past[i], use_cache=True)
            new_past.append(present)
        past = new_past
        logits = lm.lm_head(lm.norm(x))[:, -1, :]
    return generated
