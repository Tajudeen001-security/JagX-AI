from __future__ import annotations

from typing import Iterator, Optional

import torch
from torch.nn import functional as F

from runtime.generation import GenerationConfig


@torch.no_grad()
def stream_generate(
    model,
    input_ids: torch.Tensor,
    config: Optional[GenerationConfig] = None,
) -> Iterator[torch.Tensor]:
    """Yield newly generated token ids one step at a time (local, no external API)."""
    config = config or GenerationConfig()
    model.eval()
    ids = input_ids
    stop_set = set(config.stop_token_ids or [])
    past = None
    use_cache = hasattr(model, "forward")

    for _ in range(config.max_new_tokens):
        if past is not None and use_cache:
            model_input = ids[:, -1:]
        else:
            max_len = getattr(getattr(model, "cfg", None), "max_seq_len", 2048)
            model_input = ids[:, -max_len:]

        if use_cache:
            try:
                out = model(model_input, past_key_values=past, use_cache=True)
                if isinstance(out, tuple) and len(out) == 3:
                    logits, _, past = out
                else:
                    logits = out[0] if isinstance(out, tuple) else out
                    past = None
            except TypeError:
                out = model(model_input)
                logits = out[0] if isinstance(out, tuple) else out
                past = None
        else:
            out = model(model_input)
            logits = out[0] if isinstance(out, tuple) else out

        logits = logits[:, -1, :] / max(config.temperature, 1e-5)

        if config.top_k:
            v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

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
        yield next_id

        if stop_set and int(next_id[0, 0]) in stop_set:
            break
