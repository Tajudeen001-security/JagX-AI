from __future__ import annotations

import torch
from model.config import ModelConfig
from model.transformer import JagXTransformer


def run_smoke_test() -> dict:
    cfg = ModelConfig(
        vocab_size=256,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        use_swiglu=True,
        use_rms_norm=True,
    )
    model = JagXTransformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(ids, labels=ids)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)
    out = model.generate(ids[:1, :4], max_new_tokens=4, temperature=1.0, top_k=8)
    assert out.shape == (1, 8)
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss": float(loss),
        "generated_tokens": int(out.shape[-1]),
    }


if __name__ == "__main__":
    print(run_smoke_test())
