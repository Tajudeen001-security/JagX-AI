from __future__ import annotations
import torch


def build_adamw(model, lr=3e-4, weight_decay=0.1):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or name.endswith("bias") or "norm" in name.lower() else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
