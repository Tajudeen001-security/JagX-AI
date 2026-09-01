from __future__ import annotations

from pathlib import Path
from typing import Any
import torch


def save_checkpoint(path: str, model, optimizer, scheduler=None, step: int = 0, metadata: dict[str, Any] | None = None):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metadata": metadata or {},
    }
    torch.save(state, target)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location="cpu") -> int:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return int(state.get("step", 0))
