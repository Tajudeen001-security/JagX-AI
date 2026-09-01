from __future__ import annotations
from pathlib import Path
from typing import Any
import torch


def save_checkpoint(path: str, model, optimizer, scheduler=None, step: int = 0, metadata: dict[str, Any] | None = None):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "metadata": metadata or {},
        },
        target,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location="cpu") -> int:
    state = torch.load(path, map_location=map_location, weights_only=True)
    required = {"step", "model"}
    if not required.issubset(state):
        raise ValueError("invalid JagX checkpoint: missing required fields")
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    return int(state["step"])
