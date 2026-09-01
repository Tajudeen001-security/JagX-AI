from __future__ import annotations

from pathlib import Path

import torch

from security.artifact_integrity import sha256_file
from .checkpoint import load_checkpoint as load_training_checkpoint
from .checkpoint import save_checkpoint as save_training_checkpoint


def save_checkpoint(model, optimizer, step: int, path: str, metadata: dict | None = None) -> dict:
    """Compatibility facade over the canonical atomic checkpoint writer."""
    save_training_checkpoint(path, model, optimizer, step=step, metadata=metadata)
    target = Path(path)
    return {"path": str(target), "sha256": sha256_file(str(target)), "step": step}


def load_checkpoint(path: str, device="cpu") -> dict:
    """Return the serialized checkpoint payload for legacy callers."""
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or "step" not in state or "model" not in state:
        raise ValueError("invalid JagX checkpoint: missing required fields")
    return state


__all__ = ["save_checkpoint", "load_checkpoint", "load_training_checkpoint"]
