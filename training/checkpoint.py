from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 2


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler=None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
    ema=None,
) -> None:
    """Atomically persist all state required to resume training."""
    if step < 0:
        raise ValueError("step must be non-negative")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema.state_dict() if ema is not None and hasattr(ema, "state_dict") else None,
        "metadata": dict(metadata or {}),
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    ema=None,
) -> tuple[int, dict[str, Any]]:
    """Restore model/training state and return ``(step, metadata)``."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state, dict) or not {"step", "model"}.issubset(state):
        raise ValueError("invalid JagX checkpoint: missing required fields")
    version = int(state.get("format_version", 1))
    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported JagX checkpoint format {version}; "
            f"maximum supported format is {CHECKPOINT_FORMAT_VERSION}"
        )
    step = int(state["step"])
    if step < 0:
        raise ValueError("invalid JagX checkpoint: step must be non-negative")
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if ema is not None and state.get("ema") is not None and hasattr(ema, "load_state_dict"):
        ema.load_state_dict(state["ema"])
    return step, dict(state.get("metadata") or {})
