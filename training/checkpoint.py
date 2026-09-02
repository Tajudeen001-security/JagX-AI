from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 2


def _model_config(model: Any) -> dict[str, Any] | None:
    """Return a serializable model config when the model exposes one."""
    config = getattr(model, "cfg", None)
    if config is None:
        config = getattr(model, "config", None)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return dict(value)
    return None


def _numpy_rng_state() -> dict[str, Any] | None:
    """Encode NumPy RNG state using weights-only-safe primitives."""
    try:
        import numpy as np
    except ImportError:
        return None
    bit_generator, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": str(bit_generator),
        "keys": torch.as_tensor(keys, dtype=torch.uint32),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _rng_state() -> dict[str, Any]:
    """Capture process RNG streams needed for deterministic continuation."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    numpy_state = _numpy_rng_state()
    if numpy_state is not None:
        state["numpy"] = numpy_state
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Any) -> None:
    """Restore RNG streams when present; legacy checkpoints remain supported."""
    if not isinstance(state, dict):
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    numpy_state = state.get("numpy")
    if isinstance(numpy_state, dict):
        try:
            import numpy as np
            keys = numpy_state["keys"]
            if torch.is_tensor(keys):
                keys = keys.detach().cpu().numpy()
            np.random.set_state(
                (
                    str(numpy_state["bit_generator"]),
                    keys,
                    int(numpy_state["pos"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
        except (ImportError, KeyError, TypeError, ValueError):
            pass
    elif numpy_state is not None:
        # Legacy v2 checkpoints stored NumPy's raw tuple; keep compatibility.
        try:
            import numpy as np
            np.random.set_state(numpy_state)
        except (ImportError, TypeError, ValueError):
            pass
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler=None,
    step: int = 0,
    metadata: dict[str, Any] | None = None,
    ema=None,
) -> None:
    """Atomically persist all state required to resume training.

    Current checkpoints include the model configuration and RNG state so
    training can continue reproducibly without external configuration files.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata_value = dict(metadata or {})
    config = _model_config(model)
    if config is not None and "model_config" not in metadata_value:
        metadata_value["model_config"] = config
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": int(step),
        "model": model.state_dict(),
        "config": config,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema.state_dict() if ema is not None and hasattr(ema, "state_dict") else None,
        "rng_state": _rng_state(),
        "metadata": metadata_value,
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
    """Restore model/training/RNG state and return ``(step, metadata)``."""
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
    _restore_rng_state(state.get("rng_state"))
    return step, dict(state.get("metadata") or {})
