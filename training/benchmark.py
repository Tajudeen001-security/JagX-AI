from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from .evaluate import evaluate


def benchmark_model(
    model: torch.nn.Module,
    batches,
    *,
    steps: int = 20,
    device: str | None = None,
) -> dict[str, Any]:
    """Measure loss, throughput and peak CUDA memory on a deterministic batch sample."""
    if steps < 1:
        raise ValueError("steps must be positive")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(target).eval()
    iterator = iter(batches)
    for _ in range(2):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            batch = next(iterator)
        batch = {k: v.to(target) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.inference_mode():
            model(**batch)

    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
        torch.cuda.synchronize(target)
    start = time.perf_counter()
    token_count = 0
    last_loss = None
    with torch.inference_mode():
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                batch = next(iterator)
            batch = {k: v.to(target) if torch.is_tensor(v) else v for k, v in batch.items()}
            output = model(**batch)
            last_loss = output[1] if isinstance(output, (tuple, list)) else getattr(output, "loss", output)
            token_count += int(batch.get("input_ids", torch.empty(0)).numel())
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    elapsed = max(time.perf_counter() - start, 1e-9)
    result: dict[str, Any] = {
        "device": str(target),
        "steps": steps,
        "tokens": token_count,
        "tokens_per_second": token_count / elapsed,
        "step_time_ms": elapsed * 1000.0 / steps,
        "loss": float(last_loss.detach().cpu()) if torch.is_tensor(last_loss) else float(last_loss),
    }
    if target.type == "cuda":
        result["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(target) / (1024**2)
        result["gpu"] = torch.cuda.get_device_name(target)
    return result


def write_benchmark(result: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
