from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch

from tokenizer import JagXTokenizer
from training.pretraining import PretrainingConfig
from training.unified_multimodal import MultimodalBatch


def load_multimodal_jsonl(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("text"):
                raise ValueError(f"line {line_no} must contain text")
            if not any(row.get(key) for key in ("image", "audio", "video")):
                raise ValueError(f"line {line_no} must contain at least one modality")
            rows.append(row)
    if not rows:
        raise ValueError("multimodal dataset is empty")
    return rows


def _load_tensor(path: str | Path, key: str) -> torch.Tensor:
    value = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{key} tensor file must contain a torch.Tensor")
    return value.float()


def multimodal_batches(
    rows: list[dict], tokenizer: JagXTokenizer, cfg: PretrainingConfig
) -> Iterator[MultimodalBatch]:
    for start in range(0, len(rows), cfg.batch_size):
        chunk = rows[start : start + cfg.batch_size]
        if len(chunk) < cfg.batch_size and cfg.drop_remainder:
            break
        tokens = [tokenizer.encode(str(row["text"]), add_special_tokens=True)[: cfg.seq_len] for row in chunk]
        if any(not t for t in tokens):
            continue
        input_ids = torch.full((len(chunk), cfg.seq_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full_like(input_ids, -100)
        for i, t in enumerate(tokens):
            input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long)
            labels[i, : len(t)] = input_ids[i, : len(t)]

        def stack_optional(key: str):
            paths = [row.get(key) for row in chunk]
            if any(paths) and not all(paths):
                raise ValueError(f"mixed {key} presence is not supported within a batch")
            if not all(paths):
                return None
            tensors = [_load_tensor(path, key) for path in paths]
            if len({tuple(t.shape) for t in tensors}) != 1:
                raise ValueError(f"all {key} tensors must have identical shapes")
            return torch.stack(tensors)

        yield MultimodalBatch(
            input_ids=input_ids,
            labels=labels,
            images=stack_optional("image"),
            audio=stack_optional("audio"),
            video=stack_optional("video"),
        )
