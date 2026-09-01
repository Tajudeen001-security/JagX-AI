from __future__ import annotations

from typing import Iterator

import torch

from .packing import pack_tokens


def sequences_to_tensors(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert packed token sequences into (input_ids, labels) for causal LM."""
    if not sequences:
        raise ValueError("no sequences")
    ids = torch.tensor(sequences, dtype=torch.long)
    # labels identical; model shifts internally for loss
    return ids, ids.clone()


def iter_token_batches(
    tokens: list[int],
    seq_len: int,
    batch_size: int,
    *,
    drop_remainder: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    """Pack a flat token stream and yield model-ready batches."""
    sequences = pack_tokens(tokens, seq_len=seq_len, drop_remainder=drop_remainder)
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i : i + batch_size]
        if len(chunk) < batch_size and drop_remainder:
            break
        input_ids, labels = sequences_to_tensors(chunk)
        yield {"input_ids": input_ids, "labels": labels}
