from __future__ import annotations

import json
from pathlib import Path
import torch
from torch.utils.data import IterableDataset


class PackedTokenDataset(IterableDataset):
    def __init__(self, path: str, seq_len: int):
        self.path = Path(path)
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                buffer.extend(row["tokens"])
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[: self.seq_len + 1]
                    del buffer[: self.seq_len]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
