from __future__ import annotations
from pathlib import Path
import json
import torch
from torch.utils.data import IterableDataset


class JsonlTokenDataset(IterableDataset):
    """Streams token-id sequences from JSONL without loading the dataset into RAM."""

    def __init__(self, path: str, key: str = "input_ids"):
        self.path = Path(path)
        self.key = key

    def __iter__(self):
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                ids = row[self.key]
                x = torch.tensor(ids[:-1], dtype=torch.long)
                y = torch.tensor(ids[1:], dtype=torch.long)
                yield x, y
