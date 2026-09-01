from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class TextRecord:
    text: str
    source: str = "unknown"
    license: str = "unknown"
    domain: str = "general"
    language: str = "unknown"
    quality_score: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        normalized = re.sub(r"\s+", " ", self.text.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def filter_record(
    record: TextRecord,
    min_chars: int = 20,
    max_chars: int = 1_000_000,
) -> bool:
    """Quality gate for bulk corpus preparation.

    Default min_chars=20 is intentional for large-scale prepare pipelines.
    Callers that need short valid examples (e.g. unit tests, micro-corpora)
    should pass a lower min_chars or use is_viable_text().
    """
    text = normalize(record.text)
    if len(text) < min_chars or len(text) > max_chars:
        return False
    if not any(ch.isalnum() for ch in text):
        return False
    if record.quality_score < 0.1:
        return False
    return True


def is_viable_text(text: str, *, min_chars: int = 2) -> bool:
    """Structural viability only: drop empty/near-empty and non-alnum noise.

    Keeps short but real examples such as "hello world" while rejecting "x".
    """
    text = normalize(text)
    if len(text) < min_chars:
        return False
    if not any(ch.isalnum() for ch in text):
        return False
    # Single-character alnum is almost always noise for training text.
    if len(text) == 1:
        return False
    return True


def deduplicate(records: list[TextRecord]) -> list[TextRecord]:
    """Normalize whitespace, drop non-viable noise, and remove near-exact duplicates.

    Does not apply the bulk quality min_chars=20 gate so valid short examples
    are preserved. Use filter_record explicitly when preparing large corpora.
    """
    seen: set[str] = set()
    result: list[TextRecord] = []
    for r in records:
        r.text = normalize(r.text)
        if not is_viable_text(r.text):
            continue
        if r.quality_score < 0.1:
            continue
        fp = r.fingerprint
        if fp in seen:
            continue
        seen.add(fp)
        result.append(r)
    return result


def write_jsonl(records: list[TextRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[TextRecord]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield TextRecord(
                text=data.get("text", ""),
                source=data.get("source", "unknown"),
                license=data.get("license", "unknown"),
                domain=data.get("domain", "general"),
                language=data.get("language", "unknown"),
                quality_score=float(data.get("quality_score", 1.0)),
                metadata=data.get("metadata") or {},
            )


def split_records(
    records: list[TextRecord],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> dict[str, list[TextRecord]]:
    """Deterministic train/val/test split."""
    import random

    rng = random.Random(seed)
    idxs = list(range(len(records)))
    rng.shuffle(idxs)
    n = len(idxs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train : n_train + n_val]
    test_idx = idxs[n_train + n_val :]
    return {
        "train": [records[i] for i in train_idx],
        "val": [records[i] for i in val_idx],
        "test": [records[i] for i in test_idx],
    }
