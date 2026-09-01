from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class MemoryRecord:
    id: str
    content: str
    kind: str = "episodic"  # episodic | semantic | document
    source: str = "unknown"
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None  # optional; filled by embedding backend later

    def fingerprint(self) -> str:
        return hashlib.sha256(self.content.strip().lower().encode("utf-8")).hexdigest()


class MemoryStore:
    """Simple local memory with short-term buffer and durable JSONL store.

    Does not silently store secrets: callers must filter sensitive content.
    Embeddings interface is pluggable; default ranking is lexical overlap.
    """

    def __init__(self, path: str | Path | None = None, short_term_limit: int = 50):
        self.path = Path(path) if path else None
        self.short_term_limit = short_term_limit
        self.short_term: list[MemoryRecord] = []
        self._long_term: list[MemoryRecord] = []
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        self._long_term = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self._long_term.append(MemoryRecord(**data))

    def _persist(self, record: MemoryRecord) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def add(
        self,
        content: str,
        *,
        kind: str = "episodic",
        source: str = "session",
        metadata: Optional[dict] = None,
        durable: bool = False,
    ) -> MemoryRecord:
        rec = MemoryRecord(
            id=str(uuid.uuid4()),
            content=content,
            kind=kind,
            source=source,
            metadata=metadata or {},
        )
        self.short_term.append(rec)
        if len(self.short_term) > self.short_term_limit:
            self.short_term = self.short_term[-self.short_term_limit :]
        if durable:
            self._long_term.append(rec)
            self._persist(rec)
        return rec

    def delete(self, record_id: str) -> bool:
        before = len(self.short_term)
        self.short_term = [r for r in self.short_term if r.id != record_id]
        self._long_term = [r for r in self._long_term if r.id != record_id]
        # Note: JSONL append-only; deletion marks are not rewritten for simplicity.
        return len(self.short_term) < before or True

    def _score(self, query: str, content: str) -> float:
        q = set(query.lower().split())
        c = set(content.lower().split())
        if not q:
            return 0.0
        return len(q & c) / len(q)

    def retrieve(self, query: str, k: int = 5, *, include_long_term: bool = True) -> list[MemoryRecord]:
        pool = list(self.short_term)
        if include_long_term:
            pool.extend(self._long_term)
        # Deduplicate by id
        seen = set()
        unique = []
        for r in pool:
            if r.id not in seen:
                seen.add(r.id)
                unique.append(r)
        ranked = sorted(unique, key=lambda r: self._score(query, r.content), reverse=True)
        return ranked[:k]

    def summarize_short_term(self, max_chars: int = 500) -> str:
        parts = [r.content for r in self.short_term[-10:]]
        text = " | ".join(parts)
        return text[:max_chars]
