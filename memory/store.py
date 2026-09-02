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
    importance: float = 0.5  # 0..1 ranking weight
    session_id: Optional[str] = None

    def fingerprint(self) -> str:
        return hashlib.sha256(self.content.strip().lower().encode("utf-8")).hexdigest()


class MemoryStore:
    """Local memory with short-term buffer and durable JSONL store.

    Features:
    - short-term ring buffer + optional durable JSONL
    - lexical retrieval ranking weighted by importance
    - fingerprint-based deduplication on add
    - optional session isolation for multi-tenant / multi-chat use

    Does not silently store secrets: callers must filter sensitive content.
    Embeddings interface is pluggable; default ranking is lexical overlap.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        short_term_limit: int = 50,
        *,
        default_session_id: Optional[str] = None,
    ):
        self.path = Path(path) if path else None
        self.short_term_limit = short_term_limit
        self.default_session_id = default_session_id
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
                # Backward compatible load when older records lack new fields
                data.setdefault("importance", 0.5)
                data.setdefault("session_id", None)
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
        importance: float = 0.5,
        session_id: Optional[str] = None,
        dedupe: bool = True,
    ) -> MemoryRecord:
        sid = session_id if session_id is not None else self.default_session_id
        importance = max(0.0, min(1.0, float(importance)))
        rec = MemoryRecord(
            id=str(uuid.uuid4()),
            content=content,
            kind=kind,
            source=source,
            metadata=metadata or {},
            importance=importance,
            session_id=sid,
        )
        if dedupe:
            fp = rec.fingerprint()
            # Drop prior short-term duplicates with same fingerprint (+ session)
            self.short_term = [
                r
                for r in self.short_term
                if not (r.fingerprint() == fp and r.session_id == sid)
            ]
        self.short_term.append(rec)
        if len(self.short_term) > self.short_term_limit:
            # Prefer keeping higher-importance items when trimming
            ranked = sorted(self.short_term, key=lambda r: (r.importance, r.created_at))
            self.short_term = ranked[-self.short_term_limit :]
        if durable:
            if dedupe:
                fp = rec.fingerprint()
                self._long_term = [
                    r
                    for r in self._long_term
                    if not (r.fingerprint() == fp and r.session_id == sid)
                ]
            self._long_term.append(rec)
            self._persist(rec)
        return rec

    def delete(self, record_id: str) -> bool:
        before_st = len(self.short_term)
        before_lt = len(self._long_term)
        self.short_term = [r for r in self.short_term if r.id != record_id]
        self._long_term = [r for r in self._long_term if r.id != record_id]
        # Note: JSONL append-only; physical line removal is not rewritten for simplicity.
        return len(self.short_term) < before_st or len(self._long_term) < before_lt

    def _score(self, query: str, content: str) -> float:
        q = set(query.lower().split())
        c = set(content.lower().split())
        if not q:
            return 0.0
        return len(q & c) / len(q)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        include_long_term: bool = True,
        session_id: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> list[MemoryRecord]:
        sid = session_id if session_id is not None else self.default_session_id
        pool = list(self.short_term)
        if include_long_term:
            pool.extend(self._long_term)
        # Deduplicate by id
        seen: set[str] = set()
        unique: list[MemoryRecord] = []
        for r in pool:
            if r.id in seen:
                continue
            if sid is not None and r.session_id is not None and r.session_id != sid:
                continue
            if r.importance < min_importance:
                continue
            seen.add(r.id)
            unique.append(r)

        def rank_key(r: MemoryRecord) -> float:
            return self._score(query, r.content) * (0.5 + 0.5 * r.importance)

        ranked = sorted(unique, key=rank_key, reverse=True)
        return ranked[:k]

    def summarize_short_term(self, max_chars: int = 500, *, session_id: Optional[str] = None) -> str:
        sid = session_id if session_id is not None else self.default_session_id
        items = self.short_term
        if sid is not None:
            items = [r for r in items if r.session_id in (None, sid)]
        parts = [r.content for r in items[-10:]]
        text = " | ".join(parts)
        return text[:max_chars]
