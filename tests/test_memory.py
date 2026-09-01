from __future__ import annotations

import tempfile
from pathlib import Path

from memory import MemoryStore


def test_short_term_and_retrieve():
    store = MemoryStore(short_term_limit=10)
    store.add("JagX trains local models", kind="semantic", source="doc")
    store.add("user asked about RoPE", kind="episodic")
    hits = store.retrieve("RoPE models", k=2)
    assert len(hits) >= 1
    assert any("RoPE" in h.content or "models" in h.content for h in hits)


def test_durable_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "mem.jsonl"
        store = MemoryStore(path)
        store.add("persistent fact about tokenizer", durable=True, kind="semantic")
        store2 = MemoryStore(path)
        hits = store2.retrieve("tokenizer", k=1, include_long_term=True)
        assert hits and "tokenizer" in hits[0].content
