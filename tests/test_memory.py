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


def test_importance_affects_ranking():
    store = MemoryStore()
    store.add("alpha beta gamma", importance=0.1)
    store.add("alpha beta gamma important", importance=0.95)
    hits = store.retrieve("alpha beta", k=2)
    assert hits[0].importance >= hits[-1].importance


def test_fingerprint_dedupe():
    store = MemoryStore()
    a = store.add("same content twice")
    b = store.add("same content twice")
    assert a.fingerprint() == b.fingerprint()
    # Only one short-term entry remains for that fingerprint
    assert sum(1 for r in store.short_term if r.fingerprint() == a.fingerprint()) == 1


def test_session_isolation():
    store = MemoryStore()
    store.add("session A secret", session_id="A", importance=1.0)
    store.add("session B public", session_id="B", importance=1.0)
    hits_a = store.retrieve("session", k=5, session_id="A")
    assert all(h.session_id in (None, "A") for h in hits_a)
    assert any("A secret" in h.content for h in hits_a)
    assert not any("B public" in h.content for h in hits_a)
