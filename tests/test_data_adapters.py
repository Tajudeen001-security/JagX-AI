import json
import tempfile
from pathlib import Path

from data.adapters import load_jsonl_records, prepare_corpus


def test_prepare_corpus_jsonl_and_text():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        jsonl = root / "a.jsonl"
        jsonl.write_text(json.dumps({"text": "hello world from jsonl example"}) + "\n", encoding="utf-8")
        txt = root / "b.txt"
        txt.write_text("another paragraph about jagx training data\n\nsecond block here", encoding="utf-8")
        out = root / "out.jsonl"
        stats = prepare_corpus([jsonl, txt], out, min_chars=10)
        assert stats["output_records"] >= 1
        records = list(load_jsonl_records(out))
        assert all(r.text for r in records)
