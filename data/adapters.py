from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from data.pipeline import TextRecord, deduplicate, filter_record, write_jsonl


def load_jsonl_records(path: str | Path) -> Iterator[TextRecord]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, str):
                yield TextRecord(text=data)
                continue
            yield TextRecord(
                text=data.get("text") or data.get("content") or "",
                source=data.get("source", "unknown"),
                license=data.get("license", "unknown"),
                domain=data.get("domain", "general"),
                language=data.get("language", "unknown"),
                quality_score=float(data.get("quality_score", 1.0)),
                metadata=data.get("metadata") or {},
            )


def load_text_file(path: str | Path, *, source: str = "file", license: str = "unknown") -> list[TextRecord]:
    text = Path(path).read_text(encoding="utf-8")
    # Split on blank lines into paragraphs
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return [TextRecord(text=c, source=source, license=license) for c in chunks]


def prepare_corpus(
    inputs: list[str | Path],
    output: str | Path,
    *,
    min_chars: int = 20,
    dedupe: bool = True,
) -> dict:
    """Load mixed text/jsonl inputs, filter, optional dedupe, write JSONL."""
    records: list[TextRecord] = []
    for path in inputs:
        path = Path(path)
        if path.suffix.lower() == ".jsonl":
            records.extend(list(load_jsonl_records(path)))
        else:
            records.extend(load_text_file(path, source=str(path)))

    records = [r for r in records if filter_record(r, min_chars=min_chars)]
    before = len(records)
    if dedupe:
        records = deduplicate(records)
    write_jsonl(records, output)
    return {"input_records": before, "output_records": len(records), "output": str(output)}
