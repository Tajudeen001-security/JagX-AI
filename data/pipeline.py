from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class TextRecord:
    text: str
    source: str = "unknown"
    license: str = "unknown"
    domain: str = "general"
    language: str = "unknown"

    @property
    def fingerprint(self) -> str:
        normalized = re.sub(r"\s+", " ", self.text.strip().lower())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()

def filter_record(record: TextRecord, min_chars: int = 20) -> bool:
    text = normalize(record.text)
    if len(text) < min_chars:
        return False
    if not any(ch.isalnum() for ch in text):
        return False
    return True

def deduplicate(records: list[TextRecord]) -> list[TextRecord]:
    seen: set[str] = set()
    result=[]
    for r in records:
        r.text=normalize(r.text)
        if not filter_record(r) or r.fingerprint in seen:
            continue
        seen.add(r.fingerprint); result.append(r)
    return result

def write_jsonl(records: list[TextRecord], path: str):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(asdict(r),ensure_ascii=False)+"\n")
