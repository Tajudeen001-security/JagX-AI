from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SecretHit:
    rule: str
    start: int
    end: int
    snippet: str


# Conservative patterns for common accidental credential leaks in generated/agent code.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("generic_api_key", re.compile(r"(?i)(api[_-]?key|secret|token)[\"'\s:=]+[\"']?([a-zA-Z0-9_\-]{20,})[\"']?")),
    ("private_key_header", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("jwt_like", re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+")),
]


def scan_text(text: str) -> list[SecretHit]:
    hits: list[SecretHit] = []
    for name, pattern in _RULES:
        for m in pattern.finditer(text):
            snippet = text[m.start() : min(m.end(), m.start() + 24)]
            hits.append(SecretHit(rule=name, start=m.start(), end=m.end(), snippet=snippet))
    return hits


def scan_files(paths: Iterable[str]) -> dict[str, list[SecretHit]]:
    from pathlib import Path

    out: dict[str, list[SecretHit]] = {}
    for p in paths:
        path = Path(p)
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hits = scan_text(content)
        if hits:
            out[str(path)] = hits
    return out


def contains_secrets(text: str) -> bool:
    return bool(scan_text(text))
