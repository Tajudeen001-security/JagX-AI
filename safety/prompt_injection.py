from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class InjectionScan:
    flagged: bool
    patterns: tuple[str, ...]
    reason: str


# Heuristic patterns commonly used in prompt-injection attempts against agents/tools.
# This is defense-in-depth for tool-using agents; not a complete security boundary.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_previous", re.compile(r"(?i)ignore (all )?(previous|prior|above) (instructions|rules|prompts)")),
    ("system_override", re.compile(r"(?i)(you are now|new system prompt|override system)")),
    ("reveal_secrets", re.compile(r"(?i)(print|reveal|show|dump).{0,40}(api[_-]?key|secret|password|token|credentials)")),
    ("disable_safety", re.compile(r"(?i)(disable|bypass|turn off).{0,30}(safety|guardrails|filter|policy)")),
    ("tool_exfil", re.compile(r"(?i)(exfiltrate|send (all )?files? to|curl .{0,40}http)")),
]


def scan_prompt(text: str) -> InjectionScan:
    hits: list[str] = []
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            hits.append(name)
    if hits:
        return InjectionScan(flagged=True, patterns=tuple(hits), reason="potential prompt-injection patterns detected")
    return InjectionScan(flagged=False, patterns=(), reason="no injection patterns matched")


def is_safe_for_tools(text: str) -> bool:
    """Return False if the prompt should not unlock elevated tools without review."""
    return not scan_prompt(text).flagged
