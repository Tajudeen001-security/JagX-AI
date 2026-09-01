from __future__ import annotations
import hashlib

def content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

class ExactDeduplicator:
    def __init__(self):
        self.seen: set[str] = set()

    def accept(self, text: str) -> bool:
        digest = content_hash(text)
        if digest in self.seen:
            return False
        self.seen.add(digest)
        return True

    def filter(self, texts: list[str]) -> list[str]:
        return [text for text in texts if self.accept(text)]
