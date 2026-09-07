"""Language routing independent of the language model weights."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    family: str = ""
    african: bool = True


class LanguageDetector(Protocol):
    def detect(self, text: str) -> Language | None: ...


class LanguageRouter:
    """Selects the response language while preserving English as fallback."""

    def __init__(self, detector: LanguageDetector | None = None, default: Language | None = None):
        self.detector = detector
        self.default = default or Language("en", "English", "Indo-European", african=False)

    def detect(self, text: str) -> Language:
        if self.detector is None:
            return self.default
        return self.detector.detect(text) or self.default

    def instruction(self, user_text: str) -> dict[str, str]:
        language = self.detect(user_text)
        return {
            "language_code": language.code,
            "language_name": language.name,
            "system_instruction": f"Respond in {language.name} unless the user explicitly requests another language.",
        }
