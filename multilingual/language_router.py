from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .registry import LanguageSpec, get_seed_registry


class LanguageDetector(Protocol):
    def detect_text(self, text: str) -> str: ...
    def detect_audio(self, audio: bytes) -> str: ...


class SpeechProvider(Protocol):
    def transcribe(self, audio: bytes, language: str | None = None) -> str: ...
    def synthesize(self, text: str, language: str) -> bytes: ...


@dataclass(frozen=True)
class LanguageDecision:
    language: str
    confidence: float
    source: str


class LanguageRouter:
    """Keeps text and voice conversations in the language selected by context."""

    def __init__(self, detector: LanguageDetector | None = None,
                 speech: SpeechProvider | None = None,
                 registry: dict[str, LanguageSpec] | None = None) -> None:
        self.detector = detector
        self.speech = speech
        self.registry = registry or get_seed_registry()

    def route_text(self, text: str, preferred_language: str | None = None) -> LanguageDecision:
        if preferred_language and preferred_language in self.registry:
            return LanguageDecision(preferred_language, 1.0, "user_preference")
        if self.detector is None:
            return LanguageDecision("en", 0.0, "default")
        code = self.detector.detect_text(text)
        return LanguageDecision(code, 1.0, "text_detection")

    def transcribe(self, audio: bytes, preferred_language: str | None = None) -> tuple[LanguageDecision, str]:
        if self.speech is None:
            raise RuntimeError("no speech provider configured")
        decision = LanguageDecision(preferred_language or "und", 0.0, "audio")
        text = self.speech.transcribe(audio, preferred_language)
        if self.detector is not None and not preferred_language:
            decision = self.route_text(text)
        return decision, text

    def synthesize(self, text: str, language: str) -> bytes:
        if self.speech is None:
            raise RuntimeError("no speech provider configured")
        if language not in self.registry:
            raise ValueError(f"language not registered: {language}")
        return self.speech.synthesize(text, language)
