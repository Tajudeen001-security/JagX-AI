"""Voice interface contract for African-language JagX clients.

The core model remains provider-independent. ASR/TTS engines are adapters: a
client can supply speech in a detected language, receive text, generate a
response, and request speech in the same language. The contract does not claim
that an engine supports a language until its capability manifest says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VoiceLanguageCapability:
    code: str
    name: str
    asr: bool = False
    tts: bool = False
    dialects: tuple[str, ...] = ()


@dataclass
class VoiceTurn:
    audio: bytes | None = None
    text: str | None = None
    language: str | None = None
    detected_language: str | None = None
    confidence: float | None = None
    reply_language: str | None = None
    reply_text: str | None = None
    reply_audio: bytes | None = None
    metadata: dict[str, str] = field(default_factory=dict)


def choose_reply_language(turn: VoiceTurn, default: str = "und") -> str:
    """Keep the conversation in the user's detected language when reliable."""
    if turn.detected_language and (turn.confidence is None or turn.confidence >= 0.80):
        return turn.detected_language
    if turn.language:
        return turn.language
    return default
