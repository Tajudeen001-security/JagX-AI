"""Map user intent to public capability aliases and internal NVIDIA routes.

Public responses MUST only use public_id (e.g. jagx-chat). Upstream model
identifiers stay internal to the server process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CapabilityRoute:
    public_id: str
    """Name shown to clients / app UI."""

    upstream_model: str
    """NVIDIA catalog model id used only on the server when calling the API."""

    supports_vision: bool = False
    description: str = ""


# Internal routing table. Upstream ids are never returned by the public API.
_ROUTES: dict[str, CapabilityRoute] = {
    "chat": CapabilityRoute(
        public_id="jagx-chat",
        upstream_model="meta/llama-3.3-70b-instruct",
        description="General chat and reasoning",
    ),
    "fast": CapabilityRoute(
        public_id="jagx-fast",
        upstream_model="meta/llama-3.1-8b-instruct",
        description="Low-latency short answers",
    ),
    "code": CapabilityRoute(
        public_id="jagx-code",
        upstream_model="qwen/qwen2.5-coder-32b-instruct",
        description="Coding and technical tasks",
    ),
    "reason": CapabilityRoute(
        public_id="jagx-reason",
        upstream_model="deepseek-ai/deepseek-r1",
        description="Hard reasoning / analysis",
    ),
    "vision": CapabilityRoute(
        public_id="jagx-vision",
        upstream_model="meta/llama-3.2-90b-vision-instruct",
        supports_vision=True,
        description="Image understanding and vision QA",
    ),
    "vision-fast": CapabilityRoute(
        public_id="jagx-vision-fast",
        upstream_model="meta/llama-3.2-11b-vision-instruct",
        supports_vision=True,
        description="Faster vision tasks",
    ),
}

# Allow override of upstream models via env without changing public ids
# (set in process by nvidia_client.load_env_overrides).


def list_public_capabilities() -> list[dict]:
    return [
        {
            "id": r.public_id,
            "capability": key,
            "supports_vision": r.supports_vision,
            "description": r.description,
        }
        for key, r in _ROUTES.items()
    ]


def resolve_route(
    *,
    capability: Optional[str] = None,
    public_id: Optional[str] = None,
    prompt: str = "",
    has_image: bool = False,
) -> CapabilityRoute:
    """Choose a route from explicit capability, public id, or simple heuristics."""
    if public_id:
        for r in _ROUTES.values():
            if r.public_id == public_id:
                return r
    if capability:
        key = capability.lower().strip()
        if key in _ROUTES:
            return _ROUTES[key]

    if has_image:
        return _ROUTES["vision"]

    g = (prompt or "").lower()
    if any(k in g for k in ("code", "python", "function", "bug", "refactor", "implement", "compile")):
        return _ROUTES["code"]
    if any(k in g for k in ("prove", "reason step", "analyze deeply", "chain of thought")):
        return _ROUTES["reason"]
    if any(k in g for k in ("quick", "short", "fast")):
        return _ROUTES["fast"]
    return _ROUTES["chat"]


def apply_upstream_overrides(overrides: dict[str, str]) -> None:
    """Replace upstream_model for known capability keys (operator config)."""
    global _ROUTES
    updated = dict(_ROUTES)
    for key, upstream in overrides.items():
        if key in updated and upstream:
            old = updated[key]
            updated[key] = CapabilityRoute(
                public_id=old.public_id,
                upstream_model=upstream,
                supports_vision=old.supports_vision,
                description=old.description,
            )
    _ROUTES = updated
