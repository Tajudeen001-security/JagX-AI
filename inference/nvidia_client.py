"""NVIDIA NIM / integrate.api.nvidia.com client (OpenAI-compatible).

Keys from environment (GitHub Actions secrets / deployment secrets):
  NVIDIA_API_KEY          single key
  NVIDIA_API_KEYS         comma-separated keys for rotation / failover
  NVIDIA_API_BASE         default https://integrate.api.nvidia.com/v1
  JAGX_NVIDIA_MODEL_<CAP> optional upstream override per capability (chat, code, ...)

Client responses use public capability ids only — never upstream model names.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from inference.capability_router import (
    CapabilityRoute,
    apply_upstream_overrides,
    list_public_capabilities,
    resolve_route,
)

DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"


def _load_keys() -> list[str]:
    multi = os.environ.get("NVIDIA_API_KEYS") or os.environ.get("NVIDIA_API_KEY_LIST") or ""
    keys = [k.strip() for k in multi.split(",") if k.strip()]
    single = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if single and single not in keys:
        keys.insert(0, single)
    return keys


def _apply_env_model_overrides() -> None:
    overrides: dict[str, str] = {}
    for key in ("chat", "fast", "code", "reason", "vision", "vision-fast"):
        env = os.environ.get(f"JAGX_NVIDIA_MODEL_{key.upper().replace('-', '_')}")
        if env:
            overrides[key] = env.strip()
    if overrides:
        apply_upstream_overrides(overrides)


@dataclass
class NvidiaClient:
    """Thread-safe multi-key client for NVIDIA OpenAI-compatible chat."""

    base_url: str = DEFAULT_BASE
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        self._keys = _load_keys()
        self._idx = 0
        self._lock = threading.Lock()
        _apply_env_model_overrides()

    @property
    def available(self) -> bool:
        return bool(self._keys)

    def key_count(self) -> int:
        return len(self._keys)

    def _next_key(self) -> str:
        with self._lock:
            if not self._keys:
                raise RuntimeError("no NVIDIA_API_KEY configured")
            key = self._keys[self._idx % len(self._keys)]
            self._idx += 1
            return key

    def _post(self, path: str, body: dict[str, Any], *, key: Optional[str] = None) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        data = json.dumps(body).encode("utf-8")
        last_err: Optional[Exception] = None
        keys_to_try = [key] if key else list(self._keys) or []
        if not keys_to_try:
            raise RuntimeError("no NVIDIA_API_KEY configured")

        # Rotate starting point
        if key is None and len(keys_to_try) > 1:
            with self._lock:
                start = self._idx % len(keys_to_try)
                self._idx += 1
            keys_to_try = keys_to_try[start:] + keys_to_try[:start]

        for k in keys_to_try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Authorization": f"Bearer {k}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
                last_err = RuntimeError(f"nvidia http {e.code}: {err_body}")
                # Fail over on rate limit / auth-ish issues
                if e.code in (401, 403, 429, 503):
                    continue
                raise last_err from e
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(str(last_err) if last_err else "nvidia request failed")

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        route: CapabilityRoute,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any]:
        if stream:
            raise ValueError("streaming not implemented in this client yet")
        body = {
            "model": route.upstream_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        raw = self._post("/chat/completions", body)
        text = ""
        choices = raw.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            text = str(msg.get("content") or "")
        # Public payload: no upstream model id
        return {
            "object": "jagx.generation",
            "backend": "cloud",
            "provider": "nvidia",
            "model": route.public_id,
            "capability": route.public_id,
            "text": text,
            "usage": raw.get("usage") or {},
            "external_ai_api_required": True,
        }

    def complete_prompt(
        self,
        prompt: str,
        *,
        capability: Optional[str] = None,
        public_id: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        image_url: Optional[str] = None,
        image_b64: Optional[str] = None,
        media_type: str = "image/png",
    ) -> dict[str, Any]:
        has_image = bool(image_url or image_b64)
        route = resolve_route(
            capability=capability,
            public_id=public_id,
            prompt=prompt,
            has_image=has_image,
        )
        if has_image and not route.supports_vision:
            route = resolve_route(capability="vision", has_image=True)

        if has_image:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt or "Describe this image."}]
            if image_url:
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                url = f"data:{media_type};base64,{image_b64}"
                content.append({"type": "image_url", "image_url": {"url": url}})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        return self.chat(
            messages=messages,
            route=route,
            max_tokens=max_tokens,
            temperature=temperature,
        )


_client: Optional[NvidiaClient] = None
_client_lock = threading.Lock()


def get_nvidia_client() -> Optional[NvidiaClient]:
    global _client
    with _client_lock:
        if _client is None:
            c = NvidiaClient(base_url=os.environ.get("NVIDIA_API_BASE") or DEFAULT_BASE)
            if c.available:
                _client = c
        return _client


def nvidia_generate_fn(payload: dict[str, Any]) -> dict[str, Any]:
    """API generate_fn using NVIDIA when keys are present."""
    client = get_nvidia_client()
    if client is None:
        return {
            "error": "nvidia_not_configured",
            "code": "nvidia_not_configured",
            "message": "Set NVIDIA_API_KEY or NVIDIA_API_KEYS secret.",
        }
    prompt = str(payload.get("prompt") or payload.get("input") or payload.get("text") or "")
    messages = payload.get("messages")
    max_tokens = int(payload.get("max_tokens") or payload.get("tokens") or 1024)
    temperature = float(payload.get("temperature") or 0.7)
    capability = payload.get("capability") or payload.get("task")
    public_id = payload.get("model")  # public alias only
    image_url = payload.get("image_url") or payload.get("image")
    image_b64 = payload.get("image_b64") or payload.get("image_base64")

    if isinstance(messages, list) and messages:
        route = resolve_route(
            capability=str(capability) if capability else None,
            public_id=str(public_id) if public_id else None,
            prompt=prompt,
            has_image=bool(image_url or image_b64),
        )
        if image_url or image_b64:
            # Prefer vision route if images present in multimodal messages
            if not route.supports_vision:
                route = resolve_route(capability="vision", has_image=True)
        return client.chat(
            messages=messages,
            route=route,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return client.complete_prompt(
        prompt,
        capability=str(capability) if capability else None,
        public_id=str(public_id) if public_id else None,
        max_tokens=max_tokens,
        temperature=temperature,
        image_url=str(image_url) if image_url else None,
        image_b64=str(image_b64) if image_b64 else None,
        media_type=str(payload.get("media_type") or "image/png"),
    )


def public_models_list() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": c["id"], "object": "model", "supports_vision": c["supports_vision"], "description": c["description"]}
            for c in list_public_capabilities()
        ],
        "nvidia_keys_configured": bool(_load_keys()),
        "nvidia_key_slots": len(_load_keys()),
    }
