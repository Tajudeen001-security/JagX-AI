from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable, Optional
from urllib.parse import urlparse


def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "jagx-ai",
        "provider_independent": True,
    }


def model_info() -> dict[str, Any]:
    return {
        "name": "jagx",
        "runtime": "local",
        "external_ai_api_required": False,
    }


class _Handler(BaseHTTPRequestHandler):
    get_routes: dict[str, Callable[[], dict]] = {}
    post_routes: dict[str, Callable[[dict], dict]] = {}

    def log_message(self, format: str, *args) -> None:
        return

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in self.get_routes:
            self._json(200, self.get_routes[path]())
            return
        self._json(404, {"error": "not_found", "path": path})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in self.post_routes:
            self._json(404, {"error": "not_found", "path": path})
            return
        payload = self._read_json()
        try:
            result = self.post_routes[path](payload)
            self._json(200, result)
        except Exception as e:
            self._json(400, {"error": str(e)})


def _default_generate(payload: dict) -> dict:
    """Placeholder generate that does not call external APIs.

    When a model is not bound, returns an explicit local-stub response so clients
    can detect that weights are not loaded rather than silently proxying.
    """
    prompt = payload.get("prompt") or payload.get("input") or ""
    max_tokens = int(payload.get("max_tokens") or payload.get("tokens") or 64)
    return {
        "object": "jagx.generation",
        "backend": "local-stub",
        "prompt_chars": len(str(prompt)),
        "max_tokens": max_tokens,
        "text": "",
        "note": "Bind a loaded JagXTransformer via create_app(generate_fn=...) for real generation.",
    }


def _orchestrator_execute(payload: dict) -> dict:
    """Route a request through the unified runtime orchestrator."""
    from runtime.orchestrator import build_default_orchestrator

    orch = getattr(_orchestrator_execute, "_orch", None)
    if orch is None:
        orch = build_default_orchestrator()
        _orchestrator_execute._orch = orch  # type: ignore[attr-defined]
    result = orch.execute(payload)
    return result.to_dict()


def _orchestrator_health() -> dict:
    from runtime.orchestrator import build_default_orchestrator

    orch = getattr(_orchestrator_execute, "_orch", None)
    if orch is None:
        orch = build_default_orchestrator()
        _orchestrator_execute._orch = orch  # type: ignore[attr-defined]
    base = health()
    base["runtime"] = orch.health_snapshot()
    return base


def create_app(
    extra_get: Optional[dict[str, Callable[[], dict]]] = None,
    extra_post: Optional[dict[str, Callable[[dict], dict]]] = None,
    generate_fn: Optional[Callable[[dict], dict]] = None,
    use_orchestrator: bool = True,
) -> type[BaseHTTPRequestHandler]:
    get_routes: dict[str, Callable[[], dict]] = {
        "/health": _orchestrator_health if use_orchestrator else health,
        "/v1/health": _orchestrator_health if use_orchestrator else health,
        "/v1/models": model_info,
    }
    post_routes: dict[str, Callable[[dict], dict]] = {
        "/v1/generate": generate_fn or _default_generate,
        "/v1/chat": generate_fn or _default_generate,
    }
    if use_orchestrator:
        post_routes["/v1/execute"] = _orchestrator_execute
        post_routes["/v1/agent"] = lambda p: _orchestrator_execute({**p, "kind": "agent"})
        post_routes["/v1/memory"] = lambda p: _orchestrator_execute({**p, "kind": "memory"})

    if extra_get:
        get_routes.update(extra_get)
    if extra_post:
        post_routes.update(extra_post)

    class Handler(_Handler):
        pass

    Handler.get_routes = get_routes
    Handler.post_routes = post_routes
    # backward compat for older tests looking at .routes
    Handler.routes = get_routes
    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    handler = create_app()
    server = HTTPServer((host, port), handler)
    print(f"JagX API listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
