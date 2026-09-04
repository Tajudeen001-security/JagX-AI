from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse


def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "jagx-ai",
        "provider_independent": True,
    }


def model_info() -> dict[str, Any]:
    bound = bool(os.environ.get("JAGX_CHECKPOINT") and os.environ.get("JAGX_TOKENIZER"))
    return {
        "name": "jagx",
        "runtime": "local",
        "external_ai_api_required": False,
        "weights_bound": bound,
        "object": "list",
        "data": [{"id": "jagx-local", "object": "model", "weights_bound": bound}],
    }


class _Handler(BaseHTTPRequestHandler):
    get_routes: dict[str, Callable[[], dict]] = {}
    post_routes: dict[str, Callable[[dict], dict]] = {}

    def log_message(self, format: str, *args) -> None:
        return

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        rid = payload.get("request_id")
        if rid:
            self.send_header("X-Request-Id", str(rid))
        self._cors()
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

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

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
            if not isinstance(result, dict):
                result = {"data": result}
            # Explicit unavailable (no weights)
            if result.get("error") == "model_not_bound" or result.get("code") == "model_not_bound":
                self._json(503, result)
                return
            status = str(result.get("status") or "").lower()
            if status in ("failed", "error"):
                code = 400
            elif status in ("timed_out",):
                code = 504
            else:
                code = 200
            self._json(code, result)
        except Exception as e:
            self._json(400, {"error": str(e), "error_type": type(e).__name__})


def _unbound_generate(payload: dict) -> dict:
    """No fake empty text — client must bind a real checkpoint."""
    return {
        "error": "model_not_bound",
        "code": "model_not_bound",
        "message": "No JagX checkpoint is bound. Train a model, then set JAGX_CHECKPOINT and JAGX_TOKENIZER or start with jagx serve --checkpoint ...",
        "prompt_chars": len(str(payload.get("prompt") or payload.get("input") or "")),
    }


def _try_env_generate_fn() -> Optional[Callable[[dict], dict]]:
    ckpt = os.environ.get("JAGX_CHECKPOINT")
    tok = os.environ.get("JAGX_TOKENIZER")
    if not ckpt or not tok:
        return None
    if not Path(ckpt).exists():
        return None
    try:
        from api.local_generate import make_generate_fn_from_paths

        return make_generate_fn_from_paths(ckpt, tok)
    except Exception:
        return None


def _orchestrator_execute(payload: dict) -> dict:
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
    base["weights_bound"] = bool(os.environ.get("JAGX_CHECKPOINT") and Path(os.environ["JAGX_CHECKPOINT"]).exists()) if os.environ.get("JAGX_CHECKPOINT") else False
    return base


def create_app(
    extra_get: Optional[dict[str, Callable[[], dict]]] = None,
    extra_post: Optional[dict[str, Callable[[dict], dict]]] = None,
    generate_fn: Optional[Callable[[dict], dict]] = None,
    use_orchestrator: bool = True,
) -> type[BaseHTTPRequestHandler]:
    if generate_fn is None:
        generate_fn = _try_env_generate_fn() or _unbound_generate

    get_routes: dict[str, Callable[[], dict]] = {
        "/health": _orchestrator_health if use_orchestrator else health,
        "/v1/health": _orchestrator_health if use_orchestrator else health,
        "/v1/models": model_info,
    }
    post_routes: dict[str, Callable[[dict], dict]] = {
        "/v1/generate": generate_fn,
        "/v1/chat": generate_fn,
    }
    if use_orchestrator:
        post_routes["/v1/execute"] = _orchestrator_execute
        post_routes["/v1/agent"] = lambda p: _orchestrator_execute({**p, "kind": "agent"})
        post_routes["/v1/memory"] = lambda p: _orchestrator_execute({**p, "kind": "memory"})
        post_routes["/v1/code"] = lambda p: _orchestrator_execute({**p, "kind": "code"})
        post_routes["/v1/tool"] = lambda p: _orchestrator_execute({**p, "kind": "tool"})

    if extra_get:
        get_routes.update(extra_get)
    if extra_post:
        post_routes.update(extra_post)

    class Handler(_Handler):
        pass

    Handler.get_routes = get_routes
    Handler.post_routes = post_routes
    Handler.routes = get_routes
    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    handler = create_app()
    server = HTTPServer((host, port), handler)
    print(f"JagX API listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
