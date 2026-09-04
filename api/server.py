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
    bound = bool(
        os.environ.get("JAGX_CHECKPOINT")
        and os.environ.get("JAGX_TOKENIZER")
        and Path(os.environ["JAGX_CHECKPOINT"]).exists()
    )
    data: list[dict[str, Any]] = []
    if bound:
        data.append({"id": "jagx-local", "object": "model", "weights_bound": True})
    try:
        from inference.nvidia_client import public_models_list

        pub = public_models_list()
        if pub.get("nvidia_keys_configured"):
            data.extend(pub.get("data") or [])
            return {
                "object": "list",
                "data": data,
                "weights_bound": bound,
                "cloud_configured": True,
                "cloud_key_slots": pub.get("nvidia_key_slots", 0),
            }
    except Exception:
        pass
    return {
        "object": "list",
        "data": data or [{"id": "jagx-local", "object": "model", "weights_bound": bound}],
        "weights_bound": bound,
        "cloud_configured": False,
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
            code_name = result.get("code") or result.get("error")
            if code_name in ("model_not_bound", "nvidia_not_configured"):
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
    return {
        "error": "model_not_bound",
        "code": "model_not_bound",
        "message": (
            "No local checkpoint and no NVIDIA_API_KEY. "
            "Train + bind JAGX_CHECKPOINT or set NVIDIA_API_KEY secret (see docs/NVIDIA_BACKEND.md)."
        ),
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


def _try_nvidia_generate_fn() -> Optional[Callable[[dict], dict]]:
    try:
        from inference.nvidia_client import get_nvidia_client, nvidia_generate_fn

        if get_nvidia_client() is None:
            return None
        return nvidia_generate_fn
    except Exception:
        return None


def _resolve_default_generate_fn() -> Callable[[dict], dict]:
    # Prefer local JagX weights; else NVIDIA cloud; else explicit error
    return _try_env_generate_fn() or _try_nvidia_generate_fn() or _unbound_generate


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
    base["weights_bound"] = bool(
        os.environ.get("JAGX_CHECKPOINT") and Path(os.environ["JAGX_CHECKPOINT"]).exists()
    ) if os.environ.get("JAGX_CHECKPOINT") else False
    base["cloud_configured"] = bool(os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_API_KEYS"))
    return base


def create_app(
    extra_get: Optional[dict[str, Callable[[], dict]]] = None,
    extra_post: Optional[dict[str, Callable[[dict], dict]]] = None,
    generate_fn: Optional[Callable[[dict], dict]] = None,
    use_orchestrator: bool = True,
) -> type[BaseHTTPRequestHandler]:
    if generate_fn is None:
        generate_fn = _resolve_default_generate_fn()

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
