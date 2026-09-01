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


class _Handler(BaseHTTPRequestHandler):
    routes: dict[str, Callable[[], dict]] = {}

    def log_message(self, format: str, *args) -> None:  # quieter tests
        return

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in self.routes:
            self._json(200, self.routes[path]())
            return
        self._json(404, {"error": "not_found", "path": path})


def create_app(extra_routes: Optional[dict[str, Callable[[], dict]]] = None) -> type[BaseHTTPRequestHandler]:
    routes = {"/health": health, "/v1/health": health}
    if extra_routes:
        routes.update(extra_routes)

    class Handler(_Handler):
        pass

    Handler.routes = routes
    return Handler


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    handler = create_app()
    server = HTTPServer((host, port), handler)
    print(f"JagX API listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
