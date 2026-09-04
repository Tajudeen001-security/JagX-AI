"""Coding capability handler for the unified orchestrator."""

from __future__ import annotations

from typing import Any

from runtime.orchestrator import ExecutionContext


def code_handler(payload: dict[str, Any], ctx: ExecutionContext) -> dict[str, Any]:
    """Run sandboxed write→test when CodingEngine or workspace+files provided."""
    engine = payload.get("_coding_engine")

    if engine is not None and hasattr(engine, "run"):
        out = engine.run(payload)
        if isinstance(out, dict):
            out.setdefault("request_id", ctx.request_id)
            return out

    if engine is not None and hasattr(engine, "write_and_test"):
        files = payload.get("files") or {}
        if isinstance(files, dict) and files:
            result = engine.write_and_test(
                files,
                test_command=str(payload.get("test_command") or "python3 -m pytest -q"),
                timeout_s=float(payload.get("timeout_s") or 60.0),
            )
            return {
                "ok": result.ok,
                "files_written": result.files_written,
                "test_output": (result.test_output or "")[:4000],
                "error": result.error,
                "attempts": result.attempts,
                "request_id": ctx.request_id,
                "backend": "coding-engine",
            }

    workspace = payload.get("workspace") or payload.get("repo_path")
    files = payload.get("files")
    if workspace and isinstance(files, dict) and files:
        try:
            from tools.sandbox import WorkspaceSandbox
            from coding.engine import CodingEngine

            eng = CodingEngine(WorkspaceSandbox(workspace))
            result = eng.write_and_test(
                files,
                test_command=str(payload.get("test_command") or "python3 -m pytest -q"),
                timeout_s=float(payload.get("timeout_s") or 60.0),
            )
            return {
                "ok": result.ok,
                "files_written": result.files_written,
                "test_output": (result.test_output or "")[:4000],
                "error": result.error,
                "attempts": result.attempts,
                "request_id": ctx.request_id,
                "backend": "coding-engine-auto",
            }
        except Exception as exc:
            return {
                "status": "failed",
                "error": str(exc),
                "request_id": ctx.request_id,
                "backend": "coding-error",
            }

    instruction = str(
        payload.get("instruction") or payload.get("prompt") or payload.get("code") or ""
    )
    return {
        "instruction": instruction[:500],
        "status": "accepted",
        "note": "Pass files+workspace or _coding_engine for sandboxed write/test",
        "request_id": ctx.request_id,
        "backend": "coding-accepted",
    }


def register(orch) -> None:
    from runtime.orchestrator import TaskKind

    orch.register_handler(TaskKind.CODE, code_handler)
