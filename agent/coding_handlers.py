"""Real sandbox-backed handlers for coding TaskDAG nodes."""

from __future__ import annotations

from typing import Any, Callable, Optional

from agent.planner import TaskNode
from coding.engine import CodingEngine
from tools.sandbox import WorkspaceSandbox


def build_coding_handlers(
    sandbox: WorkspaceSandbox,
    *,
    files: Optional[dict[str, str]] = None,
    test_command: str = "python3 -m pytest -q",
    timeout_s: float = 60.0,
    repair_fn: Optional[Callable[[dict[str, str], str], dict[str, str]]] = None,
) -> dict[str, Callable[[TaskNode, dict[str, Any]], Any]]:
    """Handlers that perform real filesystem and test operations.

    Code content must be supplied via `files` or context['files'].
    This does not invent model-generated patches without an implementer.
    """
    engine = CodingEngine(sandbox)
    initial_files = dict(files or {})

    def inspect_repo(node: TaskNode, ctx: dict[str, Any]) -> dict[str, Any]:
        listing = sandbox.list_dir(".")
        ctx["workspace_files"] = listing
        return {
            "files": listing,
            "root": str(sandbox.root),
            "context_update": {"workspace_files": listing},
        }

    def implement(node: TaskNode, ctx: dict[str, Any]) -> dict[str, Any]:
        pending = ctx.get("files") or initial_files
        if not isinstance(pending, dict) or not pending:
            raise RuntimeError(
                "implement requires files dict in context or agent payload "
                "(no silent code invention)"
            )
        written = []
        for rel, content in pending.items():
            engine.write_file(str(rel), str(content))
            written.append(str(rel))
        ctx["files_written"] = written
        ctx["files"] = dict(pending)
        return {
            "written": written,
            "context_update": {"files_written": written, "files": dict(pending)},
        }

    def run_tests(node: TaskNode, ctx: dict[str, Any]) -> dict[str, Any]:
        result = engine.run_tests(test_command, timeout_s=timeout_s)
        ctx["last_test"] = result
        if not result.get("ok"):
            raise RuntimeError(
                (result.get("stderr") or result.get("stdout") or "tests failed")[:2000]
            )
        return {
            "ok": True,
            "stdout": (result.get("stdout") or "")[:2000],
            "context_update": {"last_test": result},
        }

    def repair(node: TaskNode, ctx: dict[str, Any]) -> dict[str, Any]:
        last = ctx.get("last_test") or {}
        if last.get("ok"):
            return {"ok": True, "note": "tests already passed; repair skipped"}
        current = dict(ctx.get("files") or initial_files)
        if not current:
            raise RuntimeError("repair has no files in context")
        if repair_fn is None:
            # Without a repair strategy, surface the test failure honestly
            err = (last.get("stderr") or last.get("stdout") or "tests failed")[:2000]
            raise RuntimeError(f"repair_fn not provided; last test error: {err}")
        test_out = (last.get("stdout") or "") + (last.get("stderr") or "")
        updated = repair_fn(current, test_out)
        if not isinstance(updated, dict):
            raise RuntimeError("repair_fn must return dict[str,str]")
        for rel, content in updated.items():
            engine.write_file(str(rel), str(content))
        ctx["files"] = updated
        # Re-run tests after repair
        result = engine.run_tests(test_command, timeout_s=timeout_s)
        ctx["last_test"] = result
        if not result.get("ok"):
            raise RuntimeError(
                (result.get("stderr") or result.get("stdout") or "tests still failing")[:2000]
            )
        return {"ok": True, "repaired": True, "context_update": {"files": updated, "last_test": result}}

    def understand(node: TaskNode, ctx: dict[str, Any]) -> dict[str, Any]:
        return {"goal": ctx.get("goal") or node.description, "status": "understood"}

    def plan(node: TaskNode, ctx: dict[str, Any]) -> dict[str, Any]:
        return {"status": "planned", "workspace": str(sandbox.root)}

    def summarize(node: TaskNode, ctx: dict[str, Any]) -> dict[str, Any]:
        return {
            "files_written": ctx.get("files_written", []),
            "last_test_ok": bool((ctx.get("last_test") or {}).get("ok")),
            "workspace": str(sandbox.root),
        }

    return {
        "understand": understand,
        "plan": plan,
        "inspect_repo": inspect_repo,
        "implement": implement,
        "run_tests": run_tests,
        "repair": repair,
        "summarize": summarize,
    }
