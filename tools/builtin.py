"""Safe default tools with no shell and no network."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tools.registry import ToolResult, ToolSpec

if TYPE_CHECKING:
    from tools.registry import ToolRegistry


def register_builtin_tools(registry: "ToolRegistry") -> None:
    """Register low-risk tools available without a workspace."""

    def echo(args: dict) -> ToolResult:
        msg = str(args.get("message") or args.get("text") or "")
        return ToolResult(ok=True, data={"echo": msg})

    registry.register(
        ToolSpec(
            name="echo",
            description="Echo a message (debug / smoke tool)",
            input_schema={"type": "object", "properties": {"message": {"type": "string"}}},
            permission="none",
            timeout_s=5.0,
        ),
        echo,
    )

    def now(_args: dict) -> ToolResult:
        return ToolResult(ok=True, data={"unix": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    registry.register(
        ToolSpec(
            name="now",
            description="Current UTC time",
            input_schema={"type": "object", "properties": {}},
            permission="none",
            timeout_s=5.0,
        ),
        now,
    )

    def list_tools(_args: dict) -> ToolResult:
        tools = [
            {"name": s.name, "description": s.description, "permission": s.permission}
            for s in registry.list_tools()
        ]
        return ToolResult(ok=True, data={"tools": tools})

    registry.register(
        ToolSpec(
            name="list_tools",
            description="List registered tools",
            input_schema={"type": "object", "properties": {}},
            permission="none",
            timeout_s=5.0,
        ),
        list_tools,
    )
