from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]
    requires_confirmation: bool = False


class ToolRegistry:
    """Small, auditable tool registry for JagX agent training/inference.

    Tools are explicit capabilities. The model never gets arbitrary Python
    execution merely by emitting a tool name; the host application decides
    which registered tools are enabled and whether confirmation is required.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name or spec.name in self._tools:
            raise ValueError(f"invalid or duplicate tool: {spec.name!r}")
        self._tools[spec.name] = spec

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "requires_confirmation": spec.requires_confirmation,
            }
            for spec in self._tools.values()
        ]

    def call(self, name: str, *, confirmed: bool = False, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        spec = self._tools[name]
        if spec.requires_confirmation and not confirmed:
            raise PermissionError(f"tool {name!r} requires explicit confirmation")
        return spec.handler(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)
