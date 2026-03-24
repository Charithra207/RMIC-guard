"""Tool execution layer — no policy or IDS; dispatch only."""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["ToolRegistry", "ToolResult"]

ToolFn = Callable[..., Any]


class ToolResult:
    """Opaque result wrapper for logging and dashboard consumption."""

    __slots__ = ("ok", "data", "error")

    def __init__(self, ok: bool, data: Any = None, error: str | None = None) -> None:
        self.ok = ok
        self.data = data
        self.error = error


class ToolRegistry:
    """Register callables by exact tool name. No enforcement."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def has(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        fn = self._tools.get(name)
        if fn is None:
            return ToolResult(False, error=f"unknown_tool:{name}")
        try:
            out = fn(**kwargs)
            return ToolResult(True, data=out)
        except Exception as exc:  # noqa: BLE001 — surface to caller as tool error
            return ToolResult(False, error=str(exc))
