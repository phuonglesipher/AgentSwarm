from __future__ import annotations

from core.models import ToolMetadata, ToolRuntime


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolRuntime] = {}

    def register(self, runtime: ToolRuntime) -> None:
        self._tools[runtime.metadata.name] = runtime

    def get(self, name: str) -> ToolRuntime:
        return self._tools[name]

    def list_metadata(self) -> list[ToolMetadata]:
        return sorted((runtime.metadata for runtime in self._tools.values()), key=lambda item: item.name)
