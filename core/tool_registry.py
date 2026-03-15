from __future__ import annotations

from core.models import ToolMetadata, ToolRuntime


class ToolRegistry:
    def __init__(self, *, preferred_namespaces: tuple[str, ...] = ("project", "agentswarm")) -> None:
        self._tools: dict[str, ToolRuntime] = {}
        self._aliases: dict[str, str] = {}
        self._namespace_priority = {
            namespace: index for index, namespace in enumerate(preferred_namespaces)
        }

    def register(self, runtime: ToolRuntime) -> None:
        metadata = runtime.metadata
        self._tools[metadata.qualified_name] = runtime
        alias_target = self._aliases.get(metadata.name)
        if alias_target is None:
            self._aliases[metadata.name] = metadata.qualified_name
            return

        current_runtime = self._tools[alias_target]
        current_priority = self._namespace_priority.get(current_runtime.metadata.namespace, 999)
        incoming_priority = self._namespace_priority.get(metadata.namespace, 999)
        if incoming_priority <= current_priority:
            self._aliases[metadata.name] = metadata.qualified_name

    def get(self, name: str) -> ToolRuntime:
        qualified_name = self._aliases.get(name, name)
        return self._tools[qualified_name]

    def list_metadata(self, include_shadowed: bool = True) -> list[ToolMetadata]:
        if include_shadowed:
            metadata = [runtime.metadata for runtime in self._tools.values()]
        else:
            metadata = [self._tools[qualified_name].metadata for qualified_name in self._aliases.values()]
        return sorted(metadata, key=lambda item: item.qualified_name)
