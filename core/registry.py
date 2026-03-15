from __future__ import annotations

from typing import Any

from core.models import WorkflowMetadata, WorkflowRuntime
from core.text_utils import tokenize


class WorkflowRegistry:
    def __init__(self, *, preferred_namespaces: tuple[str, ...] = ("project", "agentswarm")) -> None:
        self._workflows: dict[str, WorkflowRuntime] = {}
        self._aliases: dict[str, str] = {}
        self._namespace_priority = {
            namespace: index for index, namespace in enumerate(preferred_namespaces)
        }

    def register(self, runtime: WorkflowRuntime) -> None:
        metadata = runtime.metadata
        self._workflows[metadata.qualified_name] = runtime
        alias_target = self._aliases.get(metadata.name)
        if alias_target is None:
            self._aliases[metadata.name] = metadata.qualified_name
            return

        current_runtime = self._workflows[alias_target]
        current_priority = self._namespace_priority.get(current_runtime.metadata.namespace, 999)
        incoming_priority = self._namespace_priority.get(metadata.namespace, 999)
        if incoming_priority <= current_priority:
            self._aliases[metadata.name] = metadata.qualified_name

    def get(self, name: str) -> WorkflowRuntime:
        qualified_name = self._aliases.get(name, name)
        return self._workflows[qualified_name]

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        runtime = self.get(name)
        return runtime.invoke(payload)

    def list_metadata(self, exposed_only: bool = False, include_shadowed: bool = True) -> list[WorkflowMetadata]:
        if include_shadowed:
            metadata = [runtime.metadata for runtime in self._workflows.values()]
        else:
            metadata = [self._workflows[qualified_name].metadata for qualified_name in self._aliases.values()]
        if exposed_only:
            metadata = [item for item in metadata if item.exposed]
        return sorted(metadata, key=lambda item: item.qualified_name)

    def route(self, task_description: str) -> WorkflowMetadata | None:
        task_tokens = tokenize(task_description)
        best_match: WorkflowMetadata | None = None
        best_score = -1
        for metadata in self.list_metadata(exposed_only=True, include_shadowed=False):
            source = " ".join([metadata.name, metadata.qualified_name, metadata.description, *metadata.capabilities])
            score = len(task_tokens & tokenize(source))
            if score > best_score:
                best_match = metadata
                best_score = score
        return best_match
