from __future__ import annotations

from typing import Any

from core.models import BlueprintMetadata, BlueprintRuntime
from core.text_utils import tokenize


class BlueprintRegistry:
    def __init__(self) -> None:
        self._blueprints: dict[str, BlueprintRuntime] = {}

    def register(self, runtime: BlueprintRuntime) -> None:
        self._blueprints[runtime.metadata.name] = runtime

    def get(self, name: str) -> BlueprintRuntime:
        return self._blueprints[name]

    def invoke(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        runtime = self.get(name)
        return runtime.invoke(payload)

    def list_metadata(self, exposed_only: bool = False) -> list[BlueprintMetadata]:
        metadata = [runtime.metadata for runtime in self._blueprints.values()]
        if exposed_only:
            metadata = [item for item in metadata if item.exposed]
        return sorted(metadata, key=lambda item: item.name)

    def route(self, task_description: str) -> BlueprintMetadata | None:
        task_tokens = tokenize(task_description)
        best_match: BlueprintMetadata | None = None
        best_score = -1
        for metadata in self.list_metadata(exposed_only=True):
            source = " ".join([metadata.name, metadata.description, *metadata.capabilities])
            score = len(task_tokens & tokenize(source))
            if score > best_score:
                best_match = metadata
                best_score = score
        return best_match
