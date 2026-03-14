from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.llm import LLMClient, LLMManager


@dataclass(frozen=True)
class BlueprintMetadata:
    name: str
    entry: str
    version: str
    description: str
    capabilities: list[str]
    exposed: bool
    llm_profile: str | None
    blueprint_dir: Path


@dataclass(frozen=True)
class BlueprintContext:
    project_root: Path
    blueprints_root: Path
    blueprint_dir: Path
    llm: "LLMClient"
    llm_manager: "LLMManager"
    get_llm: Callable[[str | None], "LLMClient"]
    invoke_blueprint: Callable[[str, dict[str, Any]], dict[str, Any]]
    get_blueprint_graph: Callable[[str], Any]


@dataclass
class BlueprintRuntime:
    metadata: BlueprintMetadata
    invoke: Callable[[dict[str, Any]], dict[str, Any]]
    graph: Any | None = None


@dataclass
class RoutedTask:
    id: str
    description: str
    blueprint_name: str | None = None
    status: str = "planned"
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] | None = None
    error: str | None = None
