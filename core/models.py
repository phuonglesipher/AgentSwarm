from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from core.llm import LLMClient, LLMManager


@dataclass(frozen=True)
class WorkflowMetadata:
    name: str
    entry: str
    version: str
    description: str
    capabilities: list[str]
    exposed: bool
    llm_profile: str | None
    workflow_dir: Path


@dataclass(frozen=True)
class WorkflowContext:
    project_root: Path
    workflows_root: Path
    workflow_dir: Path
    llm: "LLMClient"
    llm_manager: "LLMManager"
    get_llm: Callable[[str | None], "LLMClient"]
    invoke_workflow: Callable[[str, dict[str, Any]], dict[str, Any]]
    get_workflow_graph: Callable[[str], Any]


@dataclass
class WorkflowRuntime:
    metadata: WorkflowMetadata
    invoke: Callable[[dict[str, Any]], dict[str, Any]]
    graph: Any | None = None


@dataclass
class RoutedTask:
    id: str
    description: str
    workflow_name: str | None = None
    status: str = "planned"
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] | None = None
    error: str | None = None
