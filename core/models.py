from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

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
    tools: list[str]
    workflow_dir: Path


@dataclass(frozen=True)
class WorkflowContext:
    project_root: Path
    workflows_root: Path
    workflow_dir: Path
    tools_root: Path
    llm: "LLMClient"
    llm_manager: "LLMManager"
    get_llm: Callable[[str | None], "LLMClient"]
    get_tool: Callable[[str], "ToolRuntime"]
    register_tools: Callable[[list[str], type[Any]], dict[str, Any]]
    list_tool_metadata: Callable[[], list["ToolMetadata"]]
    invoke_workflow: Callable[[str, dict[str, Any]], dict[str, Any]]
    get_workflow_graph: Callable[[str], Any]


@dataclass
class WorkflowRuntime:
    metadata: WorkflowMetadata
    invoke: Callable[[dict[str, Any]], dict[str, Any]]
    graph: Any | None = None


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    entry: str
    version: str
    description: str
    capabilities: list[str]
    output_mode: str
    state_keys_shared: list[str]
    llm_profile: str | None
    tool_dir: Path


@dataclass(frozen=True)
class ToolContext:
    project_root: Path
    tools_root: Path
    tool_dir: Path
    llm: "LLMClient"
    llm_manager: "LLMManager"
    get_llm: Callable[[str | None], "LLMClient"]


@dataclass
class ToolRuntime:
    metadata: ToolMetadata
    tool: "BaseTool"


@dataclass
class RoutedTask:
    id: str
    description: str
    workflow_name: str | None = None
    status: str = "planned"
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] | None = None
    error: str | None = None
