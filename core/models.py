from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from core.config_loader import AgentSwarmConfig, ProjectManifest
    from core.llm import LLMClient, LLMManager
    from core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class WorkflowMetadata:
    name: str
    namespace: str
    entry: str
    version: str
    description: str
    capabilities: list[str]
    exposed: bool
    llm_profile: str | None
    tools: list[str]
    workflow_dir: Path

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}::{self.name}"


@dataclass(frozen=True)
class WorkflowContext:
    project_root: Path
    agent_root: Path
    host_root: Path
    overlay_root: Path
    artifact_root: Path
    memory_root: Path
    workflows_root: Path
    workflow_dir: Path
    tools_root: Path
    runtime_paths: "RuntimePaths"
    config: "AgentSwarmConfig"
    manifest: "ProjectManifest"
    target_scope: str
    llm: "LLMClient"
    llm_manager: "LLMManager"
    get_llm: Callable[[str | None], "LLMClient"]
    get_tool: Callable[[str], "ToolRuntime"]
    register_tools: Callable[[list[str], type[Any]], dict[str, Any]]
    list_tool_metadata: Callable[[], list["ToolMetadata"]]
    invoke_workflow: Callable[[str, dict[str, Any]], dict[str, Any]]
    get_workflow_graph: Callable[[str], Any]

    def resolve_scope_root(self, scope: str) -> Path:
        return self.agent_root if scope == "agentswarm" else self.host_root


@dataclass
class WorkflowRuntime:
    metadata: WorkflowMetadata
    invoke: Callable[[dict[str, Any]], dict[str, Any]]
    graph: Any | None = None


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    namespace: str
    entry: str
    version: str
    description: str
    capabilities: list[str]
    output_mode: str
    state_keys_shared: list[str]
    llm_profile: str | None
    tool_dir: Path

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}::{self.name}"


@dataclass(frozen=True)
class ToolContext:
    project_root: Path
    agent_root: Path
    host_root: Path
    overlay_root: Path
    artifact_root: Path
    memory_root: Path
    tools_root: Path
    tool_dir: Path
    runtime_paths: "RuntimePaths"
    config: "AgentSwarmConfig"
    manifest: "ProjectManifest"
    target_scope: str
    llm: "LLMClient"
    llm_manager: "LLMManager"
    get_llm: Callable[[str | None], "LLMClient"]

    def resolve_scope_root(self, scope: str) -> Path:
        return self.agent_root if scope == "agentswarm" else self.host_root


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
