from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from core.front_matter import parse_markdown_front_matter
from core.llm import LLMManager
from core.models import WorkflowContext, WorkflowMetadata, WorkflowRuntime
from core.registry import WorkflowRegistry
from core.tool_graph import build_tool_subgraph
from core.tool_loader import load_tools


def _load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_workflow_markdown(markdown_path: Path) -> WorkflowMetadata:
    front_matter, description = parse_markdown_front_matter(markdown_path)
    return WorkflowMetadata(
        name=str(front_matter["name"]),
        entry=str(front_matter["entry"]),
        version=str(front_matter.get("version", "1.0.0")),
        description=description,
        capabilities=list(front_matter.get("capabilities", [])),
        exposed=bool(front_matter.get("exposed", True)),
        llm_profile=str(front_matter["llm_profile"]) if "llm_profile" in front_matter else None,
        tools=list(front_matter.get("tools", [])),
        workflow_dir=markdown_path.parent,
    )


def _require_graph_runtime(runtime: WorkflowRuntime) -> Any:
    graph = runtime.graph
    if graph is None or not hasattr(graph, "get_graph"):
        raise TypeError(f"Workflow {runtime.metadata.name} does not expose a graph runtime")
    return graph


def load_workflows(project_root: Path, workflows_root: Path, llm_manager: LLMManager) -> WorkflowRegistry:
    if not workflows_root.exists():
        raise FileNotFoundError(f"Workflow directory not found: {workflows_root}")

    registry = WorkflowRegistry()
    tools_root = project_root / "Tools"
    tool_registry = load_tools(project_root=project_root, tools_root=tools_root, llm_manager=llm_manager)
    workflow_markdowns = sorted(workflows_root.glob("*/Workflow.md"))
    workflow_specs: dict[str, tuple[WorkflowMetadata, Any]] = {}

    for markdown_path in workflow_markdowns:
        metadata = _parse_workflow_markdown(markdown_path)
        entry_path = metadata.workflow_dir / metadata.entry
        workflow_specs[metadata.name] = (
            metadata,
            _load_module(entry_path, f"workflow_{metadata.name.replace('-', '_')}"),
        )

    runtime_cache: dict[str, WorkflowRuntime] = {}
    building: set[str] = set()

    def build_runtime(name: str) -> WorkflowRuntime:
        if name in runtime_cache:
            return runtime_cache[name]
        if name not in workflow_specs:
            raise KeyError(f"Unknown workflow: {name}")
        if name in building:
            raise RuntimeError(f"Cyclic workflow dependency detected while building {name}")

        metadata, module = workflow_specs[name]
        building.add(name)
        try:
            context = WorkflowContext(
                project_root=project_root,
                workflows_root=workflows_root,
                workflow_dir=metadata.workflow_dir,
                tools_root=tools_root,
                llm=llm_manager.resolve(metadata.llm_profile),
                llm_manager=llm_manager,
                get_llm=lambda profile=None, active_manager=llm_manager: active_manager.resolve(profile),
                get_tool=lambda target_name, active_registry=tool_registry: active_registry.get(target_name),
                register_tools=lambda tool_names, state_schema, active_registry=tool_registry: {
                    tool_name: build_tool_subgraph(active_registry.get(tool_name), state_schema)
                    for tool_name in tool_names
                },
                list_tool_metadata=lambda active_registry=tool_registry: active_registry.list_metadata(),
                invoke_workflow=lambda target_name, payload: build_runtime(target_name).invoke(payload),
                get_workflow_graph=lambda target_name: _require_graph_runtime(build_runtime(target_name)),
            )

            graph_or_runtime = module.build_graph(context=context, metadata=metadata)
            runtime_object = graph_or_runtime.compile() if hasattr(graph_or_runtime, "compile") else graph_or_runtime
            if not hasattr(runtime_object, "invoke"):
                entry_path = metadata.workflow_dir / metadata.entry
                raise TypeError(f"{entry_path} must return a graph or runtime with an invoke method")

            runtime = WorkflowRuntime(
                metadata=metadata,
                invoke=lambda payload, active_runtime=runtime_object: active_runtime.invoke(payload),
                graph=runtime_object,
            )
            runtime_cache[name] = runtime
            registry.register(runtime)
            return runtime
        finally:
            building.remove(name)

    for workflow_name in sorted(workflow_specs):
        build_runtime(workflow_name)

    return registry
