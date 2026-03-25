from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from core.config_loader import load_agentswarm_config, load_project_manifest
from core.front_matter import parse_markdown_front_matter
from core.llm import LLMManager, ensure_traced_llm_client
from core.models import WorkflowContext, WorkflowMetadata, WorkflowRuntime
from core.registry import WorkflowRegistry
from core.runtime_paths import RuntimePaths, resolve_runtime_paths
from core.tool_graph import build_tool_subgraph
from core.tool_loader import load_tools


def _iter_workflow_markdown_files(source_root: Path) -> list[Path]:
    if not source_root.exists():
        return []
    return sorted(
        [path for path in source_root.rglob("Workflow.md") if path.is_file()],
        key=lambda path: path.relative_to(source_root).as_posix(),
    )


def _load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_workflow_markdown(markdown_path: Path, *, namespace: str) -> WorkflowMetadata:
    front_matter, description = parse_markdown_front_matter(markdown_path)
    return WorkflowMetadata(
        name=str(front_matter["name"]),
        namespace=namespace,
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
        raise TypeError(f"Workflow {runtime.metadata.qualified_name} does not expose a graph runtime")
    return graph


def load_workflows(
    project_root: Path,
    workflows_root: Path,
    llm_manager: LLMManager,
    *,
    runtime_paths: RuntimePaths | None = None,
    config: Any | None = None,
    manifest: Any | None = None,
) -> WorkflowRegistry:
    paths = runtime_paths or resolve_runtime_paths(project_root, host_root=project_root)
    built_in_workflows_root = Path(workflows_root).resolve()
    active_config = config or load_agentswarm_config(paths)
    active_manifest = manifest or load_project_manifest(paths)
    registry = WorkflowRegistry(preferred_namespaces=active_config.workflow_sources)
    tool_registry = load_tools(
        project_root=paths.agent_root,
        tools_root=paths.built_in_tools_root,
        llm_manager=llm_manager,
        runtime_paths=paths,
        config=active_config,
        manifest=active_manifest,
    )
    workflow_specs: dict[str, tuple[WorkflowMetadata, Any]] = {}
    workflow_aliases: dict[str, str] = {}
    namespace_priority = {
        namespace: index for index, namespace in enumerate(active_config.workflow_sources)
    }

    workflow_sources = [
        ("agentswarm", built_in_workflows_root),
        ("project", paths.project_workflows_root),
    ]

    for namespace, source_root in workflow_sources:
        for markdown_path in _iter_workflow_markdown_files(source_root):
            metadata = _parse_workflow_markdown(markdown_path, namespace=namespace)
            entry_path = metadata.workflow_dir / metadata.entry
            qualified_name = metadata.qualified_name
            if qualified_name in workflow_specs:
                raise ValueError(f"Duplicate workflow definition found: {qualified_name}")
            module = _load_module(entry_path, f"workflow_{qualified_name.replace('::', '__').replace('-', '_')}")
            workflow_specs[qualified_name] = (metadata, module)
            current_alias = workflow_aliases.get(metadata.name)
            if current_alias is None:
                workflow_aliases[metadata.name] = qualified_name
                continue
            current_namespace = workflow_specs[current_alias][0].namespace
            if namespace_priority.get(metadata.namespace, 999) <= namespace_priority.get(current_namespace, 999):
                workflow_aliases[metadata.name] = qualified_name

    runtime_cache: dict[str, WorkflowRuntime] = {}
    building: set[str] = set()

    def resolve_workflow_name(name: str) -> str:
        if name in workflow_specs:
            return name
        if name in workflow_aliases:
            return workflow_aliases[name]
        raise KeyError(f"Unknown workflow: {name}")

    def build_runtime(name: str) -> WorkflowRuntime:
        resolved_name = resolve_workflow_name(name)
        if resolved_name in runtime_cache:
            return runtime_cache[resolved_name]
        if resolved_name in building:
            raise RuntimeError(f"Cyclic workflow dependency detected while building {resolved_name}")

        metadata, module = workflow_specs[resolved_name]
        building.add(resolved_name)
        try:
            context = WorkflowContext(
                project_root=paths.host_root,
                agent_root=paths.agent_root,
                host_root=paths.host_root,
                overlay_root=paths.overlay_root,
                artifact_root=paths.runs_root,
                memory_root=paths.memory_root,
                workflows_root=paths.project_workflows_root if metadata.namespace == "project" else built_in_workflows_root,
                workflow_dir=metadata.workflow_dir,
                tools_root=paths.project_tools_root if metadata.namespace == "project" else paths.built_in_tools_root,
                runtime_paths=paths,
                config=active_config,
                manifest=active_manifest,
                target_scope=active_config.target_scope,
                llm=ensure_traced_llm_client(llm_manager.resolve(metadata.llm_profile)),
                llm_manager=llm_manager,
                get_llm=lambda profile=None, active_manager=llm_manager: ensure_traced_llm_client(
                    active_manager.resolve(profile)
                ),
                get_tool=lambda target_name, active_registry=tool_registry: active_registry.get(target_name),
                register_tools=lambda tool_names, state_schema, active_registry=tool_registry: {
                    active_registry.get(tool_name).metadata.qualified_name: build_tool_subgraph(
                        active_registry.get(tool_name),
                        state_schema,
                    )
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
            runtime_cache[resolved_name] = runtime
            registry.register(runtime)
            return runtime
        finally:
            building.remove(resolved_name)

    for workflow_name in sorted(workflow_specs):
        build_runtime(workflow_name)

    return registry
