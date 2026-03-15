from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from core.llm import LLMManager
from core.models import WorkflowContext, WorkflowMetadata, WorkflowRuntime
from core.registry import WorkflowRegistry


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip().strip('"').strip("'")
    lowered = cleaned.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return cleaned


def _parse_workflow_markdown(markdown_path: Path) -> WorkflowMetadata:
    content = markdown_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{markdown_path} must start with front matter delimited by ---")

    front_matter: dict[str, Any] = {}
    list_key: str | None = None
    index = 1
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "---":
            index += 1
            break
        if not stripped:
            index += 1
            continue
        if stripped.startswith("- "):
            if list_key is None:
                raise ValueError(f"List item found before a key in {markdown_path}")
            front_matter.setdefault(list_key, []).append(stripped[2:].strip())
            index += 1
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid front matter line in {markdown_path}: {stripped}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            front_matter[key] = _parse_scalar(value)
            list_key = None
        else:
            front_matter[key] = []
            list_key = key
        index += 1

    description = "\n".join(lines[index:]).strip()
    return WorkflowMetadata(
        name=str(front_matter["name"]),
        entry=str(front_matter["entry"]),
        version=str(front_matter.get("version", "1.0.0")),
        description=description,
        capabilities=list(front_matter.get("capabilities", [])),
        exposed=bool(front_matter.get("exposed", True)),
        llm_profile=str(front_matter["llm_profile"]) if "llm_profile" in front_matter else None,
        workflow_dir=markdown_path.parent,
    )


def _load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _require_graph_runtime(runtime: WorkflowRuntime) -> Any:
    graph = runtime.graph
    if graph is None or not hasattr(graph, "get_graph"):
        raise TypeError(f"Workflow {runtime.metadata.name} does not expose a graph runtime")
    return graph


def load_workflows(project_root: Path, workflows_root: Path, llm_manager: LLMManager) -> WorkflowRegistry:
    if not workflows_root.exists():
        raise FileNotFoundError(f"Workflow directory not found: {workflows_root}")

    registry = WorkflowRegistry()
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
                llm=llm_manager.resolve(metadata.llm_profile),
                llm_manager=llm_manager,
                get_llm=lambda profile=None, active_manager=llm_manager: active_manager.resolve(profile),
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
