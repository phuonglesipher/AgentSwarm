from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from core.llm import LLMManager
from core.models import BlueprintContext, BlueprintMetadata, BlueprintRuntime
from core.registry import BlueprintRegistry


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip().strip('"').strip("'")
    lowered = cleaned.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return cleaned


def _parse_blueprint_markdown(markdown_path: Path) -> BlueprintMetadata:
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
    return BlueprintMetadata(
        name=str(front_matter["name"]),
        entry=str(front_matter["entry"]),
        version=str(front_matter.get("version", "1.0.0")),
        description=description,
        capabilities=list(front_matter.get("capabilities", [])),
        exposed=bool(front_matter.get("exposed", True)),
        llm_profile=str(front_matter["llm_profile"]) if "llm_profile" in front_matter else None,
        blueprint_dir=markdown_path.parent,
    )


def _load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _require_graph_runtime(runtime: BlueprintRuntime) -> Any:
    graph = runtime.graph
    if graph is None or not hasattr(graph, "get_graph"):
        raise TypeError(f"Blueprint {runtime.metadata.name} does not expose a graph runtime")
    return graph


def load_blueprints(project_root: Path, blueprints_root: Path, llm_manager: LLMManager) -> BlueprintRegistry:
    if not blueprints_root.exists():
        raise FileNotFoundError(f"Blueprint directory not found: {blueprints_root}")

    registry = BlueprintRegistry()
    blueprint_markdowns = sorted(blueprints_root.glob("*/Blueprint.md"))
    blueprint_specs: dict[str, tuple[BlueprintMetadata, Any]] = {}

    for markdown_path in blueprint_markdowns:
        metadata = _parse_blueprint_markdown(markdown_path)
        entry_path = metadata.blueprint_dir / metadata.entry
        blueprint_specs[metadata.name] = (
            metadata,
            _load_module(entry_path, f"blueprint_{metadata.name.replace('-', '_')}"),
        )

    runtime_cache: dict[str, BlueprintRuntime] = {}
    building: set[str] = set()

    def build_runtime(name: str) -> BlueprintRuntime:
        if name in runtime_cache:
            return runtime_cache[name]
        if name not in blueprint_specs:
            raise KeyError(f"Unknown blueprint: {name}")
        if name in building:
            raise RuntimeError(f"Cyclic blueprint dependency detected while building {name}")

        metadata, module = blueprint_specs[name]
        building.add(name)
        try:
            context = BlueprintContext(
                project_root=project_root,
                blueprints_root=blueprints_root,
                blueprint_dir=metadata.blueprint_dir,
                llm=llm_manager.resolve(metadata.llm_profile),
                llm_manager=llm_manager,
                get_llm=lambda profile=None, active_manager=llm_manager: active_manager.resolve(profile),
                invoke_blueprint=lambda target_name, payload: build_runtime(target_name).invoke(payload),
                get_blueprint_graph=lambda target_name: _require_graph_runtime(build_runtime(target_name)),
            )

            graph_or_runtime = module.build_graph(context=context, metadata=metadata)
            runtime_object = graph_or_runtime.compile() if hasattr(graph_or_runtime, "compile") else graph_or_runtime
            if not hasattr(runtime_object, "invoke"):
                entry_path = metadata.blueprint_dir / metadata.entry
                raise TypeError(f"{entry_path} must return a graph or runtime with an invoke method")

            runtime = BlueprintRuntime(
                metadata=metadata,
                invoke=lambda payload, active_runtime=runtime_object: active_runtime.invoke(payload),
                graph=runtime_object,
            )
            runtime_cache[name] = runtime
            registry.register(runtime)
            return runtime
        finally:
            building.remove(name)

    for blueprint_name in sorted(blueprint_specs):
        build_runtime(blueprint_name)

    return registry
