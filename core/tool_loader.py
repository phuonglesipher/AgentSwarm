from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from core.config_loader import load_agentswarm_config, load_project_manifest
from core.front_matter import parse_markdown_front_matter
from core.llm import LLMManager, ensure_traced_llm_client
from core.models import ToolContext, ToolMetadata, ToolRuntime
from core.runtime_paths import RuntimePaths, resolve_runtime_paths
from core.tool_registry import ToolRegistry


def _parse_tool_markdown(markdown_path: Path, *, namespace: str) -> ToolMetadata:
    front_matter, description = parse_markdown_front_matter(markdown_path)
    return ToolMetadata(
        name=str(front_matter["name"]),
        namespace=namespace,
        entry=str(front_matter["entry"]),
        version=str(front_matter.get("version", "1.0.0")),
        description=description,
        capabilities=list(front_matter.get("capabilities", [])),
        output_mode=str(front_matter.get("output_mode", "message")),
        state_keys_shared=list(front_matter.get("state_keys_shared", ["messages"])),
        llm_profile=str(front_matter["llm_profile"]) if "llm_profile" in front_matter else None,
        tool_dir=markdown_path.parent,
    )


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_tools(
    project_root: Path,
    tools_root: Path,
    llm_manager: LLMManager,
    *,
    runtime_paths: RuntimePaths | None = None,
    config: Any | None = None,
    manifest: Any | None = None,
) -> ToolRegistry:
    paths = runtime_paths or resolve_runtime_paths(project_root, host_root=project_root)
    active_config = config or load_agentswarm_config(paths)
    active_manifest = manifest or load_project_manifest(paths)
    registry = ToolRegistry(preferred_namespaces=active_config.tool_sources)

    tool_sources = [
        ("agentswarm", paths.built_in_tools_root),
        ("project", paths.project_tools_root),
    ]
    for namespace, source_root in tool_sources:
        if not source_root.exists():
            continue
        for markdown_path in sorted(source_root.glob("*/Tool.md")):
            metadata = _parse_tool_markdown(markdown_path, namespace=namespace)
            entry_path = metadata.tool_dir / metadata.entry
            module_name = f"tool_{metadata.qualified_name.replace('::', '__').replace('-', '_')}"
            module = _load_module(entry_path, module_name)
            context = ToolContext(
                project_root=paths.host_root,
                agent_root=paths.agent_root,
                host_root=paths.host_root,
                overlay_root=paths.overlay_root,
                artifact_root=paths.runs_root,
                memory_root=paths.memory_root,
                tools_root=source_root,
                tool_dir=metadata.tool_dir,
                runtime_paths=paths,
                config=active_config,
                manifest=active_manifest,
                target_scope=active_config.target_scope,
                llm=ensure_traced_llm_client(llm_manager.resolve(metadata.llm_profile)),
                llm_manager=llm_manager,
                get_llm=lambda profile=None, active_manager=llm_manager: ensure_traced_llm_client(
                    active_manager.resolve(profile)
                ),
            )
            tool_object = module.build_tool(context=context, metadata=metadata)
            if not hasattr(tool_object, "invoke") or not hasattr(tool_object, "name"):
                raise TypeError(f"{entry_path} must return a LangChain tool with invoke() and name")
            registry.register(ToolRuntime(metadata=metadata, tool=tool_object))

    return registry
