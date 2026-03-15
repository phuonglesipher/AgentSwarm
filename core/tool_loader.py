from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from core.front_matter import parse_markdown_front_matter
from core.llm import LLMManager
from core.models import ToolContext, ToolMetadata, ToolRuntime
from core.tool_registry import ToolRegistry


def _parse_tool_markdown(markdown_path: Path) -> ToolMetadata:
    front_matter, description = parse_markdown_front_matter(markdown_path)
    return ToolMetadata(
        name=str(front_matter["name"]),
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


def load_tools(project_root: Path, tools_root: Path, llm_manager: LLMManager) -> ToolRegistry:
    registry = ToolRegistry()
    if not tools_root.exists():
        return registry

    for markdown_path in sorted(tools_root.glob("*/Tool.md")):
        metadata = _parse_tool_markdown(markdown_path)
        entry_path = metadata.tool_dir / metadata.entry
        module = _load_module(entry_path, f"tool_{metadata.name.replace('-', '_')}")
        context = ToolContext(
            project_root=project_root,
            tools_root=tools_root,
            tool_dir=metadata.tool_dir,
            llm=llm_manager.resolve(metadata.llm_profile),
            llm_manager=llm_manager,
            get_llm=lambda profile=None, active_manager=llm_manager: active_manager.resolve(profile),
        )
        tool_object = module.build_tool(context=context, metadata=metadata)
        if not hasattr(tool_object, "invoke") or not hasattr(tool_object, "name"):
            raise TypeError(f"{entry_path} must return a LangChain tool with invoke() and name")
        registry.register(ToolRuntime(metadata=metadata, tool=tool_object))

    return registry
