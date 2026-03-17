from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class AgentSwarmConfig:
    version: int = 1
    agent_root_name: str = "AgentSwarm"
    target_scope: str = "host_project"
    workflow_sources: tuple[str, ...] = ("project", "agentswarm")
    tool_sources: tuple[str, ...] = ("project", "agentswarm")
    source_roots: tuple[str, ...] = ("src", "app", "packages")
    doc_roots: tuple[str, ...] = ("docs", "design")
    test_roots: tuple[str, ...] = ("tests",)
    exclude_roots: tuple[str, ...] = ("AgentSwarm", ".git", ".agentswarm/runs", ".agentswarm/memory")
    memory_namespaces: tuple[str, ...] = ("shared", "project", "agentswarm")


@dataclass(frozen=True)
class ProjectManifest:
    modules: tuple[str, ...] = ()
    services: tuple[str, ...] = ()
    entrypoints: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    owners: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)


def _merge_root_lists(base: tuple[str, ...], additions: tuple[str, ...]) -> tuple[str, ...]:
    merged = list(base)
    seen = {item.replace("\\", "/").strip("/").lower() for item in base}
    for item in additions:
        normalized = item.replace("\\", "/").strip("/").lower()
        if normalized in seen:
            continue
        merged.append(item)
        seen.add(normalized)
    return tuple(merged)


def _looks_like_unreal_project(host_root: Path) -> bool:
    try:
        if any(host_root.glob("*.uproject")):
            return True
    except OSError:
        return False
    return (host_root / "Source").exists() and (host_root / "Content").exists()


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return content


def _to_tuple_of_strings(value: Any, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return fallback
    if not isinstance(value, list):
        raise ValueError(f"Expected a YAML list, got: {type(value).__name__}")
    items = [str(item).strip() for item in value if str(item).strip()]
    return tuple(items) or fallback


def load_agentswarm_config(paths: RuntimePaths) -> AgentSwarmConfig:
    data = _load_yaml_file(paths.config_path)
    defaults = AgentSwarmConfig()
    source_roots = _to_tuple_of_strings(data.get("source_roots"), defaults.source_roots)
    doc_roots = _to_tuple_of_strings(data.get("doc_roots"), defaults.doc_roots)
    test_roots = _to_tuple_of_strings(data.get("test_roots"), defaults.test_roots)
    exclude_roots = _to_tuple_of_strings(data.get("exclude_roots"), defaults.exclude_roots)

    if _looks_like_unreal_project(paths.host_root):
        source_roots = _merge_root_lists(source_roots, ("Source", "Plugins"))
        doc_roots = _merge_root_lists(doc_roots, ("docs/architecture",))
        test_roots = _merge_root_lists(test_roots, ("Source",))
        exclude_roots = _merge_root_lists(
            exclude_roots,
            (
                "Intermediate",
                "Binaries",
                "DerivedDataCache",
                "Saved",
                "Plugins/Marketplace",
                "node_modules",
                ".vs",
                ".idea",
            ),
        )

    return AgentSwarmConfig(
        version=int(data.get("version", defaults.version)),
        agent_root_name=str(data.get("agent_root", defaults.agent_root_name)),
        target_scope=str(data.get("target_scope", defaults.target_scope)),
        workflow_sources=_to_tuple_of_strings(data.get("workflow_sources"), defaults.workflow_sources),
        tool_sources=_to_tuple_of_strings(data.get("tool_sources"), defaults.tool_sources),
        source_roots=source_roots,
        doc_roots=doc_roots,
        test_roots=test_roots,
        exclude_roots=exclude_roots,
        memory_namespaces=_to_tuple_of_strings(data.get("memory_namespaces"), defaults.memory_namespaces),
    )


def load_project_manifest(paths: RuntimePaths) -> ProjectManifest:
    data = _load_yaml_file(paths.manifest_path)

    def to_tuple(key: str) -> tuple[str, ...]:
        value = data.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"{paths.manifest_path} key '{key}' must be a YAML list")
        return tuple(str(item).strip() for item in value if str(item).strip())

    return ProjectManifest(
        modules=to_tuple("modules"),
        services=to_tuple("services"),
        entrypoints=to_tuple("entrypoints"),
        keywords=to_tuple("keywords"),
        owners=to_tuple("owners"),
        raw=data,
    )


def resolve_host_roots(host_root: Path, relative_roots: tuple[str, ...]) -> list[Path]:
    return [host_root / relative_path for relative_path in relative_roots]
