from __future__ import annotations

import argparse
from pathlib import Path

from core.runtime_paths import RuntimePaths, ensure_runtime_dirs, resolve_runtime_paths


_AGENTSWARM_CONFIG_TEMPLATE = """version: 1
agent_root: AgentSwarm
target_scope: host_project
workflow_sources:
  - project
  - agentswarm
tool_sources:
  - project
  - agentswarm
source_roots:
  - src
  - app
  - packages
doc_roots:
  - docs
  - design
test_roots:
  - tests
exclude_roots:
  - AgentSwarm
  - .git
  - .agentswarm/runs
  - .agentswarm/memory
memory_namespaces:
  - shared
  - project
  - agentswarm
"""

_PROJECT_MANIFEST_TEMPLATE = """modules: []
services: []
entrypoints: []
keywords: []
owners: []
"""

_README_TEMPLATE = """# AgentSwarm Host Overlay

This folder stores host-project-specific AgentSwarm assets.

- `agentswarm.yaml`: host runtime config
- `project_manifest.yaml`: high-level project map for routing and search
- `Workflows/Share/`: shared reusable workflows
- `Workflows/GameplayWorkflows/`: gameplay-only workflows
- `Tools/`: project tools
- `memory/`: runtime memory data
- `runs/`: execution artifacts
"""


def _write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def scaffold_host_project(paths: RuntimePaths) -> list[Path]:
    created: list[Path] = []
    ensure_runtime_dirs(paths)

    for namespace in ["shared", "project", "agentswarm"]:
        namespace_dir = paths.memory_root / namespace
        namespace_dir.mkdir(parents=True, exist_ok=True)
        gitkeep = namespace_dir / ".gitkeep"
        if _write_if_missing(gitkeep, ""):
            created.append(gitkeep)

    for path in [
        paths.project_workflows_root / ".gitkeep",
        paths.project_workflows_root / "Share" / ".gitkeep",
        paths.project_workflows_root / "GameplayWorkflows" / ".gitkeep",
        paths.project_tools_root / ".gitkeep",
        paths.runs_root / ".gitkeep",
    ]:
        if _write_if_missing(path, ""):
            created.append(path)

    if _write_if_missing(paths.config_path, _AGENTSWARM_CONFIG_TEMPLATE):
        created.append(paths.config_path)
    if _write_if_missing(paths.manifest_path, _PROJECT_MANIFEST_TEMPLATE):
        created.append(paths.manifest_path)
    if _write_if_missing(paths.overlay_root / "README.md", _README_TEMPLATE):
        created.append(paths.overlay_root / "README.md")

    return created


def initialize_host_project(agent_root: Path, host_root: Path | None = None) -> tuple[RuntimePaths, list[Path]]:
    paths = resolve_runtime_paths(agent_root=agent_root, host_root=host_root)
    if not paths.is_submodule:
        ensure_runtime_dirs(paths)
        return paths, []
    return paths, scaffold_host_project(paths)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize host-project AgentSwarm overlay files.")
    parser.add_argument("--agent-root", default=".", help="Path to the AgentSwarm repo root.")
    parser.add_argument("--host-root", default="", help="Optional explicit host project root.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    agent_root = Path(args.agent_root).resolve()
    host_root = Path(args.host_root).resolve() if args.host_root.strip() else None
    paths, created = initialize_host_project(agent_root=agent_root, host_root=host_root)

    print(f"Agent root: {paths.agent_root}")
    print(f"Host root: {paths.host_root}")
    if not paths.is_submodule:
        print("Submodule mode was not detected. Runtime directories were ensured for standalone mode.")
        return

    if created:
        print("Created host overlay files:")
        for path in created:
            print(f"- {path}")
        return

    print("Host overlay files already exist. No changes were needed.")


if __name__ == "__main__":
    main()
