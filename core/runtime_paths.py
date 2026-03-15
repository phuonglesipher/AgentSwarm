from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class RuntimePaths:
    agent_root: Path
    host_root: Path
    overlay_root: Path
    runs_root: Path
    memory_root: Path
    built_in_workflows_root: Path
    built_in_tools_root: Path
    project_workflows_root: Path
    project_tools_root: Path
    config_path: Path
    manifest_path: Path

    @property
    def is_submodule(self) -> bool:
        return self.agent_root != self.host_root


def _detect_superproject_root(agent_root: Path) -> Path | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-superproject-working-tree"],
            cwd=agent_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    if completed.returncode != 0:
        return None

    output = completed.stdout.strip()
    if not output:
        return None
    return Path(output).resolve()


def resolve_runtime_paths(agent_root: Path, host_root: Path | None = None) -> RuntimePaths:
    resolved_agent_root = agent_root.resolve()

    explicit_host_root = host_root
    if explicit_host_root is None:
        env_host_root = os.environ.get("AGENTSWARM_HOST_ROOT", "").strip()
        if env_host_root:
            explicit_host_root = Path(env_host_root)

    resolved_host_root = (
        explicit_host_root.resolve()
        if explicit_host_root is not None
        else (_detect_superproject_root(resolved_agent_root) or resolved_agent_root)
    )

    overlay_root = resolved_host_root / ".agentswarm"
    return RuntimePaths(
        agent_root=resolved_agent_root,
        host_root=resolved_host_root,
        overlay_root=overlay_root,
        runs_root=overlay_root / "runs",
        memory_root=overlay_root / "memory",
        built_in_workflows_root=resolved_agent_root / "Workflows",
        built_in_tools_root=resolved_agent_root / "Tools",
        project_workflows_root=overlay_root / "Workflows",
        project_tools_root=overlay_root / "Tools",
        config_path=overlay_root / "agentswarm.yaml",
        manifest_path=overlay_root / "project_manifest.yaml",
    )


def ensure_runtime_dirs(paths: RuntimePaths) -> None:
    for path in [
        paths.overlay_root,
        paths.runs_root,
        paths.memory_root,
        paths.project_workflows_root,
        paths.project_tools_root,
    ]:
        path.mkdir(parents=True, exist_ok=True)
