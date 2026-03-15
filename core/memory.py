from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryNamespace:
    name: str
    root: Path


def get_memory_namespace(memory_root: Path, namespace: str) -> MemoryNamespace:
    path = memory_root / namespace
    path.mkdir(parents=True, exist_ok=True)
    return MemoryNamespace(name=namespace, root=path)


def write_memory_summary(graph, config: dict[str, object], run_dir: Path) -> None:
    snapshot = graph.get_state(config)
    history = list(graph.get_state_history(config))
    configurable = snapshot.config.get("configurable", {})
    tasks = snapshot.values.get("tasks", []) if isinstance(snapshot.values, dict) else []

    summary = {
        "thread_id": configurable.get("thread_id", ""),
        "memory_checkpoint_id": configurable.get("checkpoint_id", ""),
        "history_length": len(history),
        "latest_step": snapshot.metadata.get("step") if isinstance(snapshot.metadata, dict) else None,
        "created_at": snapshot.created_at,
        "next_nodes": list(snapshot.next),
        "task_count": len(tasks) if isinstance(tasks, list) else 0,
    }

    lines = [
        "# Memory Summary",
        "",
        f"Thread ID: {summary['thread_id']}",
        f"Memory Checkpoint ID: {summary['memory_checkpoint_id']}",
        f"History Length: {summary['history_length']}",
        f"Latest Step: {summary['latest_step']}",
        f"Created At: {summary['created_at']}",
        f"Next Nodes: {', '.join(summary['next_nodes']) or '(completed)'}",
        f"Task Count: {summary['task_count']}",
        "",
        "## State Keys",
    ]
    state_keys = sorted(snapshot.values) if isinstance(snapshot.values, dict) else []
    lines.extend([f"- {key}" for key in state_keys] or ["- (none)"])

    (run_dir / "memory_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "memory_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
