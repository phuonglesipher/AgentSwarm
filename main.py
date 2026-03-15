from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from core.workflow_loader import load_workflows
from core.graph_logging import GRAPH_TRACE_FILE
from core.llm import LLMManager
from core.main_graph import build_initial_state, build_main_graph, build_runtime_config


def _build_run_dir(project_root: Path) -> Path:
    runs_dir = project_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentSwarm workflow-driven LangGraph agent runner")
    parser.add_argument("--prompt", default="", help="Prompt that should be executed by the agent")
    parser.add_argument(
        "--thread-id",
        default="",
        help="Optional LangGraph thread id for checkpointed execution. Defaults to the run folder name.",
    )
    parser.add_argument("prompt_parts", nargs="*", help="Optional prompt parts when --prompt is omitted")
    return parser.parse_args()


def _write_checkpoint_summary(graph, config: dict[str, object], run_dir: Path) -> None:
    snapshot = graph.get_state(config)
    history = list(graph.get_state_history(config))
    configurable = snapshot.config.get("configurable", {})
    tasks = snapshot.values.get("tasks", []) if isinstance(snapshot.values, dict) else []

    summary = {
        "thread_id": configurable.get("thread_id", ""),
        "checkpoint_id": configurable.get("checkpoint_id", ""),
        "history_length": len(history),
        "latest_step": snapshot.metadata.get("step") if isinstance(snapshot.metadata, dict) else None,
        "created_at": snapshot.created_at,
        "next_nodes": list(snapshot.next),
        "task_count": len(tasks) if isinstance(tasks, list) else 0,
    }

    lines = [
        "# Checkpoint Summary",
        "",
        f"Thread ID: {summary['thread_id']}",
        f"Checkpoint ID: {summary['checkpoint_id']}",
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

    (run_dir / "checkpoint_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "checkpoint_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    prompt = args.prompt.strip() or " ".join(args.prompt_parts).strip()
    if not prompt:
        raise SystemExit("A prompt is required. Use --prompt \"...\" or pass a positional prompt.")

    project_root = Path(__file__).resolve().parent
    workflows_root = project_root / "Workflows"
    run_dir = _build_run_dir(project_root)
    llm_manager = LLMManager.from_env()
    thread_id = args.thread_id.strip() or run_dir.name
    runtime_config = build_runtime_config(thread_id)
    checkpointer = InMemorySaver()

    registry = load_workflows(project_root=project_root, workflows_root=workflows_root, llm_manager=llm_manager)
    main_graph = build_main_graph(registry=registry, llm_manager=llm_manager, checkpointer=checkpointer)

    result = main_graph.invoke(build_initial_state(prompt=prompt, run_dir=str(run_dir)), runtime_config)
    _write_checkpoint_summary(main_graph, runtime_config, run_dir)

    print(result["final_response"])
    print(f"Thread ID: {thread_id}")
    print(f"\nArtifacts: {run_dir}")
    print(f"Traversal log: {run_dir / GRAPH_TRACE_FILE}")


if __name__ == "__main__":
    main()
