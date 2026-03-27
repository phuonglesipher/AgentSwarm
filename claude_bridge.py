"""JSON entry point for Claude Code to invoke AgentSwarm orchestration.

Usage:
    python claude_bridge.py --prompt "fix the dodge cancel bug" --host-root "D:/s2" --output-format json

Returns structured JSON to stdout so Claude Code can parse and present results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env from AgentSwarm root so LLM_PROVIDER and model configs are picked up
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)

from langgraph.checkpoint.memory import InMemorySaver

from core.config_loader import load_agentswarm_config, load_project_manifest
from core.host_setup import initialize_host_project
from core.memory import write_memory_summary
from core.workflow_loader import load_workflows
from core.graph_logging import GRAPH_TIMELINE_FILE, GRAPH_TRACE_FILE, LLM_PROMPT_TRACE_FILE
from core.llm import LLMManager
from core.main_graph import build_initial_state, build_main_graph, build_runtime_config
from core.runtime_paths import RuntimePaths


def _build_run_dir(paths: RuntimePaths) -> Path:
    paths.runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = paths.runs_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentSwarm bridge for Claude Code")
    parser.add_argument("--prompt", required=True, help="Task to execute")
    parser.add_argument("--host-root", default="", help="Host project root directory")
    parser.add_argument(
        "--output-format",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument("--thread-id", default="", help="Optional LangGraph thread id")
    return parser.parse_args()


def _extract_task_summaries(result: dict) -> list[dict]:
    """Extract task status summaries from the graph result."""
    summaries = []
    for task in result.get("tasks", []):
        entry = {
            "id": task.get("id", ""),
            "description": task.get("description", ""),
            "workflow": task.get("workflow_name"),
            "status": task.get("status", "unknown"),
        }
        if task.get("output"):
            output = task["output"]
            if isinstance(output, dict):
                entry["score"] = output.get("review_score") or output.get("score")
                entry["approved"] = output.get("review_approved") or output.get("approved")
        if task.get("error"):
            entry["error"] = task["error"]
        summaries.append(entry)
    return summaries


def main() -> None:
    args = _parse_args()
    output_format = args.output_format

    try:
        prompt = args.prompt.strip()
        if not prompt:
            raise ValueError("A non-empty --prompt is required.")

        agent_root = Path(__file__).resolve().parent
        explicit_host_root = Path(args.host_root).resolve() if args.host_root.strip() else None
        runtime_paths, _ = initialize_host_project(agent_root=agent_root, host_root=explicit_host_root)
        config = load_agentswarm_config(runtime_paths)
        manifest = load_project_manifest(runtime_paths)
        run_dir = _build_run_dir(runtime_paths)
        llm_manager = LLMManager.from_env(working_directory=str(runtime_paths.host_root))
        thread_id = args.thread_id.strip() or run_dir.name
        runtime_config = build_runtime_config(thread_id)
        checkpointer = InMemorySaver()

        registry = load_workflows(
            project_root=runtime_paths.agent_root,
            workflows_root=runtime_paths.built_in_workflows_root,
            llm_manager=llm_manager,
            runtime_paths=runtime_paths,
            config=config,
            manifest=manifest,
        )
        main_graph = build_main_graph(
            registry=registry,
            llm_manager=llm_manager,
            checkpointer=checkpointer,
            runtime_paths=runtime_paths,
            config=config,
        )

        result = main_graph.invoke(
            build_initial_state(
                prompt=prompt,
                run_dir=str(run_dir),
                agent_root=str(runtime_paths.agent_root),
                host_root=str(runtime_paths.host_root),
                target_scope=config.target_scope,
            ),
            runtime_config,
        )
        write_memory_summary(main_graph, runtime_config, run_dir)

        tasks = result.get("tasks", [])
        completed = sum(1 for t in tasks if t.get("status") == "completed")
        total = len(tasks)

        output = {
            "success": True,
            "summary": f"Completed {completed}/{total} tasks",
            "tasks": _extract_task_summaries(result),
            "final_response": result.get("final_response", ""),
            "artifacts_dir": str(run_dir),
            "host_root": str(runtime_paths.host_root),
            "thread_id": thread_id,
        }

    except Exception as exc:
        output = {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    if output_format == "json":
        json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        if output.get("success"):
            print(output["final_response"])
            print(f"\nArtifacts: {output['artifacts_dir']}")
        else:
            print(f"Error: {output['error']}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
