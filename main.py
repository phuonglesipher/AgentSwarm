from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="AgentSwarm workflow-driven LangGraph agent runner")
    parser.add_argument("--prompt", default="", help="Prompt that should be executed by the agent")
    parser.add_argument(
        "--host-root",
        default="",
        help="Optional explicit host project root. Defaults to the superproject root when AgentSwarm is a submodule.",
    )
    parser.add_argument(
        "--thread-id",
        default="",
        help="Optional LangGraph thread id for memory-backed execution. Defaults to the run folder name.",
    )
    parser.add_argument("prompt_parts", nargs="*", help="Optional prompt parts when --prompt is omitted")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prompt = args.prompt.strip() or " ".join(args.prompt_parts).strip()
    if not prompt:
        raise SystemExit("A prompt is required. Use --prompt \"...\" or pass a positional prompt.")

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

    print(result["final_response"])
    print(f"Thread ID: {thread_id}")
    print(f"Host Project: {runtime_paths.host_root}")
    print(f"AgentSwarm: {runtime_paths.agent_root}")
    print(f"\nArtifacts: {run_dir}")
    print(f"Traversal log: {run_dir / GRAPH_TRACE_FILE}")
    print(f"Timeline log: {run_dir / GRAPH_TIMELINE_FILE}")
    print(f"LLM trace: {run_dir / LLM_PROMPT_TRACE_FILE}")


if __name__ == "__main__":
    main()
