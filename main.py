from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

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
from core.llm import LLMManager, LLMError, ensure_traced_llm_client
from core.main_graph import build_initial_state, build_main_graph, build_runtime_config
from core.runtime_paths import RuntimePaths
from core.tool_loader import load_tools
from core.tool_engine import ToolEngine, ToolEngineConfig
from core.agent_task_runner import AgentTaskRunnerConfig, RunnerError, run_agent_task_loop


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
    parser.add_argument(
        "--mode",
        choices=["workflow", "tool", "agent-task"],
        default="workflow",
        help="Execution mode. 'workflow' routes through the full pipeline. 'tool' runs tools directly via LLM selection. 'agent-task' processes GitHub AgentTask issues.",
    )
    parser.add_argument(
        "--tools",
        default="",
        help="Comma-separated tool names to restrict tool mode to (e.g. 'find-gameplay-code,load-source-context').",
    )
    parser.add_argument("--once", action="store_true", help="AgentTask mode: scan once and exit.")
    parser.add_argument("--poll", action="store_true", help="AgentTask mode: continuously poll for Todo AgentTask issues.")
    parser.add_argument("--interval-seconds", type=int, default=60, help="AgentTask mode polling interval.")
    parser.add_argument("--dry-run", action="store_true", help="AgentTask mode: report intended actions without mutating GitHub or git.")
    parser.add_argument("--agent-task-repo", default=os.getenv("AGENT_TASK_REPO", "sipherxyz/s2"), help="AgentTask mode GitHub repository.")
    parser.add_argument("--agent-task-label", default=os.getenv("AGENT_TASK_LABEL", "AgentTask"), help="AgentTask mode issue label filter.")
    parser.add_argument("--project-owner", default=os.getenv("AGENT_TASK_PROJECT_OWNER", "sipherxyz"), help="AgentTask mode GitHub Projects owner.")
    parser.add_argument("--project-number", type=int, default=int(os.getenv("AGENT_TASK_PROJECT_NUMBER", "5")), help="AgentTask mode GitHub Projects v2 number.")
    parser.add_argument("--codex-profile", default=os.getenv("AGENT_TASK_CODEX_PROFILE", os.getenv("CODEX_PROFILE", "ai-gateway")), help="AgentTask mode Codex profile.")
    parser.add_argument("--codex-model", default=os.getenv("AGENT_TASK_CODEX_MODEL", os.getenv("CODEX_MODEL", "gpt-5.5")), help="AgentTask mode Codex model.")
    parser.add_argument("--reasoning-effort", default=os.getenv("AGENT_TASK_REASONING_EFFORT", os.getenv("CODEX_REASONING_EFFORT", "xhigh")), help="AgentTask mode Codex reasoning effort.")
    parser.add_argument("--codex-sandbox", default=os.getenv("AGENT_TASK_CODEX_SANDBOX", os.getenv("CODEX_SANDBOX", "danger-full-access")), help="AgentTask mode Codex sandbox.")
    parser.add_argument("--codex-timeout-seconds", type=int, default=int(os.getenv("AGENT_TASK_CODEX_TIMEOUT_SECONDS", "1800")), help="AgentTask mode Codex timeout.")
    parser.add_argument("--max-tasks", type=int, default=0, help="AgentTask mode max tasks to process in one scan; 0 means no limit.")
    parser.add_argument("prompt_parts", nargs="*", help="Optional prompt parts when --prompt is omitted")
    return parser.parse_args()


def _build_tool_catalog(runtimes: list) -> str:
    """Build a human-readable catalog of available tools for LLM screening."""
    lines: list[str] = []
    for rt in runtimes:
        m = rt.metadata
        caps = ", ".join(m.capabilities) if m.capabilities else "general"
        lines.append(f"- **{m.name}** (capabilities: {caps}): {m.description[:200]}")
    return "\n".join(lines)


def _screen_tools(
    prompt: str,
    runtimes: list,
    llm: Any,
) -> tuple[bool, str, list[str]]:
    """LLM decides if any loaded tools are relevant to the task.

    Returns (should_proceed, reasoning, recommended_tool_names).
    On LLM failure, falls back to proceeding with all tools.
    """
    catalog = _build_tool_catalog(runtimes)
    tool_names = [rt.metadata.name for rt in runtimes]

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "should_proceed": {
                "type": "boolean",
                "description": "True if one or more tools can help with the task, false if none are relevant.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why tools match or don't match.",
            },
            "recommended_tools": {
                "type": "array",
                "items": {"type": "string", "enum": tool_names},
                "description": "Tool names that are relevant. Empty if should_proceed is false.",
            },
        },
        "required": ["should_proceed", "reasoning", "recommended_tools"],
        "additionalProperties": False,
    }

    try:
        result = llm.generate_json(
            instructions=(
                "You are a tool selection screener. Given a user task and a catalog of available tools, "
                "decide whether any tools can help accomplish the task. "
                "Only recommend tools whose capabilities genuinely match the task. "
                "If no tools are relevant, set should_proceed to false."
            ),
            input_text=f"## Task\n{prompt}\n\n## Available Tools\n{catalog}",
            schema_name="tool_screening",
            schema=schema,
        )
    except (LLMError, Exception):
        # Fallback: proceed with all tools and let ToolEngine sort it out
        return True, "LLM screening unavailable, proceeding with all tools.", tool_names

    should_proceed = bool(result.get("should_proceed", True))
    reasoning = str(result.get("reasoning", ""))
    recommended = [name for name in result.get("recommended_tools", []) if name in tool_names]

    return should_proceed, reasoning, recommended


def _run_tool_mode(
    prompt: str,
    runtime_paths: RuntimePaths,
    config: Any,
    manifest: Any,
    llm_manager: LLMManager,
    run_dir: Path,
    tools_filter: str,
) -> None:
    """Run tools directly via LLM-driven selection, bypassing workflow routing."""
    registry = load_tools(
        project_root=runtime_paths.agent_root,
        tools_root=runtime_paths.built_in_tools_root,
        llm_manager=llm_manager,
        runtime_paths=runtime_paths,
        config=config,
        manifest=manifest,
    )
    runtimes = registry.list_runtimes()

    # Apply explicit name filter if provided
    if tools_filter.strip():
        allowed = {name.strip() for name in tools_filter.split(",")}
        runtimes = [rt for rt in runtimes if rt.metadata.name in allowed]

    if not runtimes:
        print("No tools available" + (" matching filter." if tools_filter.strip() else "."))
        return

    # Screen tools for relevance (skip if user explicitly filtered)
    if not tools_filter.strip():
        llm = ensure_traced_llm_client(llm_manager.resolve(None))
        should_proceed, reasoning, recommended = _screen_tools(prompt, runtimes, llm)
        if not should_proceed:
            print(f"No suitable tools for this task.\n{reasoning}")
            return
        # Narrow to recommended tools if the LLM suggested a subset
        if recommended:
            recommended_set = set(recommended)
            narrowed = [rt for rt in runtimes if rt.metadata.name in recommended_set]
            if narrowed:
                runtimes = narrowed

    tools = [rt.tool for rt in runtimes]
    engine = ToolEngine(
        config=ToolEngineConfig(
            system_id="cli-tool-runner",
            persona="You are a tool-calling agent running in direct tool mode. Use the available tools to accomplish the task.",
            max_turns=10,
            require_tool_use=True,
        ),
        tools=tools,
        llm=ensure_traced_llm_client(llm_manager.resolve(None)),
    )

    result = engine.gather(task=prompt)

    # Write tool call log
    log_path = run_dir / "tool_run.json"
    log_data = {
        "prompt": prompt,
        "tools_used": [tc.tool_name for tc in result.tool_calls],
        "turns_used": result.turns_used,
        "success": result.success,
        "tool_calls": [
            {
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
                "result": tc.result[:1000],
                "error": tc.error,
                "turn": tc.turn,
            }
            for tc in result.tool_calls
        ],
        "summary": result.summary,
    }
    log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    print(result.summary)
    print(f"\nArtifacts: {run_dir}")
    print(f"Tool log: {log_path}")


def main() -> None:
    args = _parse_args()
    prompt = args.prompt.strip() or " ".join(args.prompt_parts).strip()
    if args.mode != "agent-task" and not prompt:
        raise SystemExit("A prompt is required. Use --prompt \"...\" or pass a positional prompt.")

    agent_root = Path(__file__).resolve().parent
    explicit_host_root = Path(args.host_root).resolve() if args.host_root.strip() else None
    runtime_paths, _ = initialize_host_project(agent_root=agent_root, host_root=explicit_host_root)
    config = load_agentswarm_config(runtime_paths)
    manifest = load_project_manifest(runtime_paths)
    llm_manager = LLMManager.from_env(working_directory=str(runtime_paths.host_root))

    if args.mode == "agent-task":
        runner_config = AgentTaskRunnerConfig(
            repo=args.agent_task_repo,
            project_owner=args.project_owner,
            project_number=args.project_number,
            label=args.agent_task_label,
            once=bool(args.once or not args.poll),
            poll=bool(args.poll),
            interval_seconds=args.interval_seconds,
            dry_run=bool(args.dry_run),
            codex_profile=args.codex_profile,
            codex_model=args.codex_model,
            reasoning_effort=args.reasoning_effort,
            codex_sandbox=args.codex_sandbox,
            codex_timeout_seconds=args.codex_timeout_seconds,
            max_tasks=args.max_tasks,
        )
        try:
            processed = run_agent_task_loop(
                runtime_paths=runtime_paths,
                config=config,
                manifest=manifest,
                llm_manager=llm_manager,
                runner_config=runner_config,
                run_dir_factory=lambda: _build_run_dir(runtime_paths),
            )
        except RunnerError as exc:
            raise SystemExit(f"AgentTask runner error: {exc}") from exc
        print(f"AgentTask runner processed {processed} task(s).")
        return

    run_dir = _build_run_dir(runtime_paths)

    if args.mode == "tool":
        _run_tool_mode(prompt, runtime_paths, config, manifest, llm_manager, run_dir, args.tools)
        return

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
