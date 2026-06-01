from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.codex_runner import CodexRunConfig, run_codex_prompt
from core.graph_logging import trace_graph_node
from core.models import WorkflowContext, WorkflowMetadata


class AgentProcessingState(TypedDict):
    task_prompt: str
    task_id: NotRequired[str]
    run_dir: NotRequired[str]
    host_root: NotRequired[str]
    codex_profile: NotRequired[str]
    codex_model: NotRequired[str]
    reasoning_effort: NotRequired[str]
    codex_sandbox: NotRequired[str]
    codex_timeout_seconds: NotRequired[int]
    summary: NotRequired[str]
    final_report: NotRequired[dict[str, Any]]
    active_task_error: NotRequired[str]


def _build_codex_prompt(state: AgentProcessingState, context: WorkflowContext) -> str:
    task_prompt = str(state.get("task_prompt", "")).strip()
    task_id = str(state.get("task_id", "")).strip() or "agent-task"
    host_root = str(state.get("host_root", "")).strip() or str(context.host_root)
    return "\n".join(
        [
            "# Agent Processing Task",
            "",
            f"Task ID: {task_id}",
            f"Workspace: {host_root}",
            "",
            "You are processing a GitHub AgentTask issue. The queue runner already handled GitHub status, comments, branch checkout, and will handle commit/push/PR after you finish.",
            "Focus only on implementing the requested code/docs/assets change in the current workspace and running the relevant validation you can run locally.",
            "Do not create commits, push branches, open PRs, or change GitHub issue/project status.",
            "",
            "## Task Prompt",
            task_prompt,
        ]
    )


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name

    def run_agent_processing(state: AgentProcessingState) -> dict[str, Any]:
        run_dir = Path(str(state.get("run_dir", "")).strip() or context.artifact_root)
        task_id = str(state.get("task_id", "")).strip() or "agent-task"
        artifact_dir = run_dir / "tasks" / task_id / metadata.name
        host_root = str(state.get("host_root", "")).strip() or str(context.host_root)
        config = CodexRunConfig(
            profile=str(state.get("codex_profile", "")).strip() or "ai-gateway",
            model=str(state.get("codex_model", "")).strip() or "gpt-5.5",
            reasoning_effort=str(state.get("reasoning_effort", "")).strip() or "xhigh",
            sandbox=str(state.get("codex_sandbox", "")).strip() or "danger-full-access",
            timeout_seconds=int(state.get("codex_timeout_seconds", 1800) or 1800),
            working_directory=host_root,
        )
        prompt = _build_codex_prompt(state, context)
        (artifact_dir / "prompt.md").parent.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "prompt.md").write_text(prompt, encoding="utf-8")

        try:
            result = run_codex_prompt(prompt, config, artifact_dir)
        except Exception as exc:
            message = f"Codex execution failed before completion: {type(exc).__name__}: {exc}"
            return {
                "active_task_error": message,
                "summary": message,
                "final_report": {
                    "status": "failed",
                    "error": message,
                    "artifact_dir": str(artifact_dir),
                    "codex_profile": config.profile,
                    "codex_model": config.model,
                    "reasoning_effort": config.reasoning_effort,
                },
            }

        status = "completed" if result.success else "failed"
        summary = result.output_text.strip() or result.stderr.strip() or result.stdout.strip()
        if not summary:
            summary = f"Codex exited with code {result.exit_code}."
        final_report = {
            "status": status,
            "exit_code": result.exit_code,
            "summary": summary,
            "artifact_dir": str(artifact_dir),
            "codex_profile": config.profile,
            "codex_model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "output_path": result.output_path,
            "stdout_path": result.stdout_path,
            "stderr_path": result.stderr_path,
        }
        update: dict[str, Any] = {
            "summary": summary,
            "final_report": final_report,
        }
        if not result.success:
            update["active_task_error"] = f"Codex exited with code {result.exit_code}: {summary[:1000]}"
        return update

    graph = StateGraph(AgentProcessingState)
    graph.add_node(
        "run_agent_processing",
        trace_graph_node(graph_name=graph_name, node_name="run_agent_processing", node_fn=run_agent_processing),
    )
    graph.add_edge(START, "run_agent_processing")
    graph.add_edge("run_agent_processing", END)
    return graph
