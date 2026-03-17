from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from core.graph_ids import to_graph_node_name
from core.graph_logging import trace_graph_node, trace_route_decision
from core.llm import LLMError, LLMManager
from core.registry import WorkflowRegistry
from core.text_utils import normalize_text, slugify


class MainTask(TypedDict):
    id: str
    description: str
    workflow_name: str | None
    status: str
    input: dict[str, Any]
    output: dict[str, Any] | None
    error: str | None


class MainState(TypedDict):
    agent_root: str
    host_root: str
    target_scope: str
    prompt: str
    run_dir: str
    tasks: list[MainTask]
    results: list[dict[str, Any]]
    final_response: str
    routing_notes: list[str]
    active_task_index: int | None
    active_task: MainTask | None
    active_task_error: str
    task_prompt: str
    task_id: str
    plan_doc: str
    review_round: int
    summary: str
    final_report: dict[str, Any]
    score: int
    feedback: str
    missing_sections: list[str]
    approved: bool
    messages: list[AnyMessage]
    pending_tool_name: str
    pending_tool_call_id: str


def _split_prompt(prompt: str) -> list[str]:
    candidates = [segment.strip() for segment in re.split(r"[\n;]+", prompt) if segment.strip()]
    return candidates or [prompt.strip()]


def _fallback_plan_tasks(prompt: str) -> list[str]:
    return _split_prompt(prompt)


def _prefer_single_task(prompt: str) -> bool:
    normalized = " ".join(prompt.split())
    if not normalized:
        return True
    if "\n" in prompt or ";" in prompt:
        return False

    lowered = normalize_text(f" {normalized} ")
    multi_markers = (
        " compare ",
        " versus ",
        " vs ",
        " and then ",
        " after that ",
        " separately ",
        " step 1",
        " step 2",
        " first,",
        " second,",
    )
    return not any(marker in lowered for marker in multi_markers)


def _compact_task_id(index: int, description: str) -> str:
    slug = slugify(description, fallback="request")
    digest = hashlib.sha1(normalize_text(description).encode("utf-8")).hexdigest()[:6]
    if len(slug) > 18:
        slug = slug[:18].rstrip("-")
    return f"task-{index}-{slug}-{digest}"


def _fallback_route_task(registry: WorkflowRegistry, description: str) -> str | None:
    match = registry.route(description)
    return match.qualified_name if match else None


def _llm_plan_tasks(llm, prompt: str, workspace_context: str = "") -> list[str]:
    schema = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                    },
                    "required": ["description"],
                    "additionalProperties": False,
                },
                "minItems": 1,
            }
        },
        "required": ["tasks"],
        "additionalProperties": False,
    }
    result = llm.generate_json(
        instructions=(
            "You are the planning step of a workflow-driven software agent called AgentSwarm. "
            "Break the user's request into 1 to 5 implementation tasks. "
            "Return concise task descriptions that can each be routed to one workflow. "
            "Operate on the host project, not on the AgentSwarm engine internals, unless the user explicitly asks to modify AgentSwarm."
        ),
        input_text=f"{workspace_context}User prompt:\n{prompt}",
        schema_name="main_graph_task_plan",
        schema=schema,
    )
    tasks = [item["description"].strip() for item in result["tasks"] if item["description"].strip()]
    return tasks or _fallback_plan_tasks(prompt)


def _llm_route_tasks(
    llm,
    registry: WorkflowRegistry,
    tasks: list[MainTask],
    workspace_context: str = "",
) -> dict[str, str | None]:
    candidates = registry.list_metadata(exposed_only=True, include_shadowed=False)
    if not candidates:
        return {}

    candidate_names = [item.qualified_name for item in candidates]
    candidate_descriptions = "\n\n".join(
        [
            (
                f"Workflow: {item.qualified_name}\n"
                f"Short Name: {item.name}\n"
                f"Namespace: {item.namespace}\n"
                f"Capabilities: {', '.join(item.capabilities)}\n"
                f"Description: {item.description}"
            )
            for item in candidates
        ]
    )
    schema = {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "workflow_name": {"type": "string", "enum": candidate_names},
                    },
                    "required": ["task_id", "workflow_name"],
                    "additionalProperties": False,
                },
                "minItems": 1,
            }
        },
        "required": ["assignments"],
        "additionalProperties": False,
    }
    task_block = "\n".join([f"- {task['id']}: {task['description']}" for task in tasks])
    result = llm.generate_json(
        instructions=(
            "You route implementation tasks to the best matching workflow. "
            "Use each workflow's description and capabilities, and assign every task to exactly one workflow. "
            "Default to workflows that operate on the host project."
        ),
        input_text=f"{workspace_context}Available workflows:\n{candidate_descriptions}\n\nTasks:\n{task_block}",
        schema_name="main_graph_routes",
        schema=schema,
    )
    return {item["task_id"]: item["workflow_name"] for item in result["assignments"]}


def _write_run_summary(run_dir: Path, state: MainState) -> None:
    lines = [
        "# Execution Summary",
        "",
        f"Prompt: {state['prompt']}",
        "",
        "## Tasks",
    ]

    for task in state["tasks"]:
        lines.extend(
            [
                f"- id: {task['id']}",
                f"  description: {task['description']}",
                f"  workflow: {task['workflow_name']}",
                f"  status: {task['status']}",
            ]
        )

    lines.extend(["", "## Final Response", "", state["final_response"]])
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _reset_active_task_fields() -> dict[str, Any]:
    return {
        "active_task_index": None,
        "active_task": None,
        "active_task_error": "",
        "task_prompt": "",
        "task_id": "",
        "plan_doc": "",
        "review_round": 0,
        "summary": "",
        "final_report": {},
        "score": 0,
        "feedback": "",
        "missing_sections": [],
        "approved": False,
        "messages": [],
        "pending_tool_name": "",
        "pending_tool_call_id": "",
    }


def _prepare_active_task(task: MainTask, index: int) -> dict[str, Any]:
    task_input = task["input"]
    return {
        "active_task_index": index,
        "active_task": task,
        "active_task_error": "",
        "task_prompt": str(task_input.get("task_prompt") or task["description"]),
        "task_id": str(task_input.get("task_id") or task["id"]),
        "plan_doc": str(task_input.get("plan_doc") or ""),
        "review_round": int(task_input.get("review_round") or 0),
        "summary": "",
        "final_report": {},
        "score": 0,
        "feedback": "",
        "missing_sections": [],
        "approved": False,
        "messages": [],
        "pending_tool_name": "",
        "pending_tool_call_id": "",
    }


def _extract_workflow_output(state: MainState, workflow_name: str | None) -> dict[str, Any] | None:
    short_workflow_name = str(workflow_name or "").split("::", 1)[-1]
    if short_workflow_name == "gameplay-reviewer-workflow":
        return {
            "score": state["score"],
            "feedback": state["feedback"],
            "missing_sections": state["missing_sections"],
            "approved": state["approved"],
            "summary": state["summary"],
        }

    if not state["summary"] and not state["final_report"]:
        return None

    return {
        "final_report": state["final_report"],
        "summary": state["summary"],
    }


def build_initial_state(
    prompt: str,
    run_dir: str,
    *,
    agent_root: str = "",
    host_root: str = "",
    target_scope: str = "host_project",
) -> MainState:
    return {
        "agent_root": agent_root,
        "host_root": host_root,
        "target_scope": target_scope,
        "prompt": prompt,
        "run_dir": run_dir,
        "tasks": [],
        "results": [],
        "final_response": "",
        "routing_notes": [],
        **_reset_active_task_fields(),
    }


def build_runtime_config(thread_id: str) -> dict[str, Any]:
    cleaned_thread_id = thread_id.strip()
    if not cleaned_thread_id:
        raise ValueError("thread_id must not be empty")
    return {"configurable": {"thread_id": cleaned_thread_id}}


def build_main_graph(
    registry: WorkflowRegistry,
    llm_manager: LLMManager,
    checkpointer: Any | None = None,
    *,
    runtime_paths: Any | None = None,
    config: Any | None = None,
):
    graph_name = "main_graph"
    host_root = str(getattr(runtime_paths, "host_root", "") or "")
    agent_root = str(getattr(runtime_paths, "agent_root", "") or "")
    target_scope = str(getattr(config, "target_scope", "host_project"))
    workspace_context = ""
    if host_root or agent_root:
        workspace_context = (
            f"Host project root: {host_root or '(unknown)'}\n"
            f"AgentSwarm engine root: {agent_root or '(unknown)'}\n"
            f"Target scope: {target_scope}\n\n"
        )
    workflow_subgraphs = {
        metadata.qualified_name: registry.get(metadata.qualified_name).graph
        for metadata in registry.list_metadata()
        if registry.get(metadata.qualified_name).graph is not None
    }
    workflow_node_names = {
        workflow_name: to_graph_node_name(workflow_name)
        for workflow_name in workflow_subgraphs
    }

    def analyze_prompt(state: MainState) -> dict[str, Any]:
        return {
            "routing_notes": [
                f"Loaded {len(registry.list_metadata())} workflows across AgentSwarm and project sources",
                f"Default Codex profile is {llm_manager.describe()}",
                f"Available LLM profiles: {', '.join(llm_manager.available_profiles())}",
                f"Target scope is {state['target_scope'] or target_scope}",
                f"Host project root is {state['host_root'] or host_root or '(unknown)'}",
                f"AgentSwarm engine root is {state['agent_root'] or agent_root or '(unknown)'}",
                "Prompt was normalized and ready for planning",
            ]
        }

    def plan_tasks(state: MainState) -> dict[str, Any]:
        planning_notes: list[str] = []
        descriptions = _fallback_plan_tasks(state["prompt"])
        planner_llm = llm_manager.resolve("planner")
        if planner_llm.is_enabled():
            try:
                descriptions = _llm_plan_tasks(planner_llm, state["prompt"], workspace_context)
                planning_notes.append(f"Task planning used {llm_manager.describe('planner')}.")
            except LLMError as exc:
                planning_notes.append(f"Planner fallback: {exc}")
        else:
            planning_notes.append("Task planning used deterministic fallback.")

        if len(descriptions) > 1 and _prefer_single_task(state["prompt"]):
            descriptions = [state["prompt"].strip()]
            planning_notes.append("Collapsed planner output to one task because the prompt describes a single request.")

        tasks: list[MainTask] = []
        for index, description in enumerate(descriptions, start=1):
            task_id = _compact_task_id(index, description)
            tasks.append(
                {
                    "id": task_id,
                    "description": description,
                    "workflow_name": None,
                    "status": "planned",
                    "input": {
                        "prompt": state["prompt"],
                        "task_prompt": description,
                        "task_id": task_id,
                        "run_dir": state["run_dir"],
                    },
                    "output": None,
                    "error": None,
                }
            )
        return {"tasks": tasks, "routing_notes": [*state["routing_notes"], *planning_notes]}

    def route_tasks(state: MainState) -> dict[str, Any]:
        routed_tasks: list[MainTask] = []
        notes = list(state["routing_notes"])
        llm_assignments: dict[str, str | None] = {}
        router_llm = llm_manager.resolve("router")
        if router_llm.is_enabled() and state["tasks"]:
            try:
                llm_assignments = _llm_route_tasks(router_llm, registry, state["tasks"], workspace_context)
                notes.append(f"Task routing used {llm_manager.describe('router')}.")
            except LLMError as exc:
                notes.append(f"Router fallback: {exc}")

        for task in state["tasks"]:
            task_copy = dict(task)
            workflow_name = llm_assignments.get(task["id"]) or _fallback_route_task(registry, task["description"])
            if workflow_name is None:
                task_copy["status"] = "unroutable"
                task_copy["error"] = "No matching workflow was found"
                notes.append(f"{task['id']} could not be routed")
            else:
                task_copy["workflow_name"] = workflow_name
                task_copy["status"] = "routed"
                notes.append(f"{task['id']} -> {workflow_name}")
            routed_tasks.append(task_copy)
        return {"tasks": routed_tasks, "routing_notes": notes}

    def select_next_task(state: MainState) -> dict[str, Any]:
        for index, task in enumerate(state["tasks"]):
            if task["status"] == "routed":
                return _prepare_active_task(task, index)
        return _reset_active_task_fields()

    def dispatch_active_task(state: MainState) -> str:
        active_task = state["active_task"]
        if active_task is None:
            return "finalize"

        workflow_name = active_task["workflow_name"]
        if workflow_name in workflow_subgraphs:
            return workflow_node_names[str(workflow_name)]
        return "mark_task_failed"

    def mark_task_failed(state: MainState) -> dict[str, Any]:
        active_task = state["active_task"]
        if active_task is None:
            return {}
        workflow_name = active_task["workflow_name"] or "unknown-workflow"
        return {"active_task_error": f"Workflow runtime unavailable: {workflow_name}"}

    def collect_task_result(state: MainState) -> dict[str, Any]:
        active_task = state["active_task"]
        active_task_index = state["active_task_index"]
        if active_task is None or active_task_index is None:
            return {}

        updated_tasks = list(state["tasks"])
        task_copy = dict(updated_tasks[active_task_index])
        updated_results = list(state["results"])
        notes = list(state["routing_notes"])

        if state["active_task_error"]:
            task_copy["status"] = "failed"
            task_copy["error"] = state["active_task_error"]
            notes.append(f"{task_copy['id']} failed: {state['active_task_error']}")
        else:
            task_copy["status"] = "completed"
            task_copy["error"] = None
            task_copy["output"] = _extract_workflow_output(state, task_copy["workflow_name"])
            if task_copy["output"] is not None:
                updated_results.append(
                    {
                        "task_id": task_copy["id"],
                        "workflow_name": task_copy["workflow_name"],
                        "result": task_copy["output"],
                    }
                )
            notes.append(f"{task_copy['id']} completed via {task_copy['workflow_name']}")

        updated_tasks[active_task_index] = task_copy
        return {
            "tasks": updated_tasks,
            "results": updated_results,
            "routing_notes": notes,
            **_reset_active_task_fields(),
        }

    def finalize(state: MainState) -> dict[str, Any]:
        lines = [
            "AgentSwarm workflow-driven execution finished.",
            "",
            "Routing notes:",
            *[f"- {note}" for note in state["routing_notes"]],
            "",
            "Task results:",
        ]
        for task in state["tasks"]:
            lines.append(
                f"- {task['id']} [{task['status']}] via {task['workflow_name'] or 'none'}"
            )
            if task["output"]:
                summary = task["output"].get("summary") or task["output"].get("final_report", "")
                if summary:
                    lines.append(f"  {summary}")
            if task["error"]:
                lines.append(f"  error: {task['error']}")

        final_response = "\n".join(lines)
        run_dir = Path(state["run_dir"])
        _write_run_summary(
            run_dir,
            {
                **state,
                "final_response": final_response,
            },
        )
        return {"final_response": final_response}

    graph = StateGraph(MainState)
    graph.add_node(
        "analyze_prompt",
        trace_graph_node(graph_name=graph_name, node_name="analyze_prompt", node_fn=analyze_prompt),
    )
    graph.add_node(
        "plan_tasks",
        trace_graph_node(graph_name=graph_name, node_name="plan_tasks", node_fn=plan_tasks),
    )
    graph.add_node(
        "route_tasks",
        trace_graph_node(graph_name=graph_name, node_name="route_tasks", node_fn=route_tasks),
    )
    graph.add_node(
        "select_next_task",
        trace_graph_node(graph_name=graph_name, node_name="select_next_task", node_fn=select_next_task),
    )
    graph.add_node(
        "mark_task_failed",
        trace_graph_node(graph_name=graph_name, node_name="mark_task_failed", node_fn=mark_task_failed),
    )
    graph.add_node(
        "collect_task_result",
        trace_graph_node(graph_name=graph_name, node_name="collect_task_result", node_fn=collect_task_result),
    )
    graph.add_node(
        "finalize",
        trace_graph_node(graph_name=graph_name, node_name="finalize", node_fn=finalize),
    )
    for workflow_name, subgraph in workflow_subgraphs.items():
        graph.add_node(workflow_node_names[workflow_name], subgraph)

    graph.add_edge(START, "analyze_prompt")
    graph.add_edge("analyze_prompt", "plan_tasks")
    graph.add_edge("plan_tasks", "route_tasks")
    graph.add_edge("route_tasks", "select_next_task")
    graph.add_conditional_edges(
        "select_next_task",
        trace_route_decision(
            graph_name=graph_name,
            router_name="dispatch_active_task",
            route_fn=dispatch_active_task,
        ),
        {
            **{
                workflow_node_names[workflow_name]: workflow_node_names[workflow_name]
                for workflow_name in workflow_subgraphs
            },
            "mark_task_failed": "mark_task_failed",
            "finalize": "finalize",
        },
    )
    for workflow_name in workflow_subgraphs:
        graph.add_edge(workflow_node_names[workflow_name], "collect_task_result")
    graph.add_edge("mark_task_failed", "collect_task_result")
    graph.add_edge("collect_task_result", "select_next_task")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
