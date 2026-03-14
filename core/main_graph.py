from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from core.llm import LLMError, LLMManager
from core.registry import BlueprintRegistry
from core.text_utils import slugify


class MainTask(TypedDict):
    id: str
    description: str
    blueprint_name: str | None
    status: str
    input: dict[str, Any]
    output: dict[str, Any] | None
    error: str | None


class MainState(TypedDict):
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


def _split_prompt(prompt: str) -> list[str]:
    candidates = [segment.strip() for segment in re.split(r"[\n;]+", prompt) if segment.strip()]
    return candidates or [prompt.strip()]


def _fallback_plan_tasks(prompt: str) -> list[str]:
    return _split_prompt(prompt)


def _fallback_route_task(registry: BlueprintRegistry, description: str) -> str | None:
    match = registry.route(description)
    return match.name if match else None


def _llm_plan_tasks(llm, prompt: str) -> list[str]:
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
            "You are the planning step of a Blueprint-driven software agent. "
            "Break the user's request into 1 to 5 implementation tasks. "
            "Return concise task descriptions that can each be routed to one blueprint."
        ),
        input_text=f"User prompt:\n{prompt}",
        schema_name="main_graph_task_plan",
        schema=schema,
    )
    tasks = [item["description"].strip() for item in result["tasks"] if item["description"].strip()]
    return tasks or _fallback_plan_tasks(prompt)


def _llm_route_tasks(
    llm,
    registry: BlueprintRegistry,
    tasks: list[MainTask],
) -> dict[str, str | None]:
    candidates = registry.list_metadata(exposed_only=True)
    if not candidates:
        return {}

    candidate_names = [item.name for item in candidates]
    candidate_descriptions = "\n\n".join(
        [
            f"Blueprint: {item.name}\nCapabilities: {', '.join(item.capabilities)}\nDescription: {item.description}"
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
                        "blueprint_name": {"type": "string", "enum": candidate_names},
                    },
                    "required": ["task_id", "blueprint_name"],
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
            "You route implementation tasks to the best matching blueprint. "
            "Use each blueprint's description and capabilities, and assign every task to exactly one blueprint."
        ),
        input_text=f"Available blueprints:\n{candidate_descriptions}\n\nTasks:\n{task_block}",
        schema_name="main_graph_routes",
        schema=schema,
    )
    return {item["task_id"]: item["blueprint_name"] for item in result["assignments"]}


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
                f"  blueprint: {task['blueprint_name']}",
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
    }


def _extract_blueprint_output(state: MainState, blueprint_name: str | None) -> dict[str, Any] | None:
    if blueprint_name == "gameplay-reviewer-blueprint":
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


def build_initial_state(prompt: str, run_dir: str) -> MainState:
    return {
        "prompt": prompt,
        "run_dir": run_dir,
        "tasks": [],
        "results": [],
        "final_response": "",
        "routing_notes": [],
        **_reset_active_task_fields(),
    }


def build_main_graph(registry: BlueprintRegistry, llm_manager: LLMManager):
    blueprint_subgraphs = {
        metadata.name: registry.get(metadata.name).graph
        for metadata in registry.list_metadata()
        if registry.get(metadata.name).graph is not None
    }

    def analyze_prompt(state: MainState) -> dict[str, Any]:
        return {
            "routing_notes": [
                f"Loaded {len(registry.list_metadata())} blueprints",
                f"Default Codex profile is {llm_manager.describe()}",
                f"Available LLM profiles: {', '.join(llm_manager.available_profiles())}",
                "Prompt was normalized and ready for planning",
            ]
        }

    def plan_tasks(state: MainState) -> dict[str, Any]:
        planning_notes: list[str] = []
        descriptions = _fallback_plan_tasks(state["prompt"])
        planner_llm = llm_manager.resolve("planner")
        if planner_llm.is_enabled():
            try:
                descriptions = _llm_plan_tasks(planner_llm, state["prompt"])
                planning_notes.append(f"Task planning used {llm_manager.describe('planner')}.")
            except LLMError as exc:
                planning_notes.append(f"Planner fallback: {exc}")
        else:
            planning_notes.append("Task planning used deterministic fallback.")

        tasks: list[MainTask] = []
        for index, description in enumerate(descriptions, start=1):
            task_id = f"task-{index}-{slugify(description, fallback='request')}"
            tasks.append(
                {
                    "id": task_id,
                    "description": description,
                    "blueprint_name": None,
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
                llm_assignments = _llm_route_tasks(router_llm, registry, state["tasks"])
                notes.append(f"Task routing used {llm_manager.describe('router')}.")
            except LLMError as exc:
                notes.append(f"Router fallback: {exc}")

        for task in state["tasks"]:
            task_copy = dict(task)
            blueprint_name = llm_assignments.get(task["id"]) or _fallback_route_task(registry, task["description"])
            if blueprint_name is None:
                task_copy["status"] = "unroutable"
                task_copy["error"] = "No matching blueprint was found"
                notes.append(f"{task['id']} could not be routed")
            else:
                task_copy["blueprint_name"] = blueprint_name
                task_copy["status"] = "routed"
                notes.append(f"{task['id']} -> {blueprint_name}")
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

        blueprint_name = active_task["blueprint_name"]
        if blueprint_name in blueprint_subgraphs:
            return str(blueprint_name)
        return "mark_task_failed"

    def mark_task_failed(state: MainState) -> dict[str, Any]:
        active_task = state["active_task"]
        if active_task is None:
            return {}
        blueprint_name = active_task["blueprint_name"] or "unknown-blueprint"
        return {"active_task_error": f"Blueprint runtime unavailable: {blueprint_name}"}

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
            task_copy["output"] = _extract_blueprint_output(state, task_copy["blueprint_name"])
            if task_copy["output"] is not None:
                updated_results.append(
                    {
                        "task_id": task_copy["id"],
                        "blueprint_name": task_copy["blueprint_name"],
                        "result": task_copy["output"],
                    }
                )
            notes.append(f"{task_copy['id']} completed via {task_copy['blueprint_name']}")

        updated_tasks[active_task_index] = task_copy
        return {
            "tasks": updated_tasks,
            "results": updated_results,
            "routing_notes": notes,
            **_reset_active_task_fields(),
        }

    def finalize(state: MainState) -> dict[str, Any]:
        lines = [
            "Blueprint-driven execution finished.",
            "",
            "Routing notes:",
            *[f"- {note}" for note in state["routing_notes"]],
            "",
            "Task results:",
        ]
        for task in state["tasks"]:
            lines.append(
                f"- {task['id']} [{task['status']}] via {task['blueprint_name'] or 'none'}"
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
    graph.add_node("analyze_prompt", analyze_prompt)
    graph.add_node("plan_tasks", plan_tasks)
    graph.add_node("route_tasks", route_tasks)
    graph.add_node("select_next_task", select_next_task)
    graph.add_node("mark_task_failed", mark_task_failed)
    graph.add_node("collect_task_result", collect_task_result)
    graph.add_node("finalize", finalize)
    for blueprint_name, subgraph in blueprint_subgraphs.items():
        graph.add_node(blueprint_name, subgraph)

    graph.add_edge(START, "analyze_prompt")
    graph.add_edge("analyze_prompt", "plan_tasks")
    graph.add_edge("plan_tasks", "route_tasks")
    graph.add_edge("route_tasks", "select_next_task")
    graph.add_conditional_edges(
        "select_next_task",
        dispatch_active_task,
        {
            **{blueprint_name: blueprint_name for blueprint_name in blueprint_subgraphs},
            "mark_task_failed": "mark_task_failed",
            "finalize": "finalize",
        },
    )
    for blueprint_name in blueprint_subgraphs:
        graph.add_edge(blueprint_name, "collect_task_result")
    graph.add_edge("mark_task_failed", "collect_task_result")
    graph.add_edge("collect_task_result", "select_next_task")
    graph.add_edge("finalize", END)

    return graph.compile()
