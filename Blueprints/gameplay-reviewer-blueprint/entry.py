from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_logging import trace_graph_node
from core.llm import LLMError
from core.models import BlueprintContext, BlueprintMetadata


class ReviewerState(TypedDict):
    task_prompt: str
    plan_doc: str
    review_round: int
    run_dir: NotRequired[str]
    task_id: NotRequired[str]
    score: int
    feedback: str
    missing_sections: list[str]
    approved: bool
    summary: str


SECTION_WEIGHTS = {
    "Overview": 15,
    "Task Type": 15,
    "Existing Docs": 10,
    "Implementation Steps": 25,
    "Unit Tests": 20,
    "Risks": 5,
    "Acceptance Criteria": 10,
}


def _detect_missing_sections(plan_doc: str) -> list[str]:
    return [section for section in SECTION_WEIGHTS if f"## {section}" not in plan_doc]


def _fallback_review(task_prompt: str, plan_doc: str, review_round: int) -> dict[str, Any]:
    missing_sections = _detect_missing_sections(plan_doc)
    score = 100 - sum(SECTION_WEIGHTS[section] for section in missing_sections)
    score = max(0, min(score, 100))

    feedback_lines = [
        f"Review round {review_round} for task: {task_prompt}",
    ]
    if missing_sections:
        feedback_lines.append("Missing sections that block approval:")
        feedback_lines.extend(f"- {section}" for section in missing_sections)
        if "Unit Tests" in missing_sections:
            feedback_lines.append("- Include concrete unit tests before implementation can start.")
    else:
        feedback_lines.append("Plan is complete and ready for implementation.")

    approved = score == 100
    return {
        "score": score,
        "feedback": "\n".join(feedback_lines),
        "missing_sections": missing_sections,
        "approved": approved,
        "summary": f"Reviewer scored the plan at {score}/100.",
    }


def build_graph(context: BlueprintContext, metadata: BlueprintMetadata):
    graph_name = metadata.name
    del metadata

    def review_plan(state: ReviewerState) -> dict[str, Any]:
        fallback = _fallback_review(state["task_prompt"], state["plan_doc"], state["review_round"])
        llm = context.llm
        if not llm.is_enabled():
            return fallback

        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 100},
                "feedback": {"type": "string"},
                "missing_sections": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(SECTION_WEIGHTS)},
                    "uniqueItems": True,
                },
                "approved": {"type": "boolean"},
            },
            "required": ["score", "feedback", "missing_sections", "approved"],
            "additionalProperties": False,
        }
        try:
            result = llm.generate_json(
                instructions=(
                    "You are gameplay-reviewer-blueprint. Review gameplay implementation plans. "
                    "Score the plan out of 100, list missing required sections, and provide direct, actionable feedback. "
                    "A plan must include Overview, Task Type, Existing Docs, Implementation Steps, Unit Tests, Risks, "
                    "and Acceptance Criteria. Do not approve if required sections are missing."
                ),
                input_text=(
                    f"Task prompt:\n{state['task_prompt']}\n\n"
                    f"Review round: {state['review_round']}\n\n"
                    f"Required sections and weights:\n{SECTION_WEIGHTS}\n\n"
                    f"Plan document:\n{state['plan_doc']}"
                ),
                schema_name="gameplay_plan_review",
                schema=schema,
            )
        except LLMError:
            return fallback

        deterministic_missing = set(fallback["missing_sections"])
        reported_missing = set(result["missing_sections"])
        missing_sections = sorted(deterministic_missing | reported_missing)
        if missing_sections:
            capped_score = 100 - sum(SECTION_WEIGHTS[section] for section in missing_sections)
            score = min(int(result["score"]), capped_score)
            approved = False
        else:
            score = 100
            approved = True

        feedback = str(result["feedback"]).strip() or fallback["feedback"]
        return {
            "score": max(0, min(score, 100)),
            "feedback": feedback,
            "missing_sections": missing_sections,
            "approved": approved,
            "summary": f"Reviewer scored the plan at {max(0, min(score, 100))}/100.",
        }

    graph = StateGraph(ReviewerState)
    graph.add_node(
        "review_plan",
        trace_graph_node(graph_name=graph_name, node_name="review_plan", node_fn=review_plan),
    )
    graph.add_edge(START, "review_plan")
    graph.add_edge("review_plan", END)
    return graph
