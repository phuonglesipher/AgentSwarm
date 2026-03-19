from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_logging import trace_graph_node
from core.llm import LLMError
from core.models import WorkflowContext, WorkflowMetadata
from core.quality_loop import QualityLoopSpec, evaluate_quality_loop
from core.text_utils import normalize_text, slugify


APPROVAL_SCORE = 90
MAX_REVIEW_ROUNDS = 3
PLAN_REVIEW_SPEC = QualityLoopSpec(
    loop_id="gameplay-plan-review",
    threshold=APPROVAL_SCORE,
    max_rounds=MAX_REVIEW_ROUNDS,
    min_rounds=1,
    require_blocker_free=True,
    require_missing_section_free=True,
    require_explicit_approval=True,
    min_score_delta=1,
    stagnation_limit=2,
)
SECTION_WEIGHTS = (
    ("Overview", 10),
    ("Task Type", 10),
    ("Existing Docs", 10),
    ("Implementation Steps", 25),
    ("Unit Tests", 20),
    ("Risks", 10),
    ("Acceptance Criteria", 15),
)
BLOCKING_SECTIONS = {"Task Type", "Implementation Steps", "Unit Tests", "Acceptance Criteria"}


class SectionReview(TypedDict):
    section: str
    score: int
    status: str
    rationale: str
    action_items: list[str]


class ReviewerState(TypedDict):
    task_prompt: str
    task_type: NotRequired[str]
    execution_track: NotRequired[str]
    task_id: NotRequired[str]
    run_dir: NotRequired[str]
    artifact_dir: NotRequired[str]
    plan_doc: str
    review_round: int
    score: int
    feedback: str
    missing_sections: list[str]
    section_reviews: list[SectionReview]
    blocking_issues: list[str]
    improvement_actions: list[str]
    approved: bool
    loop_status: str
    loop_reason: str
    loop_should_continue: bool
    loop_completed: bool
    loop_stagnated_rounds: int
    review_score: int
    review_feedback: str
    review_missing_sections: list[str]
    review_blocking_issues: list[str]
    review_improvement_actions: list[str]
    review_approved: bool
    review_loop_status: str
    review_loop_reason: str
    review_loop_should_continue: bool
    review_loop_completed: bool
    review_loop_stagnated_rounds: int
    summary: str


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_item in items:
        item = str(raw_item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _short_slug(value: str, *, fallback: str, max_length: int = 18) -> str:
    slug = slugify(value, fallback=fallback)
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(normalize_text(value).encode("utf-8")).hexdigest()[:6]
    keep = max(1, max_length - len(digest) - 1)
    return f"{slug[:keep].rstrip('-') or fallback}-{digest}"


def _artifact_dir(context: WorkflowContext, metadata: WorkflowMetadata, state: ReviewerState) -> Path:
    existing = str(state.get("artifact_dir", "")).strip()
    if existing:
        path = Path(existing)
        path.mkdir(parents=True, exist_ok=True)
        return path

    run_dir = str(state.get("run_dir", "")).strip()
    base_dir = Path(run_dir) if run_dir else Path(context.artifact_root) / "adhoc"
    task_id = str(state.get("task_id", "")).strip() or state["task_prompt"]
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:6]
    task_dir = f"{_short_slug(task_id, fallback='task')}-{digest}"
    path = base_dir / "tasks" / task_dir / metadata.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_sections(document: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    heading: str | None = None
    buffer: list[str] = []
    for line in document.splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections[heading] = "\n".join(buffer).strip()
            heading = line[3:].strip()
            buffer = []
            continue
        if heading is not None:
            buffer.append(line)
    if heading is not None:
        sections[heading] = "\n".join(buffer).strip()
    return sections


def _clean_lines(section_text: str) -> list[str]:
    return [line.strip() for line in section_text.splitlines() if line.strip()]


def _fallback_action(section: str) -> str:
    actions = {
        "Overview": "Clarify the player-facing gameplay goal and nearby systems that must remain stable.",
        "Task Type": "Explain why this work is a bugfix, feature, or maintenance task.",
        "Existing Docs": "List the design, runtime, or implementation references that constrain the change.",
        "Implementation Steps": "Name the owning runtime path and expand the change sequence into safe ordered steps.",
        "Unit Tests": "Specify the exact automated regression checks and assertions that must pass.",
        "Risks": "Name the main gameplay regression risk and how the implementation will contain it.",
        "Acceptance Criteria": "Write player-visible pass conditions and edge-case expectations.",
    }
    return actions[section]


def _score_section(name: str, weight: int, sections: dict[str, str], task_type: str) -> SectionReview:
    content = sections.get(name, "").strip()
    lines = _clean_lines(content)
    lowered = content.lower()
    score = 0
    status = "missing"
    rationale = ""

    if content:
        score = max(1, round(weight * 0.6))
        status = "needs-work"

    if name == "Overview":
        if content and len(lines) >= 2:
            score = weight
            status = "pass"
            rationale = "The gameplay goal and scope are clear enough for implementation."
        elif content:
            rationale = "The overview exists, but it still needs nearby gameplay scope or player impact."
        else:
            rationale = "The plan does not summarize the gameplay goal clearly enough."
    elif name == "Task Type":
        if content and task_type in lowered and "reason" in lowered:
            score = weight
            status = "pass"
            rationale = "The plan names the task type and justifies why that framing is correct."
        elif content and task_type in lowered:
            rationale = "The task type is named, but the approval rationale is still too thin."
        else:
            rationale = "The plan does not explain what kind of gameplay work this is."
    elif name == "Existing Docs":
        if content and len(lines) >= 1 and ("/" in content or "\\" in content):
            score = weight
            status = "pass"
            rationale = "The plan references concrete gameplay docs or runtime notes."
        elif content:
            rationale = "The references are present, but they are not grounded enough in concrete files."
        else:
            rationale = "The plan does not cite the docs or runtime references that constrain the change."
    elif name == "Implementation Steps":
        if content and len(lines) >= 3 and ("/" in content or "\\" in content):
            score = weight
            status = "pass"
            rationale = "The implementation steps are concrete, ordered, and anchored on the owning runtime path."
        elif content:
            rationale = "The implementation steps exist, but they remain too generic to trust."
        else:
            rationale = "The plan does not explain how the gameplay change will be implemented safely."
    elif name == "Unit Tests":
        if content and len(lines) >= 2 and "test" in lowered:
            score = weight
            status = "pass"
            rationale = "The plan names concrete regression tests and assertions."
        elif content:
            rationale = "The plan mentions validation, but the exact automated checks are still vague."
        else:
            rationale = "The plan does not define the regression tests needed for this gameplay change."
    elif name == "Risks":
        if content and len(lines) >= 2 and "risk" in lowered:
            score = weight
            status = "pass"
            rationale = "The main gameplay regression risk and mitigation are documented."
        elif content:
            rationale = "The risks are mentioned, but the mitigation remains too vague."
        else:
            rationale = "The plan does not describe the main regression risk."
    else:
        if content and len(lines) >= 2:
            score = weight
            status = "pass"
            rationale = "The acceptance criteria describe player-visible success conditions clearly enough."
        elif content:
            rationale = "The acceptance criteria exist, but they need more concrete pass conditions."
        else:
            rationale = "The plan does not define player-visible acceptance criteria."

    return {
        "section": name,
        "score": score,
        "status": status,
        "rationale": rationale,
        "action_items": [] if status == "pass" else [_fallback_action(name)],
    }


def _fallback_review(plan_doc: str, task_type: str) -> dict[str, Any]:
    sections = _parse_sections(plan_doc)
    section_reviews = [_score_section(name, weight, sections, task_type) for name, weight in SECTION_WEIGHTS]
    missing_sections = [item["section"] for item in section_reviews if item["status"] == "missing"]
    blocking_issues = _dedupe(
        [
            f"{item['section']}: {item['action_items'][0]}"
            for item in section_reviews
            if item["section"] in BLOCKING_SECTIONS and item["status"] != "pass"
        ]
    )
    improvement_actions = _dedupe(
        [
            action
            for item in section_reviews
            if item["status"] != "pass"
            for action in item["action_items"]
        ]
    )
    score = sum(item["score"] for item in section_reviews)
    approved = score >= APPROVAL_SCORE and not blocking_issues and not missing_sections
    return {
        "score": score,
        "approved": approved,
        "missing_sections": missing_sections,
        "section_reviews": section_reviews,
        "blocking_issues": blocking_issues,
        "improvement_actions": improvement_actions,
    }


def _compose_feedback(
    *,
    review_round: int,
    score: int,
    approved: bool,
    loop_status: str,
    loop_reason: str,
    section_reviews: list[SectionReview],
    missing_sections: list[str],
    blocking_issues: list[str],
    improvement_actions: list[str],
    notes: str,
) -> str:
    lines = [
        "# Gameplay Plan Review",
        "",
        f"- Review round: {review_round}",
        f"- Approved: {approved}",
        f"- Score: {score}/100",
        f"- Loop Status: {loop_status}",
        f"- Loop Reason: {loop_reason}",
        f"- Approval bar: >= {APPROVAL_SCORE}/100 and zero blocking issues",
        "",
        "## Section Scores",
    ]
    for item in section_reviews:
        max_score = dict(SECTION_WEIGHTS)[item["section"]]
        lines.append(f"- {item['section']}: {item['score']}/{max_score} - {item['rationale']}")
    lines.extend(["", "## Missing Sections"])
    lines.extend([f"- {item}" for item in missing_sections] or ["- None."])
    lines.extend(["", "## Blocking Issues"])
    lines.extend([f"- {item}" for item in blocking_issues] or ["- None."])
    lines.extend(["", "## Improvement Actions"])
    lines.extend([f"- {item}" for item in improvement_actions] or ["- None."])
    lines.extend(["", "## Reviewer Notes", notes.strip() or "The plan is ready for implementation."])
    return "\n".join(lines)


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name

    def review_plan(state: ReviewerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        review_round = int(state.get("review_round", 0)) + 1
        task_type = str(state.get("task_type", "feature") or "feature").strip().lower()
        fallback = _fallback_review(state["plan_doc"], task_type)
        review_result = fallback

        if context.llm.is_enabled():
            schema = {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                    "feedback": {"type": "string"},
                    "missing_sections": {"type": "array", "items": {"type": "string"}},
                    "section_reviews": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section": {"type": "string"},
                                "score": {"type": "integer"},
                                "status": {"type": "string"},
                                "rationale": {"type": "string"},
                                "action_items": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["section", "score", "status", "rationale", "action_items"],
                            "additionalProperties": False,
                        },
                    },
                    "blocking_issues": {"type": "array", "items": {"type": "string"}},
                    "improvement_actions": {"type": "array", "items": {"type": "string"}},
                    "approved": {"type": "boolean"},
                },
                "required": [
                    "score",
                    "feedback",
                    "missing_sections",
                    "section_reviews",
                    "blocking_issues",
                    "improvement_actions",
                    "approved",
                ],
                "additionalProperties": False,
            }
            try:
                generated = context.llm.generate_json(
                    instructions=(
                        "You are gameplay-reviewer-workflow. Review a gameplay implementation plan like a strict senior gameplay engineer. "
                        "Score the plan against these exact sections: Overview, Task Type, Existing Docs, Implementation Steps, Unit Tests, "
                        "Risks, Acceptance Criteria. Return structured JSON only. Approval requires a score >= 90, no blocking issues, and "
                        "no missing required sections. Focus on technical clarity, owner paths, regression coverage, and player-visible acceptance."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Task type: {task_type}\n"
                        f"Execution track: {state.get('execution_track', task_type)}\n"
                        f"Review round: {review_round}\n\n"
                        f"Plan document:\n{state['plan_doc']}\n"
                    ),
                    schema_name="gameplay_plan_review",
                    schema=schema,
                )
                review_result = {
                    "score": int(generated.get("score", fallback["score"])),
                    "approved": bool(generated.get("approved", False)),
                    "missing_sections": [str(item) for item in generated.get("missing_sections", []) if str(item).strip()],
                    "section_reviews": [
                        {
                            "section": str(item["section"]).strip(),
                            "score": max(0, int(item["score"])),
                            "status": str(item["status"]).strip(),
                            "rationale": str(item["rationale"]).strip(),
                            "action_items": [
                                str(action).strip() for action in item.get("action_items", []) if str(action).strip()
                            ],
                        }
                        for item in generated.get("section_reviews", [])
                        if str(item.get("section", "")).strip()
                    ]
                    or fallback["section_reviews"],
                    "blocking_issues": _dedupe([str(item) for item in generated.get("blocking_issues", []) if str(item).strip()]),
                    "improvement_actions": _dedupe(
                        [str(item) for item in generated.get("improvement_actions", []) if str(item).strip()]
                    ),
                }
                if not review_result["blocking_issues"] and not review_result["approved"]:
                    review_result["blocking_issues"] = fallback["blocking_issues"]
                if not review_result["improvement_actions"] and not review_result["approved"]:
                    review_result["improvement_actions"] = fallback["improvement_actions"]
            except LLMError:
                review_result = fallback

        previous_score = int(state.get("score", 0)) if review_round > 1 else None
        prior_stagnated_rounds = int(state.get("loop_stagnated_rounds", 0)) if review_round > 1 else 0
        progress = evaluate_quality_loop(
            PLAN_REVIEW_SPEC,
            round_index=review_round,
            score=review_result["score"],
            approved=bool(review_result["approved"]),
            missing_sections=review_result["missing_sections"],
            blocking_issues=review_result["blocking_issues"],
            improvement_actions=review_result["improvement_actions"],
            previous_score=previous_score,
            prior_stagnated_rounds=prior_stagnated_rounds,
        )
        feedback = _compose_feedback(
            review_round=review_round,
            score=progress.score,
            approved=progress.approved,
            loop_status=progress.status,
            loop_reason=progress.reason,
            section_reviews=review_result["section_reviews"],
            missing_sections=list(progress.missing_sections),
            blocking_issues=list(progress.blocking_issues),
            improvement_actions=list(progress.improvement_actions),
            notes=str(review_result.get("feedback", "")).strip(),
        )
        (artifact_dir / f"review_round_{review_round}.md").write_text(feedback, encoding="utf-8")
        return {
            "artifact_dir": str(artifact_dir),
            "review_round": review_round,
            "score": progress.score,
            "feedback": feedback,
            "missing_sections": list(progress.missing_sections),
            "section_reviews": list(review_result["section_reviews"]),
            "blocking_issues": list(progress.blocking_issues),
            "improvement_actions": list(progress.improvement_actions),
            "approved": progress.approved,
            "loop_status": progress.status,
            "loop_reason": progress.reason,
            "loop_should_continue": progress.should_continue,
            "loop_completed": progress.completed,
            "loop_stagnated_rounds": progress.stagnated_rounds,
            "review_score": progress.score,
            "review_feedback": feedback,
            "review_missing_sections": list(progress.missing_sections),
            "review_blocking_issues": list(progress.blocking_issues),
            "review_improvement_actions": list(progress.improvement_actions),
            "review_approved": progress.approved,
            "review_loop_status": progress.status,
            "review_loop_reason": progress.reason,
            "review_loop_should_continue": progress.should_continue,
            "review_loop_completed": progress.completed,
            "review_loop_stagnated_rounds": progress.stagnated_rounds,
            "summary": (
                f"{metadata.name} approved review round {review_round} with score {progress.score}/100."
                if progress.approved
                else (
                    f"{metadata.name} stopped after round {review_round} with score {progress.score}/100."
                    if progress.completed
                    else f"{metadata.name} requested another plan revision after round {review_round}."
                )
            ),
        }

    graph = StateGraph(ReviewerState)
    graph.add_node("review_plan", trace_graph_node(graph_name=graph_name, node_name="review_plan", node_fn=review_plan))
    graph.add_edge(START, "review_plan")
    graph.add_edge("review_plan", END)
    return graph.compile()
