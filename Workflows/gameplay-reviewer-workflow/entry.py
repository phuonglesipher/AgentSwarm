from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_logging import trace_graph_node
from core.llm import LLMError
from core.models import WorkflowContext, WorkflowMetadata
from core.quality_loop import QualityLoopSpec, evaluate_quality_loop


class SectionReview(TypedDict):
    section: str
    score: int
    max_score: int
    status: str
    rationale: str
    action_items: list[str]


class ReviewerState(TypedDict):
    task_prompt: str
    plan_doc: str
    review_round: int
    run_dir: NotRequired[str]
    task_id: NotRequired[str]
    score: int
    feedback: str
    missing_sections: list[str]
    section_reviews: list[SectionReview]
    blocking_issues: list[str]
    improvement_actions: list[str]
    approved: bool
    summary: str
    loop_status: NotRequired[str]
    loop_reason: NotRequired[str]
    loop_threshold: NotRequired[int]
    loop_max_rounds: NotRequired[int]
    loop_should_continue: NotRequired[bool]
    loop_completed: NotRequired[bool]
    loop_stagnated_rounds: NotRequired[int]


SECTION_WEIGHTS = {
    "Overview": 10,
    "Task Type": 10,
    "Existing Docs": 10,
    "Implementation Steps": 25,
    "Unit Tests": 20,
    "Risks": 10,
    "Acceptance Criteria": 15,
}
SECTION_ORDER = list(SECTION_WEIGHTS)
APPROVAL_SCORE = 90
MAX_REVIEW_ROUNDS = 3
BLOCKING_SECTIONS = {
    "Implementation Steps",
    "Unit Tests",
    "Acceptance Criteria",
}
SECTION_GUIDANCE = {
    "Overview": {
        "criteria": [
            "State the gameplay problem or feature goal in player-facing language.",
            "Name the affected systems, inputs, or states that are in scope.",
        ],
        "actions": [
            "Explain the gameplay goal or bug from the player's point of view.",
            "Call out the systems, states, or inputs that the implementation will touch.",
        ],
    },
    "Task Type": {
        "criteria": [
            "Label the task as bugfix or feature.",
            "Explain why the task belongs to that category.",
        ],
        "actions": [
            "State whether the work is a bugfix or a feature.",
            "Add one sentence explaining why the task belongs to that category.",
        ],
    },
    "Existing Docs": {
        "criteria": [
            "Reference the gameplay or design docs that informed the plan.",
            "Note when no matching docs were found so the engineer knows the plan is starting from scratch.",
        ],
        "actions": [
            "Reference the gameplay or design docs that informed the change.",
            "If no docs were found, say that explicitly and describe the baseline assumptions.",
        ],
    },
    "Implementation Steps": {
        "criteria": [
            "List the ordered implementation steps.",
            "Call out the affected gameplay systems, states, or touch points.",
            "Mention safeguards or instrumentation that will help debug regressions.",
        ],
        "actions": [
            "Break the implementation into an ordered list of concrete steps.",
            "Name the gameplay systems, states, or assets touched by each step.",
            "Add a regression safeguard such as logging, assertions, or state validation.",
        ],
    },
    "Unit Tests": {
        "criteria": [
            "List the automated tests that will be added or updated.",
            "Describe the gameplay conditions each test verifies.",
        ],
        "actions": [
            "List the automated tests that will be added or updated.",
            "Describe the gameplay condition each test verifies, including the expected assertion.",
        ],
    },
    "Risks": {
        "criteria": [
            "Describe the likely gameplay or technical risks.",
            "Add mitigation, fallback, or monitoring notes for those risks.",
        ],
        "actions": [
            "Describe the likely gameplay or technical risks in this change.",
            "Add the mitigation, fallback, or monitoring plan for each risk.",
        ],
    },
    "Acceptance Criteria": {
        "criteria": [
            "Describe the player-observable result when the work is done.",
            "Call out the regression checks or edge cases that must still pass.",
        ],
        "actions": [
            "Write player-visible acceptance criteria for the finished gameplay behavior.",
            "Add regression or edge-case checks that must still pass before the task is done.",
        ],
    },
}
PLAN_REVIEW_LOOP_SPEC = QualityLoopSpec(
    loop_id="gameplay-plan-review",
    threshold=APPROVAL_SCORE,
    max_rounds=MAX_REVIEW_ROUNDS,
    require_blocker_free=True,
    require_missing_section_free=True,
    require_explicit_approval=True,
    min_score_delta=1,
    stagnation_limit=2,
)


def _parse_plan_sections(plan_doc: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_section: str | None = None
    buffer: list[str] = []

    for line in plan_doc.splitlines():
        if line.startswith("## "):
            if current_section is not None:
                sections[current_section] = "\n".join(buffer).strip()
            heading = line[3:].strip()
            current_section = heading if heading in SECTION_WEIGHTS else None
            buffer = []
            continue

        if current_section is not None:
            buffer.append(line)

    if current_section is not None:
        sections[current_section] = "\n".join(buffer).strip()
    return sections


def _detect_missing_sections(plan_doc: str) -> list[str]:
    parsed = _parse_plan_sections(plan_doc)
    return [section for section in SECTION_ORDER if not parsed.get(section, "").strip()]


def _clean_lines(section_text: str) -> list[str]:
    return [line.strip() for line in section_text.splitlines() if line.strip()]


def _bullet_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.lstrip().startswith(("-", "*"))]


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_item in items:
        item = str(raw_item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _build_section_review(section: str, section_text: str) -> SectionReview:
    weight = SECTION_WEIGHTS[section]
    actions = list(SECTION_GUIDANCE[section]["actions"])
    if not section_text.strip():
        return {
            "section": section,
            "score": 0,
            "max_score": weight,
            "status": "missing",
            "rationale": f"The required `{section}` section is missing.",
            "action_items": actions,
        }

    lines = _clean_lines(section_text)
    bullets = _bullet_lines(lines)
    text = " ".join(line.lstrip("-* ").strip() for line in lines).lower()
    partial_score = max(1, round(weight * 0.6))

    if section == "Overview":
        if len(lines) >= 2 and (len(bullets) >= 2 or len(text) >= 80):
            return {
                "section": section,
                "score": weight,
                "max_score": weight,
                "status": "pass",
                "rationale": "The plan explains the gameplay goal and scope clearly enough to start implementation.",
                "action_items": [],
            }
        return {
            "section": section,
            "score": partial_score,
            "max_score": weight,
            "status": "needs-work",
            "rationale": "The overview exists, but it does not clearly describe the gameplay goal and scope.",
            "action_items": actions,
        }

    if section == "Task Type":
        has_type = any(token in text for token in ("bugfix", "bug fix", "feature"))
        has_reason = any(
            token in text
            for token in ("because", "reason", "regression", "unintended", "goal", "request", "scope")
        )
        if has_type and has_reason:
            return {
                "section": section,
                "score": weight,
                "max_score": weight,
                "status": "pass",
                "rationale": "The task type is named and the plan explains why it is the right classification.",
                "action_items": [],
            }
        if has_type:
            return {
                "section": section,
                "score": partial_score,
                "max_score": weight,
                "status": "needs-work",
                "rationale": "The task type is present, but the plan does not justify the classification.",
                "action_items": actions,
            }
        return {
            "section": section,
            "score": 0,
            "max_score": weight,
            "status": "missing",
            "rationale": "The task type section does not identify the work as a bugfix or feature.",
            "action_items": actions,
        }

    if section == "Existing Docs":
        if len(bullets) >= 1 and "no matching docs found" not in text:
            return {
                "section": section,
                "score": weight,
                "max_score": weight,
                "status": "pass",
                "rationale": "The plan references supporting gameplay or design docs.",
                "action_items": [],
            }
        return {
            "section": section,
            "score": partial_score,
            "max_score": weight,
            "status": "needs-work",
            "rationale": "The docs section exists, but it should better explain what references informed the plan.",
            "action_items": actions,
        }

    if section == "Implementation Steps":
        if len(bullets) >= 3:
            return {
                "section": section,
                "score": weight,
                "max_score": weight,
                "status": "pass",
                "rationale": "The implementation steps are concrete, ordered, and actionable.",
                "action_items": [],
            }
        return {
            "section": section,
            "score": partial_score,
            "max_score": weight,
            "status": "needs-work",
            "rationale": "The implementation steps need more concrete sequencing and system touch points.",
            "action_items": actions,
        }

    if section == "Unit Tests":
        has_test_language = any(
            token in text for token in ("test", "assert", "coverage", "regression", "verify", "unit")
        )
        if len(bullets) >= 2 and has_test_language:
            return {
                "section": section,
                "score": weight,
                "max_score": weight,
                "status": "pass",
                "rationale": "The plan lists concrete automated tests and the gameplay checks they cover.",
                "action_items": [],
            }
        return {
            "section": section,
            "score": partial_score,
            "max_score": weight,
            "status": "needs-work",
            "rationale": "The unit test section needs concrete automated checks and expected assertions.",
            "action_items": actions,
        }

    if section == "Risks":
        has_mitigation_language = any(
            token in text for token in ("mitig", "fallback", "guard", "rollback", "monitor", "contingency")
        )
        if len(bullets) >= 2 and has_mitigation_language:
            return {
                "section": section,
                "score": weight,
                "max_score": weight,
                "status": "pass",
                "rationale": "The plan identifies risks and explains how they will be mitigated.",
                "action_items": [],
            }
        return {
            "section": section,
            "score": partial_score,
            "max_score": weight,
            "status": "needs-work",
            "rationale": "The risks are listed, but the mitigation or fallback plan is too vague.",
            "action_items": actions,
        }

    has_acceptance_language = any(
        token in text for token in ("player", "should", "must", "when", "expected", "observable", "pass")
    )
    if len(bullets) >= 2 and has_acceptance_language:
        return {
            "section": section,
            "score": weight,
            "max_score": weight,
            "status": "pass",
            "rationale": "The plan defines player-visible outcomes and regression checks for completion.",
            "action_items": [],
        }
    return {
        "section": section,
        "score": partial_score,
        "max_score": weight,
        "status": "needs-work",
        "rationale": "The acceptance criteria need clearer player-visible outcomes and pass conditions.",
        "action_items": actions,
    }


def _derive_blocking_issues(section_reviews: list[SectionReview]) -> list[str]:
    blockers: list[str] = []
    for review in section_reviews:
        if review["status"] == "missing":
            blockers.append(f"{review['section']}: add the missing section before implementation starts.")
            continue
        if review["section"] in BLOCKING_SECTIONS and review["status"] != "pass":
            next_action = review["action_items"][0] if review["action_items"] else review["rationale"]
            blockers.append(f"{review['section']}: {next_action}")
    return _dedupe_preserve_order(blockers)


def _derive_improvement_actions(section_reviews: list[SectionReview], llm_actions: Iterable[str] = ()) -> list[str]:
    actions: list[str] = []
    for review in section_reviews:
        if review["status"] == "pass":
            continue
        actions.extend(review["action_items"])
    actions.extend(llm_actions)
    return _dedupe_preserve_order(actions)


def _compose_feedback(
    task_prompt: str,
    review_round: int,
    score: int,
    approved: bool,
    blocking_issues: list[str],
    improvement_actions: list[str],
    section_reviews: list[SectionReview],
    llm_feedback: str = "",
) -> str:
    decision = "Approved for implementation" if approved else "Revise plan before implementation"
    lines = [
        "# Gameplay Plan Review",
        "",
        f"- Task: {task_prompt}",
        f"- Review round: {review_round}",
        f"- Decision: {decision}",
        f"- Total score: {score}/100",
        f"- Approval bar: >= {APPROVAL_SCORE}/100 and zero blocking issues",
        "",
        "## Blocking Issues",
    ]
    if blocking_issues:
        lines.extend(f"- {issue}" for issue in blocking_issues)
    else:
        lines.append("- None.")

    lines.extend(["", "## Improvement Checklist"])
    if improvement_actions:
        lines.extend(f"- [ ] {item}" for item in improvement_actions)
    else:
        lines.append("- [x] No further plan changes requested before implementation.")

    lines.extend(["", "## Section Scores"])
    for review in section_reviews:
        lines.append(
            f"- {review['section']}: {review['score']}/{review['max_score']} "
            f"({review['status']}) - {review['rationale']}"
        )

    if llm_feedback.strip():
        lines.extend(["", "## Reviewer Notes", llm_feedback.strip()])

    return "\n".join(lines)


def _format_rubric_for_prompt() -> str:
    lines = [f"Approval rule: score >= {APPROVAL_SCORE} and no blocking issues.", ""]
    for section in SECTION_ORDER:
        criteria = SECTION_GUIDANCE[section]["criteria"]
        lines.append(f"- {section} ({SECTION_WEIGHTS[section]} points)")
        lines.extend(f"  - {criterion}" for criterion in criteria)
    return "\n".join(lines)


def _fallback_review(task_prompt: str, plan_doc: str, review_round: int) -> dict[str, Any]:
    parsed_sections = _parse_plan_sections(plan_doc)
    section_reviews = [_build_section_review(section, parsed_sections.get(section, "")) for section in SECTION_ORDER]
    missing_sections = [review["section"] for review in section_reviews if review["status"] == "missing"]
    score = sum(review["score"] for review in section_reviews)
    blocking_issues = _derive_blocking_issues(section_reviews)
    improvement_actions = _derive_improvement_actions(section_reviews)
    progress = evaluate_quality_loop(
        PLAN_REVIEW_LOOP_SPEC,
        round_index=review_round,
        score=score,
        approved=score >= APPROVAL_SCORE,
        missing_sections=missing_sections,
        blocking_issues=blocking_issues,
        improvement_actions=improvement_actions,
    )
    feedback = _compose_feedback(
        task_prompt=task_prompt,
        review_round=review_round,
        score=score,
        approved=progress.approved,
        blocking_issues=blocking_issues,
        improvement_actions=improvement_actions,
        section_reviews=section_reviews,
    )
    return {
        "score": score,
        "feedback": feedback,
        "missing_sections": missing_sections,
        "section_reviews": section_reviews,
        "blocking_issues": blocking_issues,
        "improvement_actions": improvement_actions,
        "approved": progress.approved,
        "summary": (
            f"Reviewer scored the plan at {score}/100 and "
            f"{'approved it' if progress.approved else 'requested revisions'}."
        ),
        "loop_status": progress.status,
        "loop_reason": progress.reason,
        "loop_threshold": progress.threshold,
        "loop_max_rounds": progress.max_rounds,
        "loop_should_continue": progress.should_continue,
        "loop_completed": progress.completed,
        "loop_stagnated_rounds": progress.stagnated_rounds,
    }


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
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
                    "items": {"type": "string", "enum": SECTION_ORDER},
                    "uniqueItems": True,
                },
                "section_reviews": {
                    "type": "array",
                    "minItems": len(SECTION_ORDER),
                    "maxItems": len(SECTION_ORDER),
                    "items": {
                        "type": "object",
                        "properties": {
                            "section": {"type": "string", "enum": SECTION_ORDER},
                            "score": {"type": "integer", "minimum": 0, "maximum": max(SECTION_WEIGHTS.values())},
                            "status": {"type": "string", "enum": ["pass", "needs-work", "missing"]},
                            "rationale": {"type": "string"},
                            "action_items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["section", "score", "status", "rationale", "action_items"],
                        "additionalProperties": False,
                    },
                },
                "blocking_issues": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "improvement_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
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
            result = llm.generate_json(
                instructions=(
                    "You are gameplay-reviewer-workflow. Review gameplay implementation plans. "
                    "Evaluate every required section against the rubric. "
                    "Return one section review per required section, with a score, status, rationale, and concrete action items. "
                    "Treat missing sections as score 0. Only approve when the plan reaches the approval rule and has no blocking issues. "
                    "Keep the feedback concise and actionable."
                ),
                input_text=(
                    f"Task prompt:\n{state['task_prompt']}\n\n"
                    f"Review round: {state['review_round']}\n\n"
                    f"Required rubric:\n{_format_rubric_for_prompt()}\n\n"
                    f"Plan document:\n{state['plan_doc']}"
                ),
                schema_name="gameplay_plan_review",
                schema=schema,
            )
        except LLMError:
            return fallback

        fallback_reviews = {
            review["section"]: review
            for review in fallback["section_reviews"]
        }
        llm_reviews = {
            review["section"]: review
            for review in result.get("section_reviews", [])
            if isinstance(review, dict) and review.get("section") in SECTION_WEIGHTS
        }

        merged_reviews: list[SectionReview] = []
        for section in SECTION_ORDER:
            base_review = fallback_reviews[section]
            raw_review = llm_reviews.get(section)
            if raw_review is None or base_review["status"] == "missing":
                merged_reviews.append(base_review)
                continue

            raw_score = int(raw_review.get("score", base_review["score"]))
            score = max(0, min(raw_score, SECTION_WEIGHTS[section]))
            status = str(raw_review.get("status", base_review["status"])).strip().lower()
            if status not in {"pass", "needs-work", "missing"}:
                status = base_review["status"]
            if status == "missing":
                status = "needs-work"
                score = min(score, base_review["score"])
            if status == "pass" and score < SECTION_WEIGHTS[section]:
                status = "needs-work"
            if status == "pass":
                score = SECTION_WEIGHTS[section]
            if status != "pass" and score == SECTION_WEIGHTS[section]:
                score = max(base_review["score"], SECTION_WEIGHTS[section] - 1)

            action_items = []
            for item in raw_review.get("action_items", []):
                if isinstance(item, str):
                    action_items.append(item)
            if status == "pass":
                action_items = []
            else:
                action_items = _dedupe_preserve_order(action_items or base_review["action_items"])

            rationale = str(raw_review.get("rationale") or base_review["rationale"]).strip() or base_review["rationale"]
            merged_reviews.append(
                {
                    "section": section,
                    "score": score,
                    "max_score": SECTION_WEIGHTS[section],
                    "status": status,
                    "rationale": rationale,
                    "action_items": action_items,
                }
            )

        missing_sections = [review["section"] for review in merged_reviews if review["status"] == "missing"]
        score = sum(review["score"] for review in merged_reviews)
        blocking_issues = _derive_blocking_issues(merged_reviews)
        blocking_issues.extend(str(item) for item in result.get("blocking_issues", []) if isinstance(item, str))
        blocking_issues = _dedupe_preserve_order(blocking_issues)
        improvement_actions = _derive_improvement_actions(merged_reviews, result.get("improvement_actions", []))
        progress = evaluate_quality_loop(
            PLAN_REVIEW_LOOP_SPEC,
            round_index=state["review_round"],
            score=score,
            approved=bool(result.get("approved")),
            missing_sections=missing_sections,
            blocking_issues=blocking_issues,
            improvement_actions=improvement_actions,
        )
        feedback = _compose_feedback(
            task_prompt=state["task_prompt"],
            review_round=state["review_round"],
            score=score,
            approved=progress.approved,
            blocking_issues=blocking_issues,
            improvement_actions=improvement_actions,
            section_reviews=merged_reviews,
            llm_feedback=str(result.get("feedback", "")).strip(),
        )
        return {
            "score": max(0, min(score, 100)),
            "feedback": feedback,
            "missing_sections": missing_sections,
            "section_reviews": merged_reviews,
            "blocking_issues": blocking_issues,
            "improvement_actions": improvement_actions,
            "approved": progress.approved,
            "summary": (
                f"Reviewer scored the plan at {max(0, min(score, 100))}/100 and "
                f"{'approved it' if progress.approved else 'requested revisions'}."
            ),
            "loop_status": progress.status,
            "loop_reason": progress.reason,
            "loop_threshold": progress.threshold,
            "loop_max_rounds": progress.max_rounds,
            "loop_should_continue": progress.should_continue,
            "loop_completed": progress.completed,
            "loop_stagnated_rounds": progress.stagnated_rounds,
        }

    graph = StateGraph(ReviewerState)
    graph.add_node(
        "review_plan",
        trace_graph_node(graph_name=graph_name, node_name="review_plan", node_fn=review_plan),
    )
    graph.add_edge(START, "review_plan")
    graph.add_edge("review_plan", END)
    return graph
