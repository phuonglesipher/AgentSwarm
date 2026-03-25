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
from core.natural_language_prompts import build_prompt_brief
from core.quality_loop import QualityLoopSpec, evaluate_quality_loop
from core.scoring import ScoreAssessment, ScorePolicy, evaluate_score_decision
from core.text_utils import normalize_text, slugify


APPROVAL_SCORE = 90
MIN_REVIEW_ROUNDS = 2
MAX_REVIEW_ROUNDS = 3
MANDATORY_VERIFICATION_ACTION = (
    "Run one more plan revision pass that independently tightens the player-visible outcome, "
    "grounded owner evidence, and regression coverage before implementation."
)
PLAN_REVIEW_SPEC = QualityLoopSpec(
    loop_id="gameplay-plan-review",
    threshold=APPROVAL_SCORE,
    max_rounds=MAX_REVIEW_ROUNDS,
    min_rounds=MIN_REVIEW_ROUNDS,
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
SECTION_WEIGHT_MAP = dict(SECTION_WEIGHTS)
PLAN_REVIEW_SCORE_POLICY = ScorePolicy(
    system_id="gameplay-plan-review",
    threshold=APPROVAL_SCORE,
    require_blocker_free=True,
    require_missing_section_free=True,
    require_explicit_approval=True,
)
PROCESS_ONLY_REVIEW_PATTERNS = (
    r"\breview round\b",
    r"\bround metadata\b",
    r"\bcurrent review context\b",
    r"\bprocess[- ]gate\b",
    r"\bindependent verif(?:ication|ier)\b",
    r"\bsign[- ]off\b",
    r"\bevidence artifact\b",
    r"\bverification artifact\b",
    r"\bartifact naming\b",
    r"\btraceability\b",
    r"\bapproval[- ]trace\b",
    r"\bmetadata\b",
    r"\bcurrent-round\b",
    r"\bround-\d+\b",
    r"\blog filenames?\b",
    r"\btargeted test command",
)
GENERIC_HARD_BLOCKER_PREFIXES = (
    "Player Outcome:",
    "Current Behavior Evidence:",
    "Speculation Control:",
    "Edge and Regression Coverage:",
)


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
    review_score_confidence: NotRequired[float | None]
    review_score_confidence_label: NotRequired[str]
    review_score_confidence_reason: NotRequired[str]
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


def _is_process_only_review_item(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in PROCESS_ONLY_REVIEW_PATTERNS)


def _apply_process_drift_guardrails(
    *,
    review_result: dict[str, Any],
) -> dict[str, Any]:
    sanitized_section_reviews = list(review_result["section_reviews"])
    generated_missing_sections = _dedupe(
        [
            *[str(item) for item in review_result.get("missing_sections", []) if str(item).strip()],
            *[item["section"] for item in sanitized_section_reviews if item["status"] == "missing"],
        ]
    )
    generated_blocking_issues = _dedupe(
        [
            str(item)
            for item in review_result.get("blocking_issues", [])
            if str(item).strip() and not _is_process_only_review_item(str(item))
        ]
    )
    if all(item["status"] == "pass" for item in sanitized_section_reviews):
        generated_blocking_issues = _dedupe(
            [
                item
                for item in generated_blocking_issues
                if not any(item.startswith(prefix) for prefix in GENERIC_HARD_BLOCKER_PREFIXES)
            ]
        )
    generated_improvement_actions = _dedupe(
        [
            str(item)
            for item in review_result.get("improvement_actions", [])
            if str(item).strip() and not _is_process_only_review_item(str(item))
        ]
    )
    score = sum(item["score"] for item in sanitized_section_reviews)
    missing_sections = generated_missing_sections
    blocking_issues = generated_blocking_issues
    improvement_actions = generated_improvement_actions
    approved = bool(review_result.get("approved", False))
    if score >= APPROVAL_SCORE and not missing_sections and not blocking_issues:
        approved = True
    return {
        "score": score,
        "approved": approved,
        "missing_sections": missing_sections,
        "section_reviews": sanitized_section_reviews,
        "blocking_issues": blocking_issues,
        "improvement_actions": improvement_actions,
    }


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


def _to_score_assessments(section_reviews: list[SectionReview]) -> list[ScoreAssessment]:
    return [
        ScoreAssessment(
            label=item["section"],
            score=int(item["score"]),
            max_score=SECTION_WEIGHT_MAP[item["section"]],
            status=item["status"],
            rationale=item["rationale"],
            action_items=tuple(item["action_items"]),
        )
        for item in section_reviews
    ]


def _normalize_section_reviews(
    section_reviews: list[SectionReview],
) -> list[SectionReview]:
    normalized_lookup: dict[str, SectionReview] = {}
    for item in section_reviews:
        section = str(item.get("section", "")).strip()
        if section not in SECTION_WEIGHT_MAP or section in normalized_lookup:
            continue
        weight = SECTION_WEIGHT_MAP[section]
        score = max(0, min(int(item.get("score", 0)), weight))
        raw_status = str(item.get("status", "")).strip().lower()
        if raw_status == "missing" or score == 0:
            status = "missing"
        elif raw_status in {"pass", "approved"} and score >= weight:
            status = "pass"
        else:
            status = "needs-work"
        normalized_lookup[section] = {
            "section": section,
            "score": score,
            "status": status,
            "rationale": str(item.get("rationale", "")).strip(),
            "action_items": (
                []
                if status == "pass"
                else _dedupe([str(action).strip() for action in item.get("action_items", []) if str(action).strip()])
            ),
        }

    return [normalized_lookup[section] for section, _ in SECTION_WEIGHTS]


def _compose_feedback(
    *,
    review_round: int,
    score: int,
    approved: bool,
    confidence_label: str,
    confidence_reason: str,
    confidence: float | None,
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
        f"- Scoring confidence: {confidence_label}{f' ({confidence:.1f}x noise floor)' if confidence is not None else ''}",
        f"- Confidence reason: {confidence_reason}",
        f"- Loop Status: {loop_status}",
        f"- Loop Reason: {loop_reason}",
        f"- Approval bar: >= {APPROVAL_SCORE}/100 and zero blocking issues",
        f"- Minimum review depth: {MIN_REVIEW_ROUNDS} round(s)",
        "",
        "## Hard Blocker And Scoring Rules",
        "- [hard blocker] Player Outcome: the plan must name the player-visible result and scope boundary.",
        "- [hard blocker] Current Behavior Evidence: the plan must cite grounded docs, runtime paths, and the owner.",
        "- [hard blocker] Speculation Control: implementation steps must stay anchored on current ownership.",
        "- [hard blocker] Edge and Regression Coverage: tests and acceptance criteria must protect adjacent gameplay paths.",
        "",
        "## Section Scores",
    ]
    for item in section_reviews:
        max_score = SECTION_WEIGHT_MAP[item["section"]]
        lines.append(f"- {item['section']}: {item['score']}/{max_score} - {item['rationale']}")
    lines.extend(["", "## Missing Sections"])
    lines.extend([f"- {item}" for item in missing_sections] or ["- None."])
    lines.extend(["", "## Blocking Issues"])
    lines.extend([f"- {item}" for item in blocking_issues] or ["- None."])
    lines.extend(["", "## Improvement Actions"])
    lines.extend([f"- {item}" for item in improvement_actions] or ["- None."])
    lines.extend(["", "## Reviewer Notes", notes.strip() or "The plan is ready for implementation."])
    return "\n".join(lines)


def _blocked_review_response(
    *,
    artifact_dir: Path,
    metadata: WorkflowMetadata,
    review_round: int,
    reason: str,
    loop_status: str,
) -> dict[str, Any]:
    improvement_action = "Enable the reviewer LLM and rerun gameplay plan review so the scoring assessments come from LLM output."
    feedback = _compose_feedback(
        review_round=review_round,
        score=0,
        approved=False,
        confidence_label="unmeasured",
        confidence_reason="MAD confidence is unavailable because no LLM-generated assessments were produced.",
        confidence=None,
        loop_status=loop_status,
        loop_reason=reason,
        section_reviews=[],
        missing_sections=[],
        blocking_issues=[reason],
        improvement_actions=[improvement_action],
        notes=reason,
    )
    (artifact_dir / f"review_round_{review_round}.md").write_text(feedback, encoding="utf-8")
    return {
        "artifact_dir": str(artifact_dir),
        "review_round": review_round,
        "score": 0,
        "feedback": feedback,
        "missing_sections": [],
        "section_reviews": [],
        "blocking_issues": [reason],
        "improvement_actions": [improvement_action],
        "approved": False,
        "loop_status": loop_status,
        "loop_reason": reason,
        "loop_should_continue": False,
        "loop_completed": True,
        "loop_stagnated_rounds": 0,
        "review_score": 0,
        "review_feedback": feedback,
        "review_missing_sections": [],
        "review_blocking_issues": [reason],
        "review_improvement_actions": [improvement_action],
        "review_approved": False,
        "review_loop_status": loop_status,
        "review_loop_reason": reason,
        "review_loop_should_continue": False,
        "review_loop_completed": True,
        "review_loop_stagnated_rounds": 0,
        "review_score_confidence": None,
        "review_score_confidence_label": "unmeasured",
        "review_score_confidence_reason": "MAD confidence is unavailable because no LLM-generated assessments were produced.",
        "summary": f"{metadata.name} stopped in review round {review_round} because gameplay plan assessments were unavailable.",
    }


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name

    def review_plan(state: ReviewerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        review_round = int(state.get("review_round", 0)) + 1
        task_type = str(state.get("task_type", "feature") or "feature").strip().lower()
        reviewer_llm = context.get_llm("reviewer")
        if not reviewer_llm.is_enabled():
            return _blocked_review_response(
                artifact_dir=artifact_dir,
                metadata=metadata,
                review_round=review_round,
                reason="Reviewer LLM is unavailable, so gameplay plan assessments cannot be generated.",
                loop_status="llm-unavailable",
            )

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
            generated = reviewer_llm.generate_json(
                instructions=(
                    "You are gameplay-reviewer-workflow. Review a gameplay implementation plan like a strict senior gameplay engineer. "
                    "Score the plan against these exact sections: Overview, Task Type, Existing Docs, Implementation Steps, Unit Tests, "
                    "Risks, Acceptance Criteria. Approval requires a score >= 90, no blocking issues, and "
                    "no missing required sections. Focus on technical clarity, owner paths, regression coverage, and player-visible acceptance. "
                    f"Minimum final-approval depth is {MIN_REVIEW_ROUNDS} review rounds. Do not approve early just because round one sounds plausible. "
                    "Hard blockers: player outcome, grounded current-behavior evidence, speculation control around the owning runtime path, and adjacent-path regression coverage. "
                    "Do not block on review-round bookkeeping, artifact naming, sign-off records, or other process-only approval trace details."
                ),
                input_text=build_prompt_brief(
                    opening="Review the current gameplay implementation plan as a strict senior gameplay engineer.",
                    sections=[
                        ("Task request", state["task_prompt"].strip()),
                        (
                            "Review context",
                            "\n".join(
                                [
                                    f"- Task type: {task_type}",
                                    f"- Execution track: {state.get('execution_track', task_type)}",
                                    f"- Review round: {review_round}",
                                ]
                            ),
                        ),
                        (
                            "Hard blocker and scoring rules",
                            "\n".join(
                                [
                                    "- [hard blocker] Player Outcome: the plan must name the player-visible result and scope boundary.",
                                    "- [hard blocker] Current Behavior Evidence: the plan must cite grounded docs, runtime paths, and the owner.",
                                    "- [hard blocker] Speculation Control: implementation steps must stay anchored on current ownership.",
                                    "- [hard blocker] Edge and Regression Coverage: tests and acceptance criteria must protect adjacent gameplay paths.",
                                ]
                            ),
                        ),
                        ("Plan document", state["plan_doc"].strip()),
                    ],
                    closing=(
                        "Score it hard, keep the feedback technical, and require another independent verification "
                        "pass before final approval can stick. Do not drift into process-only asks."
                    ),
                ),
                schema_name="gameplay_plan_review",
                schema=schema,
            )
            raw_section_reviews = [
                {
                    "section": str(item["section"]).strip(),
                    "score": max(0, int(item["score"])),
                    "status": str(item["status"]).strip(),
                    "rationale": str(item["rationale"]).strip(),
                    "action_items": [str(action).strip() for action in item.get("action_items", []) if str(action).strip()],
                }
                for item in generated.get("section_reviews", [])
                if str(item.get("section", "")).strip()
            ]
            expected_sections = {section for section, _ in SECTION_WEIGHTS}
            received_sections = {item["section"] for item in raw_section_reviews}
            if len(raw_section_reviews) != len(SECTION_WEIGHTS) or received_sections != expected_sections:
                raise ValueError("Reviewer LLM did not return the full gameplay plan assessment set.")
            normalized_section_reviews = _normalize_section_reviews(raw_section_reviews)
            review_result = {
                "score": sum(item["score"] for item in normalized_section_reviews),
                "feedback": str(generated.get("feedback", "")).strip(),
                "approved": bool(generated.get("approved", False)),
                "missing_sections": _dedupe([str(item) for item in generated.get("missing_sections", []) if str(item).strip()]),
                "section_reviews": normalized_section_reviews,
                "blocking_issues": _dedupe([str(item) for item in generated.get("blocking_issues", []) if str(item).strip()]),
                "improvement_actions": _dedupe(
                    [str(item) for item in generated.get("improvement_actions", []) if str(item).strip()]
                ),
            }
            review_result = _apply_process_drift_guardrails(
                review_result=review_result,
            )
        except (LLMError, TypeError, ValueError) as exc:
            return _blocked_review_response(
                artifact_dir=artifact_dir,
                metadata=metadata,
                review_round=review_round,
                reason=f"Reviewer LLM failed to produce usable gameplay plan assessments: {exc}",
                loop_status="llm-error",
            )

        if review_round < MIN_REVIEW_ROUNDS:
            enforced_actions = _dedupe([*review_result["improvement_actions"], MANDATORY_VERIFICATION_ACTION])
            notes = "\n".join(
                item
                for item in [
                    str(review_result.get("feedback", "")).strip(),
                    f"Minimum review depth is {MIN_REVIEW_ROUNDS} rounds, so this plan still needs one more independent pass.",
                ]
                if item
            ).strip()
            review_result = {
                **review_result,
                "approved": False,
                "improvement_actions": enforced_actions,
                "feedback": notes,
            }

        score_decision = evaluate_score_decision(
            PLAN_REVIEW_SCORE_POLICY,
            round_index=review_round,
            assessments=_to_score_assessments(review_result["section_reviews"]),
            explicit_approval=bool(review_result["approved"]),
            blocking_issues=review_result["blocking_issues"],
            missing_sections=review_result["missing_sections"],
            improvement_actions=review_result["improvement_actions"],
            artifact_dir=artifact_dir,
        )
        previous_score = int(state.get("score", 0)) if review_round > 1 else None
        prior_stagnated_rounds = int(state.get("loop_stagnated_rounds", 0)) if review_round > 1 else 0
        progress = evaluate_quality_loop(
            PLAN_REVIEW_SPEC,
            round_index=review_round,
            score=score_decision.score,
            approved=score_decision.approved,
            missing_sections=score_decision.missing_sections,
            blocking_issues=score_decision.blocking_issues,
            improvement_actions=score_decision.improvement_actions,
            previous_score=previous_score,
            prior_stagnated_rounds=prior_stagnated_rounds,
        )
        feedback = _compose_feedback(
            review_round=review_round,
            score=progress.score,
            approved=progress.approved,
            confidence_label=score_decision.confidence_label,
            confidence_reason=score_decision.confidence_reason,
            confidence=score_decision.confidence,
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
            "review_score_confidence": score_decision.confidence,
            "review_score_confidence_label": score_decision.confidence_label,
            "review_score_confidence_reason": score_decision.confidence_reason,
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
