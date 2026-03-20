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
BLOCKING_SECTIONS = {"Task Type", "Implementation Steps", "Unit Tests", "Acceptance Criteria"}
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


def _is_process_only_section_review(item: SectionReview) -> bool:
    if item["status"] == "pass":
        return False
    signals = [item["rationale"], *item["action_items"]]
    normalized_signals = [signal for signal in signals if str(signal).strip()]
    return bool(normalized_signals) and all(_is_process_only_review_item(signal) for signal in normalized_signals)


def _apply_process_drift_guardrails(
    *,
    plan_doc: str,
    review_result: dict[str, Any],
    fallback_section_reviews: list[SectionReview],
) -> dict[str, Any]:
    fallback_lookup = {item["section"]: item for item in fallback_section_reviews}
    sanitized_section_reviews: list[SectionReview] = []
    for item in review_result["section_reviews"]:
        fallback = fallback_lookup.get(item["section"], item)
        sanitized_section_reviews.append(fallback if _is_process_only_section_review(item) else item)

    generated_missing_sections = _dedupe([str(item) for item in review_result.get("missing_sections", []) if str(item).strip()])
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
    derived_missing_sections, derived_blocking_issues, derived_improvement_actions = _derive_review_requirements(
        plan_doc,
        sanitized_section_reviews,
    )
    score = sum(item["score"] for item in sanitized_section_reviews)
    missing_sections = _dedupe([*generated_missing_sections, *derived_missing_sections])
    blocking_issues = _dedupe([*generated_blocking_issues, *derived_blocking_issues])
    improvement_actions = _dedupe([*generated_improvement_actions, *derived_improvement_actions])
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


def _contains_path_hint(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+", text))


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
    fallback_reviews: list[SectionReview],
) -> list[SectionReview]:
    fallback_lookup = {item["section"]: item for item in fallback_reviews}
    normalized_lookup: dict[str, SectionReview] = {}

    for item in section_reviews:
        section = str(item.get("section", "")).strip()
        if section not in SECTION_WEIGHT_MAP or section in normalized_lookup:
            continue
        weight = SECTION_WEIGHT_MAP[section]
        fallback = fallback_lookup[section]
        score = max(0, min(int(item.get("score", 0)), weight))
        raw_status = str(item.get("status", "")).strip().lower()
        if raw_status == "missing" or score == 0:
            status = "missing"
        elif raw_status in {"pass", "approved"} and score >= weight:
            status = "pass"
        else:
            status = "needs-work"
        if section in BLOCKING_SECTIONS and status == "needs-work":
            score = min(max(1, score), max(1, round(weight * 0.6)))
        action_items = _dedupe(
            [str(action).strip() for action in item.get("action_items", []) if str(action).strip()]
        )
        normalized_lookup[section] = {
            "section": section,
            "score": score,
            "status": status,
            "rationale": str(item.get("rationale", "")).strip() or fallback["rationale"],
            "action_items": [] if status == "pass" else (action_items or list(fallback["action_items"])),
        }

    normalized: list[SectionReview] = []
    for section, _ in SECTION_WEIGHTS:
        normalized.append(normalized_lookup.get(section, fallback_lookup[section]))
    return normalized


def _derive_review_requirements(
    plan_doc: str,
    section_reviews: list[SectionReview],
) -> tuple[list[str], list[str], list[str]]:
    sections = _parse_sections(plan_doc)
    review_lookup = {item["section"]: item for item in section_reviews}
    missing_sections = [item["section"] for item in section_reviews if item["status"] == "missing"]
    blocking_issues = _dedupe(
        [
            f"{item['section']}: {(item['action_items'] or [_fallback_action(item['section'])])[0]}"
            for item in section_reviews
            if item["section"] in BLOCKING_SECTIONS and item["status"] != "pass"
        ]
    )
    improvement_actions = _dedupe(
        [
            action
            for item in section_reviews
            if item["status"] != "pass"
            for action in (item["action_items"] or [_fallback_action(item["section"])])
        ]
    )

    overview_lines = _clean_lines(sections.get("Overview", ""))
    acceptance_lines = _clean_lines(sections.get("Acceptance Criteria", ""))
    docs_text = sections.get("Existing Docs", "")
    implementation_text = sections.get("Implementation Steps", "")
    tests_text = sections.get("Unit Tests", "")
    risks_text = sections.get("Risks", "")
    lower_combined = "\n".join([tests_text, risks_text, "\n".join(acceptance_lines)]).lower()

    hard_blockers: list[str] = []
    hard_actions: list[str] = []
    if len(overview_lines) < 2 or len(acceptance_lines) < 2:
        hard_blockers.append(
            "Player Outcome: Name the player-visible result and the nearby gameplay boundary that must remain stable."
        )
        hard_actions.append("Clarify the player-visible outcome and the nearby gameplay boundary.")
    if not (_contains_path_hint(docs_text) and _contains_path_hint(implementation_text)):
        hard_blockers.append(
            "Current Behavior Evidence: Cite grounded docs, runtime paths, and the owning gameplay file before implementation."
        )
        hard_actions.append("Ground the plan in concrete docs, runtime paths, and the owning gameplay file.")
    if not _contains_path_hint(implementation_text) or "confirm the owning" in implementation_text.lower():
        hard_blockers.append(
            "Speculation Control: Anchor the implementation steps on the current runtime owner instead of an unconfirmed path."
        )
        hard_actions.append("Anchor the implementation steps on the current runtime owner instead of an unconfirmed path.")
    if len(_clean_lines(tests_text)) < 2 or not any(
        marker in lower_combined for marker in ("neighbor", "adjacent", "unchanged", "regression", "edge", "non-")
    ):
        hard_blockers.append(
            "Edge and Regression Coverage: Protect the adjacent gameplay path with explicit regression tests and acceptance criteria."
        )
        hard_actions.append(
            "Protect the adjacent gameplay path with explicit regression tests and acceptance criteria."
        )

    return (
        _dedupe([*missing_sections]),
        _dedupe([*blocking_issues, *hard_blockers]),
        _dedupe([*improvement_actions, *hard_actions]),
    )


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
    missing_sections, blocking_issues, improvement_actions = _derive_review_requirements(plan_doc, section_reviews)
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
                normalized_section_reviews = _normalize_section_reviews(
                    [
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
                    ],
                    fallback["section_reviews"],
                )
                review_result = {
                    "score": sum(item["score"] for item in normalized_section_reviews),
                    "approved": bool(generated.get("approved", False)),
                    "missing_sections": _dedupe(
                        [str(item) for item in generated.get("missing_sections", []) if str(item).strip()]
                    ),
                    "section_reviews": normalized_section_reviews,
                    "blocking_issues": _dedupe(
                        [str(item) for item in generated.get("blocking_issues", []) if str(item).strip()]
                    ),
                    "improvement_actions": _dedupe(
                        [str(item) for item in generated.get("improvement_actions", []) if str(item).strip()]
                    ),
                }
                review_result = _apply_process_drift_guardrails(
                    plan_doc=state["plan_doc"],
                    review_result=review_result,
                    fallback_section_reviews=fallback["section_reviews"],
                )
            except LLMError:
                review_result = fallback

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
