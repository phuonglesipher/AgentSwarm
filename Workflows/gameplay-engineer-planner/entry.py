from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_logging import trace_graph_node, trace_route_decision
from core.llm import LLMError
from core.models import WorkflowContext, WorkflowMetadata
from core.quality_loop import QualityLoopSpec, evaluate_quality_loop
from core.scoring import ScoreAssessment, ScorePolicy, evaluate_score_decision
from core.text_utils import keyword_tokens, normalize_text, slugify, tokenize


APPROVAL_SCORE = 90
MAX_PLANNING_ROUNDS = 3
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".lua",
    ".md",
    ".py",
    ".rb",
    ".rs",
    ".rst",
    ".sh",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
PLAN_LOOP_SPEC = QualityLoopSpec(
    loop_id="gameplay-solution-plan-review",
    threshold=APPROVAL_SCORE,
    max_rounds=MAX_PLANNING_ROUNDS,
    min_rounds=1,
    require_blocker_free=True,
    require_missing_section_free=True,
    require_explicit_approval=True,
    min_score_delta=1,
    stagnation_limit=2,
)
SECTION_WEIGHTS = (
    ("Problem Framing", 15),
    ("Current Context", 15),
    ("Proposed Solution", 20),
    ("Execution Plan", 20),
    ("Validation Plan", 15),
    ("Risks and Open Questions", 15),
)
SECTION_WEIGHT_MAP = dict(SECTION_WEIGHTS)
PLAN_SCORE_POLICY = ScorePolicy(
    system_id="gameplay-solution-plan-review",
    threshold=APPROVAL_SCORE,
    require_blocker_free=True,
    require_missing_section_free=True,
    require_explicit_approval=True,
)


class PlannerSectionReview(TypedDict):
    section: str
    score: int
    status: str
    rationale: str
    action_items: list[str]


class PlannerState(TypedDict):
    prompt: NotRequired[str]
    task_prompt: str
    task_id: NotRequired[str]
    run_dir: NotRequired[str]
    artifact_dir: str
    planning_round: int
    research_report: str
    solution_plan: str
    score: int
    feedback: str
    missing_sections: list[str]
    section_reviews: list[PlannerSectionReview]
    blocking_issues: list[str]
    improvement_actions: list[str]
    approved: bool
    loop_status: str
    loop_reason: str
    loop_should_continue: bool
    loop_completed: bool
    loop_stagnated_rounds: int
    score_confidence: NotRequired[float | None]
    score_confidence_label: NotRequired[str]
    score_confidence_reason: NotRequired[str]
    doc_hits: list[str]
    source_hits: list[str]
    test_hits: list[str]
    final_report: dict[str, Any]
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


def _artifact_dir(context: WorkflowContext, metadata: WorkflowMetadata, state: PlannerState) -> Path:
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


def _should_skip(path: Path, scope_root: Path, exclude_roots: tuple[str, ...]) -> bool:
    try:
        relative = path.relative_to(scope_root).as_posix().lower()
    except ValueError:
        return True
    for excluded in exclude_roots:
        normalized = excluded.replace("\\", "/").strip("/").lower()
        if not normalized:
            continue
        if relative == normalized or relative.startswith(f"{normalized}/"):
            return True
    return False


def _safe_read_text(path: Path, *, limit: int = 700) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def _resolve_roots(scope_root: Path, relative_roots: tuple[str, ...]) -> list[Path]:
    candidates = [scope_root / relative_root for relative_root in relative_roots]
    existing = [path for path in candidates if path.exists()]
    return existing or [scope_root]


def _find_relevant_files(
    *,
    scope_root: Path,
    relative_roots: tuple[str, ...],
    exclude_roots: tuple[str, ...],
    query_text: str,
    max_hits: int = 5,
) -> list[str]:
    query_tokens = keyword_tokens(query_text) or tokenize(query_text)
    scored: list[tuple[int, str]] = []
    for root in _resolve_roots(scope_root, relative_roots):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if _should_skip(path, scope_root, exclude_roots):
                continue
            relative_path = path.relative_to(scope_root).as_posix()
            haystack = f"{relative_path.replace('_', ' ')}\n{_safe_read_text(path).replace('_', ' ')}"
            normalized_haystack = normalize_text(haystack)
            score = sum(1 for token in query_tokens if token in normalized_haystack)
            if score > 0:
                scored.append((score, relative_path))
    scored.sort(key=lambda item: (-item[0], item[1].lower()))
    hits: list[str] = []
    for _, relative_path in scored:
        if relative_path in hits:
            continue
        hits.append(relative_path)
        if len(hits) >= max_hits:
            break
    return hits


def _collect_context(context: WorkflowContext, task_prompt: str) -> dict[str, list[str]]:
    scope_root = context.resolve_scope_root("host_project")
    return {
        "doc_hits": _find_relevant_files(
            scope_root=scope_root,
            relative_roots=context.config.doc_roots,
            exclude_roots=context.config.exclude_roots,
            query_text=task_prompt,
        ),
        "source_hits": _find_relevant_files(
            scope_root=scope_root,
            relative_roots=context.config.source_roots,
            exclude_roots=context.config.exclude_roots,
            query_text=task_prompt,
        ),
        "test_hits": _find_relevant_files(
            scope_root=scope_root,
            relative_roots=context.config.test_roots,
            exclude_roots=context.config.exclude_roots,
            query_text=task_prompt,
        ),
    }


def _format_bullets(items: list[str], *, empty_message: str = "None.") -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return f"- {empty_message}"
    return "\n".join(f"- {item}" for item in cleaned)


def _fallback_research_report(state: PlannerState, context_hits: dict[str, list[str]]) -> str:
    return "\n".join(
        [
            "# Gameplay Planner Research Report",
            "",
            "## Goal",
            f"- {state['task_prompt']}",
            "",
            "## Grounded Evidence",
            *_format_bullets(
                [*context_hits["doc_hits"], *context_hits["source_hits"], *context_hits["test_hits"]],
                empty_message="No grounded evidence found yet.",
            ).splitlines(),
            "",
            "## Constraints",
            "- Keep the solution scoped to gameplay behavior and avoid speculative non-gameplay changes.",
            "",
            "## Proposed Direction",
            "- Anchor the plan on the live gameplay runtime owner and the nearest regression test path.",
            "",
            "## Open Questions",
            "- Which neighboring gameplay state or path needs the strongest regression protection?",
        ]
    )


def _fallback_solution_plan(state: PlannerState, context_hits: dict[str, list[str]]) -> str:
    docs = context_hits["doc_hits"] or ["No strong gameplay docs found yet."]
    sources = context_hits["source_hits"] or ["No grounded runtime owner found yet."]
    tests = context_hits["test_hits"] or ["No grounded validation path found yet."]
    return "\n".join(
        [
            "# Gameplay Solution Plan",
            "",
            "## Problem Framing",
            f"- {state['task_prompt']}",
            "- Keep the change inside gameplay-only ownership and protect adjacent gameplay states.",
            "",
            "## Current Context",
            *_format_bullets([*docs, *sources, *tests]).splitlines(),
            "",
            "## Proposed Solution",
            "- Implement the change in the strongest current gameplay owner instead of broad helper code.",
            "- Keep the scope narrow and avoid speculative non-gameplay rewrites.",
            "",
            "## Execution Plan",
            "- Confirm the owning runtime path and the exact hook where the gameplay behavior should change.",
            "- Sequence the implementation so ownership, validation, and edge-case handling stay explicit.",
            "",
            "## Validation Plan",
            "- Add or update automated regression checks around the owning gameplay path.",
            "- Spell out the nearby gameplay path that must remain unchanged.",
            "",
            "## Risks and Open Questions",
            "- Risk: adjacent gameplay states could drift if the hook is too broad.",
            "- Open question: which neighboring path needs the most explicit regression coverage?",
        ]
    )


def _normalize_section_reviews(section_reviews: list[PlannerSectionReview]) -> list[PlannerSectionReview]:
    review_lookup = {item["section"]: item for item in section_reviews}
    normalized: list[PlannerSectionReview] = []
    for section, weight in SECTION_WEIGHTS:
        item = review_lookup[section]
        score = max(0, min(int(item["score"]), weight))
        raw_status = str(item["status"]).strip().lower()
        if raw_status == "missing" or score == 0:
            status = "missing"
        elif raw_status in {"pass", "approved"} and score >= weight:
            status = "pass"
        else:
            status = "needs-work"
        normalized.append(
            {
                "section": section,
                "score": score,
                "status": status,
                "rationale": str(item["rationale"]).strip(),
                "action_items": (
                    []
                    if status == "pass"
                    else _dedupe([str(action).strip() for action in item.get("action_items", []) if str(action).strip()])
                ),
            }
        )
    return normalized


def _to_score_assessments(section_reviews: list[PlannerSectionReview]) -> list[ScoreAssessment]:
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


def _compose_review_feedback(
    *,
    planning_round: int,
    score: int,
    approved: bool,
    confidence_label: str,
    confidence_reason: str,
    confidence: float | None,
    loop_status: str,
    loop_reason: str,
    section_reviews: list[PlannerSectionReview],
    missing_sections: list[str],
    blocking_issues: list[str],
    improvement_actions: list[str],
    reviewer_feedback: str,
) -> str:
    lines = [
        "# Gameplay Solution Plan Review",
        "",
        f"- Planning round: {planning_round}",
        f"- Approved: {approved}",
        f"- Score: {score}/100",
        f"- Scoring confidence: {confidence_label}{f' ({confidence:.1f}x noise floor)' if confidence is not None else ''}",
        f"- Confidence reason: {confidence_reason}",
        f"- Loop Status: {loop_status}",
        f"- Loop Reason: {loop_reason}",
        f"- Approval bar: >= {APPROVAL_SCORE}/100 and zero blocking issues",
        "",
        "## Hard Blocker And Scoring Rules",
        "- [hard blocker] Player Outcome: the plan must name the player-visible result and scope boundary.",
        "- [hard blocker] Current Behavior Evidence: the plan must point to grounded docs, runtime paths, or tests.",
        "- [hard blocker] Speculation Control: the proposal must stay anchored on current ownership instead of guesses.",
        "- [hard blocker] Edge and Regression Coverage: validation must protect adjacent gameplay paths explicitly.",
        "",
        "## Section Scores",
    ]
    weights = dict(SECTION_WEIGHTS)
    for item in section_reviews:
        lines.append(f"- {item['section']}: {item['score']}/{weights[item['section']]} - {item['rationale']}")
    lines.extend(["", "## Missing Sections"])
    lines.extend([f"- {item}" for item in missing_sections] or ["- None."])
    lines.extend(["", "## Blocking Issues"])
    lines.extend([f"- {item}" for item in blocking_issues] or ["- None."])
    lines.extend(["", "## Improvement Actions"])
    lines.extend([f"- {item}" for item in improvement_actions] or ["- None."])
    lines.extend(["", "## Reviewer Notes", reviewer_feedback.strip() or "The plan is ready."])
    return "\n".join(lines)


def _blocked_review_response(
    *,
    artifact_dir: Path,
    metadata: WorkflowMetadata,
    planning_round: int,
    reason: str,
    loop_status: str,
) -> dict[str, Any]:
    improvement_action = "Enable the reviewer LLM and rerun planner review so the scoring assessments come from LLM output."
    feedback = _compose_review_feedback(
        planning_round=planning_round,
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
        reviewer_feedback=reason,
    )
    (artifact_dir / f"planner_review_round_{planning_round}.md").write_text(feedback, encoding="utf-8")
    return {
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
        "score_confidence": None,
        "score_confidence_label": "unmeasured",
        "score_confidence_reason": "MAD confidence is unavailable because no LLM-generated assessments were produced.",
        "summary": f"{metadata.name} stopped because gameplay solution-plan assessments were unavailable.",
    }


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name

    def research_solution(state: PlannerState) -> dict[str, Any]:
        planning_round = int(state.get("planning_round", 0)) + 1
        artifact_dir = _artifact_dir(context, metadata, state)
        context_hits = _collect_context(context, state["task_prompt"])
        research_report = _fallback_research_report(state, context_hits)
        if context.llm.is_enabled():
            try:
                research_report = context.llm.generate_text(
                    instructions=(
                        "Write a concise markdown gameplay planning research report with these exact sections: Goal, Grounded Evidence, "
                        "Constraints, Proposed Direction, Open Questions. Keep the report grounded in gameplay-only evidence."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Planning round: {planning_round}\n"
                        f"Doc hits:\n{_format_bullets(context_hits['doc_hits'])}\n\n"
                        f"Source hits:\n{_format_bullets(context_hits['source_hits'])}\n\n"
                        f"Test hits:\n{_format_bullets(context_hits['test_hits'])}\n\n"
                        f"Previous score: {state.get('score', 0)}/100\n"
                        f"Previous blocking issues:\n{_format_bullets(list(state.get('blocking_issues', [])))}\n\n"
                        f"Previous improvement actions:\n{_format_bullets(list(state.get('improvement_actions', [])))}\n"
                    ),
                )
            except LLMError:
                research_report = _fallback_research_report(state, context_hits)
        (artifact_dir / f"planner_research_round_{planning_round}.md").write_text(research_report, encoding="utf-8")
        (artifact_dir / "research_report.md").write_text(research_report, encoding="utf-8")
        return {
            "artifact_dir": str(artifact_dir),
            "planning_round": planning_round,
            "research_report": research_report,
            "doc_hits": context_hits["doc_hits"],
            "source_hits": context_hits["source_hits"],
            "test_hits": context_hits["test_hits"],
            "summary": f"{metadata.name} gathered gameplay planning evidence for round {planning_round}.",
        }

    def draft_solution_plan(state: PlannerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        context_hits = {
            "doc_hits": list(state.get("doc_hits", [])),
            "source_hits": list(state.get("source_hits", [])),
            "test_hits": list(state.get("test_hits", [])),
        }
        solution_plan = _fallback_solution_plan(state, context_hits)
        if context.llm.is_enabled():
            try:
                solution_plan = context.llm.generate_text(
                    instructions=(
                        "Rewrite the full markdown gameplay solution plan using these exact sections: Problem Framing, Current Context, "
                        "Proposed Solution, Execution Plan, Validation Plan, Risks and Open Questions. Keep the plan ready for implementation handoff."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Planning round: {state['planning_round']}\n"
                        f"Previous score: {state.get('score', 0)}/100\n"
                        f"Blocking issues:\n{_format_bullets(list(state.get('blocking_issues', [])))}\n\n"
                        f"Improvement actions:\n{_format_bullets(list(state.get('improvement_actions', [])))}\n\n"
                        f"Research report:\n{state['research_report']}\n"
                    ),
                )
            except LLMError:
                solution_plan = _fallback_solution_plan(state, context_hits)
        (artifact_dir / f"planner_plan_round_{state['planning_round']}.md").write_text(solution_plan, encoding="utf-8")
        (artifact_dir / "solution_plan.md").write_text(solution_plan, encoding="utf-8")
        return {
            "solution_plan": solution_plan,
            "summary": f"{metadata.name} drafted gameplay solution plan round {state['planning_round']}.",
        }

    def review_solution_plan(state: PlannerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        reviewer_llm = context.get_llm("reviewer")
        planning_round = int(state["planning_round"])
        if not reviewer_llm.is_enabled():
            return _blocked_review_response(
                artifact_dir=artifact_dir,
                metadata=metadata,
                planning_round=planning_round,
                reason="Reviewer LLM is unavailable, so gameplay solution-plan assessments cannot be generated.",
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
                    "You review gameplay solution plans before implementation. Score the plan hard against Problem Framing, Current Context, "
                    "Proposed Solution, Execution Plan, Validation Plan, and Risks and Open Questions. Return JSON only."
                ),
                input_text=(
                    f"Task prompt:\n{state['task_prompt']}\n\n"
                    f"Planning round: {planning_round}\n\n"
                    "Hard blocker and scoring rules:\n"
                    "- [hard blocker] Player Outcome: the plan must name the player-visible result.\n"
                    "- [hard blocker] Current Behavior Evidence: the plan must cite grounded docs, code, or tests.\n"
                    "- [hard blocker] Speculation Control: the plan must stay anchored on current runtime ownership.\n"
                    "- [hard blocker] Edge and Regression Coverage: the validation plan must protect neighboring gameplay paths.\n\n"
                    f"Solution plan:\n{state['solution_plan']}\n"
                ),
                schema_name="gameplay_solution_plan_review",
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
                raise ValueError("Reviewer LLM did not return the full gameplay solution-plan assessment set.")
            ordered_section_reviews = [
                next(item for item in raw_section_reviews if item["section"] == section)
                for section, _ in SECTION_WEIGHTS
            ]
            normalized_section_reviews = _normalize_section_reviews(ordered_section_reviews)
            review_result = {
                "score": sum(item["score"] for item in normalized_section_reviews),
                "feedback": str(generated.get("feedback", "")).strip(),
                "missing_sections": _dedupe(
                    [
                        *[str(item) for item in generated.get("missing_sections", []) if str(item).strip()],
                        *[item["section"] for item in normalized_section_reviews if item["status"] == "missing"],
                    ]
                ),
                "section_reviews": normalized_section_reviews,
                "blocking_issues": _dedupe([str(item) for item in generated.get("blocking_issues", []) if str(item).strip()]),
                "improvement_actions": _dedupe(
                    [str(item) for item in generated.get("improvement_actions", []) if str(item).strip()]
                ),
                "approved": bool(generated.get("approved", False)),
            }
        except (LLMError, TypeError, ValueError) as exc:
            return _blocked_review_response(
                artifact_dir=artifact_dir,
                metadata=metadata,
                planning_round=planning_round,
                reason=f"Reviewer LLM failed to produce usable gameplay solution-plan assessments: {exc}",
                loop_status="llm-error",
            )

        score_decision = evaluate_score_decision(
            PLAN_SCORE_POLICY,
            round_index=planning_round,
            assessments=_to_score_assessments(review_result["section_reviews"]),
            explicit_approval=bool(review_result.get("approved", False)),
            blocking_issues=review_result["blocking_issues"],
            missing_sections=review_result["missing_sections"],
            improvement_actions=review_result["improvement_actions"],
            artifact_dir=artifact_dir,
        )
        previous_score = int(state.get("score", 0)) if planning_round > 1 else None
        prior_stagnated_rounds = int(state.get("loop_stagnated_rounds", 0)) if planning_round > 1 else 0
        progress = evaluate_quality_loop(
            PLAN_LOOP_SPEC,
            round_index=planning_round,
            score=score_decision.score,
            approved=score_decision.approved,
            missing_sections=score_decision.missing_sections,
            blocking_issues=score_decision.blocking_issues,
            improvement_actions=score_decision.improvement_actions,
            previous_score=previous_score,
            prior_stagnated_rounds=prior_stagnated_rounds,
        )
        feedback = _compose_review_feedback(
            planning_round=planning_round,
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
            reviewer_feedback=review_result["feedback"],
        )
        (artifact_dir / f"planner_review_round_{planning_round}.md").write_text(feedback, encoding="utf-8")
        return {
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
            "score_confidence": score_decision.confidence,
            "score_confidence_label": score_decision.confidence_label,
            "score_confidence_reason": score_decision.confidence_reason,
            "summary": (
                f"{metadata.name} approved planning round {planning_round}."
                if progress.approved
                else (
                    f"{metadata.name} stopped after planning round {planning_round}."
                    if progress.completed
                    else f"{metadata.name} requested another planning round."
                )
            ),
        }

    def finalize(state: PlannerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        final_status = "completed" if state["approved"] else "review-blocked"
        confidence_value = state.get("score_confidence")
        confidence_suffix = f" ({confidence_value:.1f}x noise floor)" if confidence_value is not None else ""
        final_report = {
            "status": final_status,
            "planning_rounds": int(state.get("planning_round", 0)),
            "score": int(state.get("score", 0)),
            "approved": bool(state.get("approved", False)),
            "loop_status": str(state.get("loop_status", "unknown")),
            "loop_reason": str(state.get("loop_reason", "")),
            "score_confidence": state.get("score_confidence"),
            "score_confidence_label": str(state.get("score_confidence_label", "unmeasured")),
            "score_confidence_reason": str(state.get("score_confidence_reason", "")),
            "doc_hits": list(state.get("doc_hits", [])),
            "source_hits": list(state.get("source_hits", [])),
            "test_hits": list(state.get("test_hits", [])),
            "blocking_issues": list(state.get("blocking_issues", [])),
            "improvement_actions": list(state.get("improvement_actions", [])),
        }
        (artifact_dir / "final_report.md").write_text(
            "\n".join(
                [
                    "# Gameplay Planner Final Report",
                    "",
                    f"- Status: {final_status}",
                    f"- Planning rounds: {state.get('planning_round', 0)}",
                    f"- Score: {state.get('score', 0)}/100",
                    f"- Approved: {state.get('approved', False)}",
                    f"- Scoring confidence: {state.get('score_confidence_label', 'unmeasured')}{confidence_suffix}",
                    f"- Confidence reason: {state.get('score_confidence_reason', '')}",
                    f"- Loop Status: {state.get('loop_status', 'unknown')}",
                    f"- Loop Reason: {state.get('loop_reason', '')}",
                    "",
                    "## Blocking Issues",
                    *([f"- {item}" for item in state.get("blocking_issues", [])] or ["- None."]),
                    "",
                    "## Improvement Actions",
                    *([f"- {item}" for item in state.get("improvement_actions", [])] or ["- None."]),
                    "",
                    "## Latest Review",
                    state.get("feedback", ""),
                ]
            ),
            encoding="utf-8",
        )
        return {
            "artifact_dir": str(artifact_dir),
            "final_report": final_report,
            "summary": (
                f"{metadata.name} finished with an approved gameplay solution plan."
                if state["approved"]
                else f"{metadata.name} stopped before the gameplay solution plan reached approval."
            ),
        }

    def review_gate(state: PlannerState) -> str:
        return "research_solution" if state["loop_should_continue"] else "finalize"

    graph = StateGraph(PlannerState)
    graph.add_node(
        "research_solution",
        trace_graph_node(graph_name=graph_name, node_name="research_solution", node_fn=research_solution),
    )
    graph.add_node(
        "draft_solution_plan",
        trace_graph_node(graph_name=graph_name, node_name="draft_solution_plan", node_fn=draft_solution_plan),
    )
    graph.add_node(
        "review_solution_plan",
        trace_graph_node(graph_name=graph_name, node_name="review_solution_plan", node_fn=review_solution_plan),
    )
    graph.add_node("finalize", trace_graph_node(graph_name=graph_name, node_name="finalize", node_fn=finalize))
    graph.add_edge(START, "research_solution")
    graph.add_edge("research_solution", "draft_solution_plan")
    graph.add_edge("draft_solution_plan", "review_solution_plan")
    graph.add_conditional_edges(
        "review_solution_plan",
        trace_route_decision(graph_name=graph_name, router_name="review_gate", route_fn=review_gate),
        {"research_solution": "research_solution", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()
