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
from core.scoring import ScoreAssessment, ScorePolicy, evaluate_score_decision
from core.text_utils import normalize_text, slugify


APPROVAL_SCORE = 90
MIN_REVIEW_ROUNDS = 2
MAX_REVIEW_ROUNDS = 3
LOOP_SPEC = QualityLoopSpec(
    loop_id="template-investigation-review",
    threshold=APPROVAL_SCORE,
    max_rounds=MAX_REVIEW_ROUNDS,
    min_rounds=MIN_REVIEW_ROUNDS,
    require_blocker_free=True,
    require_missing_section_free=False,
    require_explicit_approval=True,
    min_score_delta=1,
    stagnation_limit=2,
)
REVIEW_SCORE_POLICY = ScorePolicy(
    system_id="template-investigation-review",
    threshold=APPROVAL_SCORE,
    require_blocker_free=True,
    require_missing_section_free=False,
    require_explicit_approval=True,
)
MANDATORY_VERIFICATION_ACTION = (
    "Run one more investigation pass that independently re-validates the causal chain with fresh evidence, "
    "a concrete call-order proof, or a read-only reproduction result before final handoff."
)
REVIEW_CRITERIA = (
    ("Focus", 25, "Task Framing", "Root Cause Hypothesis"),
    ("Evidence & Ownership", 20, "Project Root Findings", "Candidate Ownership"),
    ("Architecture", 20, "Architecture Notes"),
    ("Clean Code", 15, "Clean Code Notes"),
    ("Optimization", 10, "Optimization Notes"),
    ("Verification", 10, "Verification Plan"),
)
PROCESS_ONLY_FEEDBACK_KEYWORDS = (
    "dri",
    "merge authority",
    "sign off",
    "sign-off",
    "signoff",
    "accountability",
    "named test owner",
    "owner assignment",
    "commit",
    "pull request",
    "pr ",
    "provenance",
    "regression-origin",
    "regression origin",
    "behavioral delta",
    "what changed and where",
    "regression boundary",
    "stakeholder",
    "approval chain",
    "release gate",
    "release sign",
    "qa handoff",
    "qa sign",
    "deployment plan",
    "rollback plan",
    "jira",
    "ticket",
    "sprint",
    "backlog",
    "code review",
    "peer review",
    "change management",
)
_PROCESS_PATTERNS = (
    re.compile(r"assign\s+(a\s+)?named\b", re.IGNORECASE),
    re.compile(r"before\s+(implementation|deployment|release)\s+starts", re.IGNORECASE),
    re.compile(r"who\s+(is|will be)\s+(responsible|accountable)", re.IGNORECASE),
    re.compile(r"\borganizational\b", re.IGNORECASE),
)


REVIEWER_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["APPROVE", "REVISE"]},
        "overall_score": {"type": "integer"},
        "criterion_scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "score": {"type": "integer"},
                    "max_score": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["criterion", "score", "max_score", "rationale"],
                "additionalProperties": False,
            },
        },
        "blocking_issues": {"type": "array", "items": {"type": "string"}},
        "improvement_actions": {"type": "array", "items": {"type": "string"}},
        "senior_notes": {"type": "string"},
    },
    "required": ["decision", "overall_score", "criterion_scores", "blocking_issues", "improvement_actions", "senior_notes"],
    "additionalProperties": False,
}


class CriterionAssessment(TypedDict):
    criterion: str
    score: int
    max_score: int
    status: str
    rationale: str
    action_items: list[str]


class ReviewerState(TypedDict):
    task_prompt: str
    task_id: NotRequired[str]
    run_dir: NotRequired[str]
    artifact_dir: NotRequired[str]
    investigation_doc: str
    review_round: int
    review_doc: str
    review_score: int
    review_feedback: str
    review_blocking_issues: list[str]
    review_improvement_actions: list[str]
    review_criterion_scores: list[CriterionAssessment]
    review_approved: bool
    loop_status: str
    loop_reason: str
    loop_should_continue: bool
    loop_completed: bool
    loop_stagnated_rounds: int
    review_score_confidence: NotRequired[float | None]
    review_score_confidence_label: NotRequired[str]
    review_score_confidence_reason: NotRequired[str]
    summary: str


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in items:
        item = str(raw).strip()
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

def _to_score_assessments(criterion_scores: list[CriterionAssessment]) -> list[ScoreAssessment]:
    return [
        ScoreAssessment(
            label=item["criterion"],
            score=int(item["score"]),
            max_score=int(item["max_score"]),
            status=item["status"],
            rationale=item["rationale"],
            action_items=tuple(item["action_items"]),
        )
        for item in criterion_scores
    ]


def _is_process_only_feedback(text: str) -> bool:
    lowered = str(text).strip().lower()
    if not lowered:
        return True
    if any(keyword in lowered for keyword in PROCESS_ONLY_FEEDBACK_KEYWORDS):
        return True
    return any(pattern.search(text) for pattern in _PROCESS_PATTERNS)


def _compose_review_doc(
    score: int,
    approved: bool,
    criterion_scores: list[CriterionAssessment],
    blocking_issues: list[str],
    improvement_actions: list[str],
    senior_notes: str,
    confidence: float | None = None,
    confidence_label: str = "unmeasured",
    confidence_reason: str = "",
) -> str:
    lines = [
        "# Investigation Review",
        "",
        f"Decision: {'APPROVE' if approved else 'REVISE'}",
        f"Overall Score: {score}/100",
        f"Scoring Confidence: {confidence_label}{f' ({confidence:.1f}x noise floor)' if confidence is not None else ''}",
        f"Confidence Reason: {confidence_reason}",
        "",
        "## Criterion Scores",
    ]
    for item in criterion_scores:
        lines.append(f"- {item['criterion']}: {item['score']}/{item['max_score']} - {item['rationale']}")
    lines.extend(["", "## Blocking Issues"])
    lines.extend([f"- {item}" for item in blocking_issues] or ["- None."])
    lines.extend(["", "## Improvement Checklist"])
    lines.extend([f"- [ ] {item}" for item in improvement_actions] or ["- [x] No further investigation changes requested."])
    lines.extend(["", "## Senior Engineer Notes", senior_notes.strip() or "The investigation is ready for handoff."])
    return "\n".join(lines)


def _extract_heading_block(document: str, heading: str) -> list[str]:
    lines = document.splitlines()
    active = False
    collected: list[str] = []
    for line in lines:
        if line.strip() == f"## {heading}":
            active = True
            continue
        if active and line.startswith("## "):
            break
        if active and line.strip():
            collected.append(line.rstrip())
    return collected


def _parse_review_doc_or_raise(review_doc: str) -> dict[str, Any]:
    decision_match = re.search(r"Decision:\s*(APPROVE|REVISE)", review_doc, flags=re.IGNORECASE)
    score_match = re.search(r"Overall Score:\s*(\d{1,3})\s*/\s*100", review_doc, flags=re.IGNORECASE)
    if decision_match is None or score_match is None:
        raise ValueError("Reviewer LLM response is missing the required Decision or Overall Score fields.")

    max_score_lookup = {name: weight for name, weight, *_ in REVIEW_CRITERIA}
    parsed_scores: list[CriterionAssessment] = []
    seen_criteria: set[str] = set()
    for line in _extract_heading_block(review_doc, "Criterion Scores"):
        match = re.match(
            r"-\s*(?P<criterion>[^:]+):\s*(?P<score>\d{1,3})\s*/\s*(?P<max>\d{1,3})\s*-\s*(?P<rationale>.+)",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        criterion = match.group("criterion").strip()
        if criterion not in max_score_lookup or criterion in seen_criteria:
            continue
        seen_criteria.add(criterion)
        max_score = max_score_lookup[criterion]
        item_score = max(0, min(int(match.group("score")), max_score))
        status = "pass" if item_score >= max_score else "needs-work"
        if item_score == 0:
            status = "missing"
        parsed_scores.append(
            {
                "criterion": criterion,
                "score": item_score,
                "max_score": max_score,
                "status": status,
                "rationale": match.group("rationale").strip(),
                "action_items": [],
            }
        )

    if len(parsed_scores) != len(REVIEW_CRITERIA):
        raise ValueError("Reviewer LLM did not return the full investigation criterion assessment set.")

    ordered_scores = [{**next(item for item in parsed_scores if item["criterion"] == name)} for name, *_ in REVIEW_CRITERIA]
    raw_blocking_issues: list[str] = []
    for line in _extract_heading_block(review_doc, "Blocking Issues"):
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if not item.lower().startswith("none."):
                raw_blocking_issues.append(item)
    blocking_issues = _dedupe([item for item in raw_blocking_issues if not _is_process_only_feedback(item)])

    raw_improvement_actions: list[str] = []
    for line in _extract_heading_block(review_doc, "Improvement Checklist"):
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        item = re.sub(r"^-\s*\[[ xX]\]\s*", "", stripped).strip()
        if not item.lower().startswith("no further investigation changes requested."):
            raw_improvement_actions.append(item)
    improvement_actions = _dedupe([item for item in raw_improvement_actions if not _is_process_only_feedback(item)])

    score = sum(item["score"] for item in ordered_scores)
    explicit_decision = decision_match.group(1).strip().lower() == "approve"
    approved = explicit_decision or (score >= APPROVAL_SCORE and not blocking_issues)
    final_review_doc = _compose_review_doc(
        score,
        approved,
        ordered_scores,
        blocking_issues,
        improvement_actions,
        "\n".join(_extract_heading_block(review_doc, "Senior Engineer Notes")).strip(),
    )
    return {
        "review_doc": final_review_doc,
        "score": score,
        "approved": approved,
        "criterion_scores": ordered_scores,
        "blocking_issues": blocking_issues,
        "improvement_actions": improvement_actions,
    }


def _parse_review_json(data: dict[str, Any]) -> dict[str, Any]:
    max_score_lookup = {name: weight for name, weight, *_ in REVIEW_CRITERIA}
    parsed_scores: list[CriterionAssessment] = []
    seen_criteria: set[str] = set()
    for item in data.get("criterion_scores", []):
        criterion = str(item.get("criterion", "")).strip()
        if criterion not in max_score_lookup or criterion in seen_criteria:
            continue
        seen_criteria.add(criterion)
        max_score = max_score_lookup[criterion]
        item_score = max(0, min(int(item.get("score", 0)), max_score))
        status = "pass" if item_score >= max_score else "needs-work"
        if item_score == 0:
            status = "missing"
        parsed_scores.append(
            {
                "criterion": criterion,
                "score": item_score,
                "max_score": max_score,
                "status": status,
                "rationale": str(item.get("rationale", "")).strip(),
                "action_items": [],
            }
        )

    for name, weight, *_ in REVIEW_CRITERIA:
        if name not in seen_criteria:
            parsed_scores.append(
                {
                    "criterion": name,
                    "score": 0,
                    "max_score": weight,
                    "status": "missing",
                    "rationale": "Criterion not assessed by reviewer.",
                    "action_items": [],
                }
            )

    ordered_scores = [{**next(item for item in parsed_scores if item["criterion"] == name)} for name, *_ in REVIEW_CRITERIA]
    raw_blocking_issues = [str(item).strip() for item in data.get("blocking_issues", []) if str(item).strip() and not str(item).strip().lower().startswith("none")]
    blocking_issues = _dedupe([item for item in raw_blocking_issues if not _is_process_only_feedback(item)])
    raw_improvement_actions = [str(item).strip() for item in data.get("improvement_actions", []) if str(item).strip() and not str(item).strip().lower().startswith("no further")]
    improvement_actions = _dedupe([item for item in raw_improvement_actions if not _is_process_only_feedback(item)])

    score = sum(item["score"] for item in ordered_scores)
    decision = str(data.get("decision", "REVISE")).strip().upper()
    explicit_decision = decision == "APPROVE"
    approved = explicit_decision or (score >= APPROVAL_SCORE and not blocking_issues)
    final_review_doc = _compose_review_doc(
        score,
        approved,
        ordered_scores,
        blocking_issues,
        improvement_actions,
        str(data.get("senior_notes", "")).strip(),
    )
    return {
        "review_doc": final_review_doc,
        "score": score,
        "approved": approved,
        "criterion_scores": ordered_scores,
        "blocking_issues": blocking_issues,
        "improvement_actions": improvement_actions,
    }


def _blocked_review_response(
    *,
    artifact_path: Path,
    metadata: WorkflowMetadata,
    review_round: int,
    reason: str,
    loop_status: str,
) -> dict[str, Any]:
    improvement_action = "Enable the reviewer LLM and rerun investigation review so the scoring assessments come from LLM output."
    review_doc = _compose_review_doc(
        0,
        False,
        [],
        [reason],
        [improvement_action],
        reason,
        confidence_label="unmeasured",
        confidence_reason="MAD confidence is unavailable because no LLM-generated assessments were produced.",
    )
    (artifact_path / f"review_round_{review_round}.md").write_text(review_doc, encoding="utf-8")
    return {
        "review_round": review_round,
        "artifact_dir": str(artifact_path),
        "review_doc": review_doc,
        "review_score": 0,
        "review_feedback": review_doc,
        "review_blocking_issues": [reason],
        "review_improvement_actions": [improvement_action],
        "review_criterion_scores": [],
        "review_approved": False,
        "loop_status": loop_status,
        "loop_reason": reason,
        "loop_should_continue": False,
        "loop_completed": True,
        "loop_stagnated_rounds": 0,
        "review_score_confidence": None,
        "review_score_confidence_label": "unmeasured",
        "review_score_confidence_reason": "MAD confidence is unavailable because no LLM-generated assessments were produced.",
        "summary": f"{metadata.name} stopped in review round {review_round} because investigation assessments were unavailable.",
    }


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name

    def review(state: ReviewerState) -> dict[str, Any]:
        review_round = int(state.get("review_round", 0)) + 1
        artifact_path = _artifact_dir(context, metadata, state)
        reviewer_llm = context.get_llm("reviewer")
        if not reviewer_llm.is_enabled():
            return _blocked_review_response(
                artifact_path=artifact_path,
                metadata=metadata,
                review_round=review_round,
                reason="Reviewer LLM is unavailable, so investigation assessments cannot be generated.",
                loop_status="llm-unavailable",
            )

        review_instructions = (
            "You are a strict senior engineer reviewing an investigation brief. Score it hard against focus, evidence and ownership, "
            "architecture, clean code thinking, optimization awareness, and verification quality. "
            f"The criteria are: {', '.join(f'{name} ({weight} pts)' for name, weight, *_ in REVIEW_CRITERIA)}. "
            f"Minimum final-approval depth is {MIN_REVIEW_ROUNDS} review rounds. If the current round is below that floor, require one more "
            "pass that independently re-validates the causal chain with fresh evidence, clearer ordering proof, or a read-only reproduction. "
            "Do not approve early just because the first brief sounds plausible. "
            "Only gate on technical investigation quality. Do not require organizational ownership assignment, DRI naming, commit/PR provenance, "
            "or other process artifacts unless they are explicitly present in the provided evidence."
        )
        review_input = (
            f"Task prompt:\n{state['task_prompt']}\n\n"
            f"Review round: {review_round}/{MAX_REVIEW_ROUNDS}\n\n"
            f"Minimum rounds required before final approval can stick: {MIN_REVIEW_ROUNDS}\n\n"
            f"Investigation document:\n{state['investigation_doc']}\n\n"
            "Act like a demanding senior engineer who cares about clean code, focus, optimization, architecture, and validation quality."
        )
        try:
            try:
                review_json = reviewer_llm.generate_json(
                    instructions=review_instructions,
                    input_text=review_input,
                    schema_name="investigation_review",
                    schema=REVIEWER_OUTPUT_SCHEMA,
                )
                review_result = _parse_review_json(review_json)
            except (LLMError, ValueError, KeyError, TypeError, AttributeError):
                generated_review = reviewer_llm.generate_text(
                    instructions=(
                        review_instructions + " Return markdown using this exact shape: "
                        "# Investigation Review, Decision: APPROVE or REVISE, Overall Score: NN/100, ## Criterion Scores, ## Blocking Issues, "
                        "## Improvement Checklist, ## Senior Engineer Notes. Use one bullet per criterion in the form `- Criterion: score/max - rationale`. "
                        "If there are no blocking issues, write exactly `- None.` under Blocking Issues. "
                        "If there are no further investigation changes requested, write exactly `- [x] No further investigation changes requested.` "
                        "Keep Overall Score numerically consistent with the criterion bullets. Do not use JSON."
                    ),
                    input_text=review_input,
                )
                review_result = _parse_review_doc_or_raise(generated_review)
        except (LLMError, ValueError) as exc:
            return _blocked_review_response(
                artifact_path=artifact_path,
                metadata=metadata,
                review_round=review_round,
                reason=f"Reviewer LLM failed to produce usable investigation assessments: {exc}",
                loop_status="llm-error",
            )

        if review_round < MIN_REVIEW_ROUNDS:
            enforced_actions = _dedupe([*review_result["improvement_actions"], MANDATORY_VERIFICATION_ACTION])
            enforced_notes = "\n".join(
                item
                for item in [
                    "\n".join(_extract_heading_block(review_result["review_doc"], "Senior Engineer Notes")).strip(),
                    f"Minimum verification depth is {MIN_REVIEW_ROUNDS} rounds, so this brief still needs one more independent pass.",
                ]
                if item
            ).strip()
            review_result = {
                **review_result,
                "approved": False,
                "improvement_actions": enforced_actions,
                "review_doc": _compose_review_doc(
                    review_result["score"],
                    False,
                    review_result["criterion_scores"],
                    review_result["blocking_issues"],
                    enforced_actions,
                    enforced_notes,
                ),
            }

        previous_score = int(state.get("review_score", 0)) if review_round > 1 else None
        prior_stagnated_rounds = int(state.get("loop_stagnated_rounds", 0)) if review_round > 1 else 0
        score_decision = evaluate_score_decision(
            REVIEW_SCORE_POLICY,
            round_index=review_round,
            assessments=_to_score_assessments(review_result["criterion_scores"]),
            explicit_approval=bool(review_result["approved"]),
            blocking_issues=review_result["blocking_issues"],
            improvement_actions=review_result["improvement_actions"],
            artifact_dir=artifact_path,
        )
        progress = evaluate_quality_loop(
            LOOP_SPEC,
            round_index=review_round,
            score=score_decision.score,
            approved=score_decision.approved,
            blocking_issues=score_decision.blocking_issues,
            previous_score=previous_score,
            prior_stagnated_rounds=prior_stagnated_rounds,
            improvement_actions=score_decision.improvement_actions,
        )

        senior_notes = "\n".join(_extract_heading_block(review_result["review_doc"], "Senior Engineer Notes")).strip()
        final_review_doc = _compose_review_doc(
            progress.score,
            progress.approved,
            review_result["criterion_scores"],
            list(score_decision.blocking_issues),
            list(score_decision.improvement_actions),
            senior_notes,
            confidence=score_decision.confidence,
            confidence_label=score_decision.confidence_label,
            confidence_reason=score_decision.confidence_reason,
        )
        (artifact_path / f"review_round_{review_round}.md").write_text(final_review_doc, encoding="utf-8")
        summary = (
            f"{metadata.name} approved investigation review round {review_round} with score {progress.score}/100."
            if progress.approved
            else (
                f"{metadata.name} stopped after review round {review_round} with score {progress.score}/100. Loop status: {progress.status}."
                if progress.completed
                else f"{metadata.name} scored {progress.score}/100 in review round {review_round} and requested another investigation pass."
            )
        )
        return {
            "review_round": review_round,
            "artifact_dir": str(artifact_path),
            "review_doc": final_review_doc,
            "review_score": progress.score,
            "review_feedback": final_review_doc,
            "review_blocking_issues": list(score_decision.blocking_issues),
            "review_improvement_actions": list(score_decision.improvement_actions),
            "review_criterion_scores": list(review_result["criterion_scores"]),
            "review_approved": progress.approved,
            "loop_status": progress.status,
            "loop_reason": progress.reason,
            "loop_should_continue": progress.should_continue,
            "loop_completed": progress.completed,
            "loop_stagnated_rounds": progress.stagnated_rounds,
            "review_score_confidence": score_decision.confidence,
            "review_score_confidence_label": score_decision.confidence_label,
            "review_score_confidence_reason": score_decision.confidence_reason,
            "summary": summary,
        }

    graph = StateGraph(ReviewerState)
    graph.add_node("review", trace_graph_node(graph_name=graph_name, node_name="review", node_fn=review))
    graph.add_edge(START, "review")
    graph.add_edge("review", END)
    return graph.compile()
