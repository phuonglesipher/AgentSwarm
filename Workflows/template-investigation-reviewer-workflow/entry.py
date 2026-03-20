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
from core.text_utils import keyword_tokens, normalize_text, slugify, tokenize


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
BLOCKING_CRITERIA = {"Focus", "Evidence & Ownership", "Architecture", "Verification"}
CRITERION_ACTIONS = {
    "Focus": "Tighten the investigation around the most credible root cause instead of broad project commentary.",
    "Evidence & Ownership": "Name the concrete files, modules, docs, or tests that most likely own the issue.",
    "Architecture": "Explain the relevant component boundary or runtime handoff more clearly.",
    "Clean Code": "Call out maintainability, coupling, duplication, naming, or change-safety concerns explicitly.",
    "Optimization": "Mention likely hot paths, redundant work, or why optimization is intentionally a non-goal.",
    "Verification": "Add concrete validation steps, regression checks, or measurements for the hypothesis.",
}
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
)


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


def _partial_score(weight: int) -> int:
    return max(1, round(weight * 0.6))


def _criterion_sections(criterion: str) -> tuple[str, ...]:
    for name, _, *sections in REVIEW_CRITERIA:
        if name == criterion:
            return tuple(sections)
    return ()


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
    return any(keyword in lowered for keyword in PROCESS_ONLY_FEEDBACK_KEYWORDS)


def _criteria_for_feedback(text: str) -> set[str]:
    lowered = str(text).strip().lower()
    if not lowered or _is_process_only_feedback(lowered):
        return set()

    mapped: set[str] = set()
    if any(keyword in lowered for keyword in ("root cause", "hypothesis", "call-chain", "call chain", "scope", "focus")):
        mapped.add("Focus")
    if any(keyword in lowered for keyword in ("owner", "ownership", "file", "module", "path", "source", "doc", "test owner")):
        mapped.add("Evidence & Ownership")
    if any(keyword in lowered for keyword in ("architecture", "boundary", "handoff", "contract", "override", "state model")):
        mapped.add("Architecture")
    if any(keyword in lowered for keyword in ("clean code", "maintain", "coupling", "duplication", "naming", "readability", "change-safety")):
        mapped.add("Clean Code")
    if any(keyword in lowered for keyword in ("optimiz", "performance", "hot path", "cache", "allocation", "redund")):
        mapped.add("Optimization")
    if any(keyword in lowered for keyword in ("verify", "validation", "assert", "measure", "repro", "proof", "deterministic", "state-transition", "state transition")):
        mapped.add("Verification")
    return mapped


def _normalize_criterion_scores(
    criterion_scores: list[CriterionAssessment],
    *,
    blocking_issues: list[str],
) -> list[CriterionAssessment]:
    criteria_to_cap: set[str] = set()
    for item in blocking_issues:
        criteria_to_cap.update(_criteria_for_feedback(item))

    normalized: list[CriterionAssessment] = []
    for item in criterion_scores:
        criterion = item["criterion"]
        score = int(item["score"])
        max_score = int(item["max_score"])
        status = str(item["status"])
        rationale = str(item["rationale"]).strip()
        action_items = list(item["action_items"])

        if criterion in criteria_to_cap:
            capped_score = min(score, _partial_score(max_score))
            if capped_score != score or status == "pass":
                score = capped_score
                status = "needs-work" if score > 0 else "missing"
                action_items = _dedupe([*action_items, CRITERION_ACTIONS[criterion]])
                rationale = f"{rationale} Reviewer requested another technical pass here.".strip()

        normalized.append(
            {
                "criterion": criterion,
                "score": score,
                "max_score": max_score,
                "status": status,
                "rationale": rationale,
                "action_items": [] if status == "pass" else _dedupe(action_items or [CRITERION_ACTIONS[criterion]]),
            }
        )
    return normalized


def _assess_criterion(
    criterion: str,
    weight: int,
    sections: dict[str, str],
    task_prompt: str,
) -> CriterionAssessment:
    combined = "\n".join(sections.get(section_name, "") for section_name in _criterion_sections(criterion)).strip()
    lines = _clean_lines(combined)
    score = 0
    status = "missing"
    rationale = ""

    if combined:
        score = _partial_score(weight)
        status = "needs-work"

    if criterion == "Focus":
        tokens = keyword_tokens(task_prompt) or tokenize(task_prompt)
        if combined and len(lines) >= 3 and len(tokens & tokenize(combined)) >= 2:
            score, status = weight, "pass"
            rationale = "The investigation stays on the core task and names a credible root-cause direction."
        elif combined:
            rationale = "The investigation has a direction, but it still needs a tighter root-cause narrative."
        else:
            rationale = "The investigation does not frame the task tightly enough to guide the next step."
    elif criterion == "Evidence & Ownership":
        if combined and len([line for line in lines if _contains_path_hint(line)]) >= 2:
            score, status = weight, "pass"
            rationale = "The brief points to concrete project files and a credible ownership boundary."
        elif combined:
            rationale = "There is some evidence, but ownership is not grounded in enough concrete files or tests."
        else:
            rationale = "The brief does not identify concrete files, docs, or tests that likely own the issue."
    elif criterion == "Architecture":
        if combined and (len(lines) >= 2 or len(combined) >= 120):
            score, status = weight, "pass"
            rationale = "The architecture notes explain the relevant boundary or runtime handoff clearly enough."
        elif combined:
            rationale = "Architecture is mentioned, but the module boundary still needs clarification."
        else:
            rationale = "The brief does not explain the relevant architectural boundary."
    elif criterion == "Clean Code":
        keywords = ("maintain", "coupling", "duplicate", "naming", "safe", "clarity", "complex")
        if combined and (len(lines) >= 2 or any(token in combined.lower() for token in keywords)):
            score, status = weight, "pass"
            rationale = "The investigation considers maintainability and change safety, not only the immediate fix."
        elif combined:
            rationale = "Clean code concerns are hinted at, but the maintainability tradeoffs remain thin."
        else:
            rationale = "The brief does not discuss maintainability or change-safety concerns."
    elif criterion == "Optimization":
        keywords = ("optimiz", "hot path", "loop", "memory", "redund", "perf", "cache", "io", "allocation")
        if combined and (len(lines) >= 1 or any(token in combined.lower() for token in keywords)):
            score, status = weight, "pass"
            rationale = "The investigation says how performance should be treated instead of guessing blindly."
        elif combined:
            rationale = "Optimization is mentioned, but the likely hotspot or non-goal is still vague."
        else:
            rationale = "The brief does not explain whether performance matters here."
    else:
        keywords = ("test", "assert", "validate", "measure", "repro", "log", "observe", "regression")
        if combined and len(lines) >= 2 and any(token in combined.lower() for token in keywords):
            score, status = weight, "pass"
            rationale = "The verification plan is concrete enough to confirm or reject the hypothesis quickly."
        elif combined:
            rationale = "Verification exists, but it still needs sharper regression or measurement steps."
        else:
            rationale = "The brief does not explain how the hypothesis will be validated."

    return {
        "criterion": criterion,
        "score": score,
        "max_score": weight,
        "status": status,
        "rationale": rationale,
        "action_items": [] if status == "pass" else [CRITERION_ACTIONS[criterion]],
    }


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


def _fallback_review(task_prompt: str, investigation_doc: str) -> dict[str, Any]:
    sections = _parse_sections(investigation_doc)
    criterion_scores = [_assess_criterion(name, weight, sections, task_prompt) for name, weight, *_ in REVIEW_CRITERIA]
    blocking_issues = _dedupe(
        [
            f"{item['criterion']}: {item['action_items'][0]}"
            for item in criterion_scores
            if item["criterion"] in BLOCKING_CRITERIA and item["status"] != "pass"
        ]
    )
    improvement_actions = _dedupe(
        [
            action
            for item in criterion_scores
            if item["status"] != "pass"
            for action in item["action_items"]
        ]
    )
    score = sum(item["score"] for item in criterion_scores)
    approved = score >= APPROVAL_SCORE and not blocking_issues
    review_doc = _compose_review_doc(
        score,
        approved,
        criterion_scores,
        blocking_issues,
        improvement_actions,
        (
            "The investigation is tight enough to move forward."
            if approved
            else "The investigation still needs sharper ownership, architecture, or validation detail."
        ),
    )
    return {
        "review_doc": review_doc,
        "score": score,
        "approved": approved,
        "criterion_scores": criterion_scores,
        "blocking_issues": blocking_issues,
        "improvement_actions": improvement_actions,
    }


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


def _parse_review_doc(review_doc: str, fallback: dict[str, Any]) -> dict[str, Any]:
    if re.search(r"Overall Score:\s*(\d{1,3})\s*/\s*100", review_doc, flags=re.IGNORECASE) is None:
        return fallback

    fallback_lookup = {item["criterion"]: item for item in fallback["criterion_scores"]}
    parsed_scores: list[CriterionAssessment] = []

    for line in _extract_heading_block(review_doc, "Criterion Scores"):
        match = re.match(
            r"-\s*(?P<criterion>[^:]+):\s*(?P<score>\d{1,3})\s*/\s*(?P<max>\d{1,3})\s*-\s*(?P<rationale>.+)",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        criterion = match.group("criterion").strip()
        if criterion not in fallback_lookup:
            continue
        base = fallback_lookup[criterion]
        max_score = base["max_score"]
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
                "rationale": match.group("rationale").strip() or base["rationale"],
                "action_items": [] if status == "pass" else list(base["action_items"]),
            }
        )

    if len(parsed_scores) != len(REVIEW_CRITERIA):
        parsed_scores = list(fallback["criterion_scores"])
    else:
        parsed_lookup = {item["criterion"]: item for item in parsed_scores}
        parsed_scores = [parsed_lookup[name] for name, *_ in REVIEW_CRITERIA]

    raw_blocking_issues = []
    for line in _extract_heading_block(review_doc, "Blocking Issues"):
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if not item.lower().startswith("none."):
                raw_blocking_issues.append(item)
    technical_blocking_issues = [item for item in raw_blocking_issues if not _is_process_only_feedback(item)]
    if raw_blocking_issues:
        blocking_issues = _dedupe(technical_blocking_issues)
    else:
        blocking_issues = _dedupe(list(fallback["blocking_issues"]))

    raw_improvement_actions = []
    for line in _extract_heading_block(review_doc, "Improvement Checklist"):
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        item = re.sub(r"^-\s*\[[ xX]\]\s*", "", stripped).strip()
        if not item.lower().startswith("no further investigation changes requested."):
            raw_improvement_actions.append(item)
    technical_improvement_actions = [
        item for item in raw_improvement_actions if not _is_process_only_feedback(item)
    ]
    if raw_improvement_actions:
        improvement_actions = _dedupe(technical_improvement_actions)
    else:
        improvement_actions = _dedupe(list(fallback["improvement_actions"]))

    normalized_scores = _normalize_criterion_scores(parsed_scores, blocking_issues=blocking_issues)
    score = sum(item["score"] for item in normalized_scores)
    approved = score >= APPROVAL_SCORE and not blocking_issues

    final_review_doc = _compose_review_doc(
        score,
        approved,
        normalized_scores,
        blocking_issues,
        improvement_actions,
        "\n".join(_extract_heading_block(review_doc, "Senior Engineer Notes")).strip(),
    )
    return {
        "review_doc": final_review_doc,
        "score": score,
        "approved": approved,
        "criterion_scores": normalized_scores,
        "blocking_issues": blocking_issues,
        "improvement_actions": improvement_actions,
    }


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name

    def review(state: ReviewerState) -> dict[str, Any]:
        review_round = int(state.get("review_round", 0)) + 1
        artifact_path = _artifact_dir(context, metadata, state)
        fallback = _fallback_review(state["task_prompt"], state["investigation_doc"])
        review_result = fallback

        if context.llm.is_enabled():
            try:
                generated_review = context.llm.generate_text(
                    instructions=(
                        "You are a strict senior engineer reviewing an investigation brief. Score it hard against focus, evidence and ownership, "
                        "architecture, clean code thinking, optimization awareness, and verification quality. Return markdown using this exact shape: "
                        "# Investigation Review, Decision: APPROVE or REVISE, Overall Score: NN/100, ## Criterion Scores, ## Blocking Issues, "
                        "## Improvement Checklist, ## Senior Engineer Notes. Use one bullet per criterion in the form `- Criterion: score/max - rationale`. "
                        "If there are no blocking issues, write exactly `- None.` under Blocking Issues. "
                        "If there are no further investigation changes requested, write exactly `- [x] No further investigation changes requested.` "
                        f"Minimum final-approval depth is {MIN_REVIEW_ROUNDS} review rounds. If the current round is below that floor, require one more "
                        "pass that independently re-validates the causal chain with fresh evidence, clearer ordering proof, or a read-only reproduction. "
                        "Do not approve early just because the first brief sounds plausible. "
                        "Only gate on technical investigation quality. Do not require organizational ownership assignment, DRI naming, commit/PR provenance, "
                        "or other process artifacts unless they are explicitly present in the provided evidence. Keep Overall Score numerically consistent "
                        "with the criterion bullets. Do not use JSON."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Review round: {review_round}/{MAX_REVIEW_ROUNDS}\n\n"
                        f"Minimum rounds required before final approval can stick: {MIN_REVIEW_ROUNDS}\n\n"
                        f"Investigation document:\n{state['investigation_doc']}\n\n"
                        "Act like a demanding senior engineer who cares about clean code, focus, optimization, architecture, and validation quality."
                    ),
                )
                review_result = _parse_review_doc(generated_review, fallback)
            except LLMError:
                review_result = fallback

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
