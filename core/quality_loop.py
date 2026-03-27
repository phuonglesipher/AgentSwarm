from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _dedupe_preserve_order(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_item in items:
        item = str(raw_item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


@dataclass(frozen=True)
class QualityLoopSpec:
    loop_id: str
    threshold: int
    max_rounds: int
    min_rounds: int = 1
    require_blocker_free: bool = True
    require_missing_section_free: bool = True
    require_explicit_approval: bool = True
    min_score_delta: int = 1
    stagnation_limit: int = 0


@dataclass(frozen=True)
class QualityLoopProgress:
    loop_id: str
    round_index: int
    score: int
    threshold: int
    max_rounds: int
    approved: bool
    missing_sections: tuple[str, ...]
    blocking_issues: tuple[str, ...]
    improvement_actions: tuple[str, ...]
    score_delta: int
    stagnated_rounds: int
    status: str
    should_continue: bool
    completed: bool
    reason: str


def evaluate_quality_loop(
    spec: QualityLoopSpec,
    *,
    round_index: int,
    score: int,
    approved: bool,
    missing_sections: Iterable[str] = (),
    blocking_issues: Iterable[str] = (),
    improvement_actions: Iterable[str] = (),
    previous_score: int | None = None,
    prior_stagnated_rounds: int = 0,
) -> QualityLoopProgress:
    if round_index <= 0:
        raise ValueError("round_index must be positive")
    if spec.max_rounds <= 0:
        raise ValueError("max_rounds must be positive")
    if spec.min_rounds <= 0:
        raise ValueError("min_rounds must be positive")
    if spec.min_rounds > spec.max_rounds:
        raise ValueError("min_rounds must be less than or equal to max_rounds")
    if spec.threshold < 0:
        raise ValueError("threshold must be non-negative")
    if spec.stagnation_limit < 0:
        raise ValueError("stagnation_limit must be non-negative")

    normalized_missing_sections = _dedupe_preserve_order(missing_sections)
    normalized_blocking_issues = _dedupe_preserve_order(blocking_issues)
    normalized_improvement_actions = _dedupe_preserve_order(improvement_actions)

    clamped_score = max(0, min(int(score), 100))
    if previous_score is None:
        score_delta = clamped_score
        stagnated_rounds = 0
    else:
        score_delta = clamped_score - max(0, min(int(previous_score), 100))
        stagnated_rounds = max(0, int(prior_stagnated_rounds))
        if score_delta < spec.min_score_delta and clamped_score < spec.threshold:
            stagnated_rounds += 1
        else:
            stagnated_rounds = 0

    meets_score = clamped_score >= spec.threshold
    meets_approval = bool(approved) if spec.require_explicit_approval else True
    meets_blockers = (not normalized_blocking_issues) if spec.require_blocker_free else True
    meets_missing_sections = (not normalized_missing_sections) if spec.require_missing_section_free else True

    if round_index < spec.min_rounds:
        status = "retry"
        should_continue = True
        completed = False
        reason = (
            f"Loop `{spec.loop_id}` needs at least {spec.min_rounds} round(s); "
            f"round {round_index} is still a verification pass."
        )
    elif meets_score and meets_approval and meets_blockers and meets_missing_sections:
        status = "passed"
        should_continue = False
        completed = True
        reason = f"Loop `{spec.loop_id}` reached the score threshold and cleared all blocking conditions."
    elif spec.stagnation_limit and stagnated_rounds >= spec.stagnation_limit:
        status = "stagnated"
        should_continue = False
        completed = True
        reason = (
            f"Loop `{spec.loop_id}` stopped after {stagnated_rounds} low-progress round(s) without enough score delta."
        )
    elif round_index >= spec.max_rounds:
        status = "max-rounds"
        should_continue = False
        completed = True
        reason = f"Loop `{spec.loop_id}` reached the maximum of {spec.max_rounds} round(s)."
    else:
        status = "retry"
        should_continue = True
        completed = False
        pending_reasons: list[str] = []
        if not meets_score:
            pending_reasons.append(f"score {clamped_score}/{spec.threshold}")
        if not meets_approval:
            pending_reasons.append("explicit approval missing")
        if not meets_blockers:
            pending_reasons.append(f"{len(normalized_blocking_issues)} blocking issue(s)")
        if not meets_missing_sections:
            pending_reasons.append(f"{len(normalized_missing_sections)} missing section(s)")
        reason = (
            f"Loop `{spec.loop_id}` needs another round because " + ", ".join(pending_reasons) + "."
            if pending_reasons
            else f"Loop `{spec.loop_id}` needs another round."
        )

    return QualityLoopProgress(
        loop_id=spec.loop_id,
        round_index=round_index,
        score=clamped_score,
        threshold=spec.threshold,
        max_rounds=spec.max_rounds,
        approved=status == "passed",
        missing_sections=normalized_missing_sections,
        blocking_issues=normalized_blocking_issues,
        improvement_actions=normalized_improvement_actions,
        score_delta=score_delta,
        stagnated_rounds=stagnated_rounds,
        status=status,
        should_continue=should_continue,
        completed=completed,
        reason=reason,
    )
