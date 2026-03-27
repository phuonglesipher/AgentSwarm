from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .history import load_score_history, record_score_snapshot
from .models import ScoreAssessment, ScoreDecision, ScorePolicy, ScoreSnapshot
from .stats import mad_confidence


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


def _normalize_assessments(assessments: Iterable[ScoreAssessment]) -> tuple[ScoreAssessment, ...]:
    normalized: list[ScoreAssessment] = []
    for item in assessments:
        max_score = max(0, int(item.max_score))
        score = max(0, min(int(item.score), max_score)) if max_score > 0 else 0
        normalized.append(
            ScoreAssessment(
                label=str(item.label).strip(),
                score=score,
                max_score=max_score,
                status=str(item.status).strip() or "missing",
                rationale=str(item.rationale).strip(),
                action_items=_dedupe_preserve_order(item.action_items),
            )
        )
    return tuple(normalized)


def _normalize_score(raw_score: int, max_score: int) -> int:
    if max_score <= 0:
        return 0
    return max(0, min(int(round((raw_score / max_score) * 100)), 100))


def _confidence_verdict(
    policy: ScorePolicy,
    *,
    score_values: list[int],
    baseline_score: int | None,
    current_score: int,
) -> tuple[float | None, str, str]:
    sample_count = len(score_values)
    if sample_count < policy.min_confidence_samples:
        return (
            None,
            "unmeasured",
            (
                f"MAD confidence needs at least {policy.min_confidence_samples} scored round(s); "
                f"only {sample_count} observation(s) are available."
            ),
        )

    confidence = mad_confidence(
        score_values,
        baseline=baseline_score,
        current=current_score,
        min_samples=policy.min_confidence_samples,
    )
    if confidence is None:
        return (
            None,
            "unmeasured",
            "MAD confidence is unavailable because the observed score noise floor collapsed to zero.",
        )
    if confidence == float('inf'):
        return (
            confidence,
            "stable",
            "All scored observations are identical, so the score is perfectly stable.",
        )
    if confidence >= policy.strong_confidence_threshold:
        return (
            confidence,
            "strong",
            f"MAD confidence is {confidence:.1f}x the noise floor, so the score improvement looks durable.",
        )
    if confidence >= policy.confidence_threshold:
        return (
            confidence,
            "marginal",
            f"MAD confidence is {confidence:.1f}x the noise floor, so the improvement is above noise but still marginal.",
        )
    return (
        confidence,
        "weak",
        (
            f"MAD confidence is {confidence:.1f}x the noise floor, which is still within the unstable band "
            f"below {policy.confidence_threshold:.1f}x."
        ),
    )


def evaluate_score_decision(
    policy: ScorePolicy,
    *,
    round_index: int,
    assessments: Iterable[ScoreAssessment],
    explicit_approval: bool,
    blocking_issues: Iterable[str] = (),
    missing_sections: Iterable[str] = (),
    improvement_actions: Iterable[str] = (),
    artifact_dir: Path | None = None,
) -> ScoreDecision:
    if round_index <= 0:
        raise ValueError("round_index must be positive")

    normalized_assessments = _normalize_assessments(assessments)
    raw_score = sum(item.score for item in normalized_assessments)
    max_score = sum(item.max_score for item in normalized_assessments)
    score = _normalize_score(raw_score, max_score)

    prior_history = ()
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        prior_history = tuple(item for item in load_score_history(artifact_dir, policy) if item.round_index != round_index)

    score_values = [item.score for item in prior_history]
    score_values.append(score)
    baseline_score = score_values[0] if score_values else None
    score_delta_from_baseline = None if baseline_score is None else score - baseline_score
    confidence, confidence_label, confidence_reason = _confidence_verdict(
        policy,
        score_values=score_values,
        baseline_score=baseline_score,
        current_score=score,
    )

    normalized_blocking_issues = list(_dedupe_preserve_order(blocking_issues))
    normalized_missing_sections = _dedupe_preserve_order(missing_sections)
    normalized_improvement_actions = list(_dedupe_preserve_order(improvement_actions))
    approval_reasons: list[str] = []

    if policy.require_confidence_when_available and confidence is not None and confidence < policy.confidence_threshold and score < policy.confidence_override_score:
        normalized_blocking_issues.append(
            (
                f"Scoring Confidence: MAD confidence is {confidence:.1f}x noise floor; "
                f"needs >= {policy.confidence_threshold:.1f}x before approval."
            )
        )
        normalized_improvement_actions.append(
            "Run another independent review pass to lift scoring confidence above the MAD noise floor."
        )
        approval_reasons.append(confidence_reason)

    deduped_blocking_issues = _dedupe_preserve_order(normalized_blocking_issues)
    deduped_improvement_actions = _dedupe_preserve_order(normalized_improvement_actions)
    meets_threshold = score >= policy.threshold
    meets_explicit_approval = bool(explicit_approval) if policy.require_explicit_approval else True
    meets_blockers = (not deduped_blocking_issues) if policy.require_blocker_free else True
    meets_missing_sections = (not normalized_missing_sections) if policy.require_missing_section_free else True

    if not meets_threshold:
        approval_reasons.append(f"Score {score}/100 is below the {policy.threshold}/100 threshold.")
    if not meets_explicit_approval:
        approval_reasons.append("Explicit reviewer approval is still missing.")
    if not meets_blockers:
        approval_reasons.append(f"{len(deduped_blocking_issues)} blocking issue(s) remain.")
    if not meets_missing_sections:
        approval_reasons.append(f"{len(normalized_missing_sections)} required section(s) are still missing.")

    approved = meets_threshold and meets_explicit_approval and meets_blockers and meets_missing_sections
    snapshot = ScoreSnapshot(
        system_id=policy.system_id,
        round_index=round_index,
        score=score,
        threshold=policy.threshold,
        max_score=max_score,
        explicit_approval=bool(explicit_approval),
        approved=approved,
        confidence=confidence,
        confidence_label=confidence_label,
        confidence_reason=confidence_reason,
        baseline_score=baseline_score,
        score_delta_from_baseline=score_delta_from_baseline,
    )
    history = tuple([*prior_history, snapshot])
    if artifact_dir is not None:
        history = record_score_snapshot(artifact_dir, policy, snapshot)

    return ScoreDecision(
        system_id=policy.system_id,
        round_index=round_index,
        score=score,
        raw_score=raw_score,
        max_score=max_score,
        threshold=policy.threshold,
        assessments=normalized_assessments,
        explicit_approval=bool(explicit_approval),
        approved=approved,
        blocking_issues=deduped_blocking_issues,
        missing_sections=normalized_missing_sections,
        improvement_actions=deduped_improvement_actions,
        confidence=confidence,
        confidence_label=confidence_label,
        confidence_reason=confidence_reason,
        confidence_samples=len(score_values),
        baseline_score=baseline_score,
        score_delta_from_baseline=score_delta_from_baseline,
        approval_reasons=_dedupe_preserve_order(approval_reasons),
        history=history,
    )
