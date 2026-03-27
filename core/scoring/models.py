from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreAssessment:
    label: str
    score: int
    max_score: int
    status: str
    rationale: str = ""
    action_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScorePolicy:
    system_id: str
    threshold: int
    require_blocker_free: bool = True
    require_missing_section_free: bool = False
    require_explicit_approval: bool = True
    min_confidence_samples: int = 3
    confidence_threshold: float = 1.0
    strong_confidence_threshold: float = 2.0
    require_confidence_when_available: bool = True
    confidence_override_score: int = 95
    history_filename: str = "score_history.jsonl"


@dataclass(frozen=True)
class ScoreSnapshot:
    system_id: str
    round_index: int
    score: int
    threshold: int
    max_score: int
    explicit_approval: bool
    approved: bool
    confidence: float | None
    confidence_label: str
    confidence_reason: str
    baseline_score: int | None
    score_delta_from_baseline: int | None


@dataclass(frozen=True)
class ScoreDecision:
    system_id: str
    round_index: int
    score: int
    raw_score: int
    max_score: int
    threshold: int
    assessments: tuple[ScoreAssessment, ...]
    explicit_approval: bool
    approved: bool
    blocking_issues: tuple[str, ...]
    missing_sections: tuple[str, ...]
    improvement_actions: tuple[str, ...]
    confidence: float | None
    confidence_label: str
    confidence_reason: str
    confidence_samples: int
    baseline_score: int | None
    score_delta_from_baseline: int | None
    approval_reasons: tuple[str, ...]
    history: tuple[ScoreSnapshot, ...]
