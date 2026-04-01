from __future__ import annotations

import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_logging import trace_graph_node
from core.models import WorkflowContext, WorkflowMetadata
from core.review import HardBlocker, ReviewCriterion, ReviewEngine, ReviewProfile


# ------------------------------------------------------------------ #
#  Review profile
# ------------------------------------------------------------------ #

PROFILE = ReviewProfile(
    system_id="gameplay-plan-review",
    display_name="Gameplay Plan Review",
    criteria=(
        ReviewCriterion("Overview", 10),
        ReviewCriterion("Task Type", 10),
        ReviewCriterion("Existing Docs", 10),
        ReviewCriterion("Implementation Steps", 25),
        ReviewCriterion("Unit Tests", 20),
        ReviewCriterion("Risks", 10),
        ReviewCriterion("Acceptance Criteria", 15),
    ),
    approval_threshold=90,
    min_rounds=2,
    max_rounds=3,
    require_missing_section_free=True,
    hard_blockers=(
        HardBlocker("Player Outcome:", "the plan must name the player-visible result and scope boundary."),
        HardBlocker("Current Behavior Evidence:", "the plan must cite grounded docs, runtime paths, and the owner."),
        HardBlocker("Speculation Control:", "implementation steps must stay anchored on current ownership."),
        HardBlocker("Edge and Regression Coverage:", "tests and acceptance criteria must protect adjacent gameplay paths."),
    ),
    mandatory_action=(
        "Run one more plan revision pass that independently tightens the player-visible outcome, "
        "grounded owner evidence, and regression coverage before implementation."
    ),
    prompt_persona=(
        "You are gameplay-reviewer-workflow. Review a gameplay implementation plan like a strict senior gameplay engineer. "
        "Focus on technical clarity, owner paths, regression coverage, and player-visible acceptance."
    ),
    prompt_domain_instructions=(
        "Hard blockers: player outcome, grounded current-behavior evidence, speculation control around the owning runtime path, "
        "and adjacent-path regression coverage. "
        "Do not block on review-round bookkeeping, artifact naming, sign-off records, or other process-only approval trace details."
    ),
    prompt_round_guidance=None,
    extra_process_keywords=(),
    extra_process_patterns=(
        re.compile(r"\breview round\b", re.IGNORECASE),
        re.compile(r"\bround metadata\b", re.IGNORECASE),
        re.compile(r"\bcurrent review context\b", re.IGNORECASE),
        re.compile(r"\bprocess[- ]gate\b", re.IGNORECASE),
        re.compile(r"\bindependent verif(?:ication|ier)\b", re.IGNORECASE),
        re.compile(r"\bsign[- ]off\b", re.IGNORECASE),
        re.compile(r"\bevidence artifact\b", re.IGNORECASE),
        re.compile(r"\bverification artifact\b", re.IGNORECASE),
        re.compile(r"\bartifact naming\b", re.IGNORECASE),
        re.compile(r"\btraceability\b", re.IGNORECASE),
        re.compile(r"\bapproval[- ]trace\b", re.IGNORECASE),
        re.compile(r"\bmetadata\b", re.IGNORECASE),
        re.compile(r"\bcurrent-round\b", re.IGNORECASE),
        re.compile(r"\bround-\d+\b", re.IGNORECASE),
        re.compile(r"\blog filenames?\b", re.IGNORECASE),
        re.compile(r"\btargeted test command", re.IGNORECASE),
    ),
    supports_markdown_fallback=False,
    doc_field_name="plan_doc",
    review_doc_title="Gameplay Plan Review",
    notes_heading="Reviewer Notes",
    no_action_text="None.",
    domain_noun="gameplay plan",
    schema_name="gameplay_plan_review",
    # Backward-compat: gameplay-engineer-workflow reads both review_* and unprefixed fields,
    # plus review_loop_* variants.
    state_field_aliases=(
        ("review_score", "score"),
        ("review_feedback", "feedback"),
        ("review_missing_sections", "missing_sections"),
        ("review_criterion_scores", "section_reviews"),
        ("review_blocking_issues", "blocking_issues"),
        ("review_improvement_actions", "improvement_actions"),
        ("review_approved", "approved"),
        ("loop_status", "review_loop_status"),
        ("loop_reason", "review_loop_reason"),
        ("loop_should_continue", "review_loop_should_continue"),
        ("loop_completed", "review_loop_completed"),
        ("loop_stagnated_rounds", "review_loop_stagnated_rounds"),
    ),
)


# ------------------------------------------------------------------ #
#  State type (preserved for backward compatibility)
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Graph construction
# ------------------------------------------------------------------ #

def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    engine = ReviewEngine(PROFILE, context, metadata)
    graph = StateGraph(ReviewerState)
    graph.add_node(
        "review_plan",
        trace_graph_node(graph_name=metadata.name, node_name="review_plan", node_fn=engine.review),
    )
    graph.add_edge(START, "review_plan")
    graph.add_edge("review_plan", END)
    return graph.compile()
