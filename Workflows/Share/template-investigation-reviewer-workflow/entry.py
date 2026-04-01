from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_logging import trace_graph_node
from core.models import WorkflowContext, WorkflowMetadata
from core.review import HardBlocker, ReviewCriterion, ReviewEngine, ReviewProfile


# ------------------------------------------------------------------ #
#  Domain-specific mandatory action builder
# ------------------------------------------------------------------ #

def _build_mandatory_action(improvement_actions: list[str], blocking_issues: list[str]) -> str:
    base = "Run one more investigation pass that independently re-validates"
    keywords = " ".join(improvement_actions + blocking_issues).lower()
    if "caller" in keywords or "consumer" in keywords:
        return f"{base} the API consumers with grep/glob evidence (prove caller presence/absence)."
    if "test" in keywords or "reproduction" in keywords or "regression" in keywords:
        return f"{base} the root cause with a focused read-only reproduction or source code trace."
    if "proof" in keywords or "call-order" in keywords or "sequencing" in keywords:
        return f"{base} the call-order proof with source code traces or static call graph analysis."
    return f"{base} the causal chain with at least one new piece of concrete source code evidence."


# ------------------------------------------------------------------ #
#  Review profile
# ------------------------------------------------------------------ #

PROFILE = ReviewProfile(
    system_id="template-investigation-review",
    display_name="Investigation Review",
    criteria=(
        ReviewCriterion("Focus", 20, ("Task Framing", "Root Cause Hypothesis")),
        ReviewCriterion("Evidence & Ownership", 25, ("Project Root Findings", "Candidate Ownership", "Consumer & Caller Analysis")),
        ReviewCriterion("Architecture", 15, ("Architecture Notes",)),
        ReviewCriterion("Clean Code", 10, ("Clean Code Notes",)),
        ReviewCriterion("Optimization", 10, ("Optimization Notes",)),
        ReviewCriterion("Verification", 20, ("Verification Plan",)),
    ),
    approval_threshold=90,
    min_rounds=2,
    max_rounds=5,
    require_missing_section_free=True,
    mandatory_action=_build_mandatory_action,
    prompt_persona=(
        "You are a strict senior engineer reviewing an investigation brief. Score it hard against focus, evidence and ownership, "
        "architecture, clean code thinking, optimization awareness, and verification quality."
    ),
    prompt_domain_instructions=(
        "Under Evidence & Ownership, explicitly check whether the investigator searched for and documented "
        "external consumers/callers of the APIs under investigation. If the brief lacks a 'Consumer & Caller "
        "Analysis' section or does not prove presence/absence of callers outside the owning module, deduct "
        "from Evidence & Ownership and flag it as an improvement action.\n\n"
        "Scoring calibration:\n"
        "- Evidence & Ownership 25/25: Hypothesis grounded in specific file references AND grep/glob results "
        "proving caller presence/absence AND clear ownership attribution.\n"
        "- Evidence & Ownership 15/25: Hypothesis stated but lacks grep evidence or caller analysis is superficial.\n"
        "- Evidence & Ownership <10/25: No hypothesis, or hypothesis is speculation without file references.\n"
        "- Verification 20/20: Concrete verification steps using source code analysis, config validation, or reproducible static checks.\n"
        "- Verification 12/20: Plan exists but vague ('write a test' without specifics).\n"
        "- Verification <8/20: No verification plan or plan is generic boilerplate.\n"
        "- Focus 20/20: Every section directly serves the task prompt. No tangents.\n"
        "- Focus <12/20: Significant scope creep or sections that don't serve the investigation."
    ),
    prompt_round_guidance=lambda r, m: (
        "This is round 1. Score on discovery quality and pathway clarity. "
        "Don't penalize for incomplete verification — that's what round 2 is for."
        if r == 1
        else (
            f"This is round {r} (>= min rounds). Score on convergence. "
            "If the hypothesis hasn't narrowed or new evidence wasn't added, score Focus and Evidence harshly."
            if r >= m
            else ""
        )
    ),
    doc_field_name="investigation_doc",
    review_doc_title="Investigation Review",
    notes_heading="Senior Engineer Notes",
    no_action_text="No further investigation changes requested.",
    domain_noun="investigation",
    schema_name="investigation_review",
)


# ------------------------------------------------------------------ #
#  State type (preserved for backward compatibility)
# ------------------------------------------------------------------ #

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
    review_missing_sections: list[str]
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
        "review",
        trace_graph_node(graph_name=metadata.name, node_name="review", node_fn=engine.review),
    )
    graph.add_edge(START, "review")
    graph.add_edge("review", END)
    return graph.compile()
