from __future__ import annotations

import re
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_logging import trace_graph_node
from core.models import WorkflowContext, WorkflowMetadata
from core.review import ReviewCriterion, ReviewEngine, ReviewProfile


# ------------------------------------------------------------------ #
#  Review profile
# ------------------------------------------------------------------ #

PROFILE = ReviewProfile(
    system_id="optimize-investigation-review",
    display_name="Optimization Investigation Review",
    criteria=(
        ReviewCriterion("Problem Scoping", 15, ("Task Framing",)),
        ReviewCriterion("Evidence Rigor", 25, ("Source Code Evidence", "Hot Path Identification")),
        ReviewCriterion("System-Specific Analysis", 20),
        ReviewCriterion("Optimization Quality", 20, ("Optimization Recommendations",)),
        ReviewCriterion("Risk & Regression", 10, ("Regression Risk",)),
        ReviewCriterion("Verification Completeness", 10, ("Verification Plan",)),
    ),
    approval_threshold=85,
    min_rounds=2,
    max_rounds=4,
    require_missing_section_free=False,
    mandatory_action=(
        "Run one more investigation pass that independently re-validates the performance hypothesis with fresh source code evidence, "
        "config file analysis, call frequency analysis, or structural code path reasoning before final handoff."
    ),
    prompt_persona=(
        "You are a strict senior engineer reviewing an optimization investigation brief for {optimization_domain}. "
        "Focus on evidence of actual performance hotspots — source code analysis, config file evidence, call frequency analysis, "
        "code path tracing, or structural cost breakdowns. Speculation alone is not sufficient for approval. "
        "Check whether the investigator provided concrete evidence from source code, config files, or static analysis."
    ),
    prompt_domain_instructions="",
    prompt_round_guidance=None,
    extra_process_keywords=(
        "nanite showstats",
    ),
    extra_process_patterns=(
        re.compile(r"test\s+on\s+(actual|real)\s+hardware", re.IGNORECASE),
    ),
    supports_markdown_fallback=True,
    doc_field_name="investigation_doc",
    review_doc_title="Optimization Investigation Review",
    notes_heading="Senior Engineer Notes",
    no_action_text="No further investigation changes requested.",
    domain_noun="optimization investigation",
    schema_name="optimization_investigation_review",
    dynamic_criteria_field="review_criteria",
    dynamic_domain_field="optimization_domain",
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
    review_score_confidence: NotRequired[float | None]
    review_score_confidence_label: NotRequired[str]
    review_score_confidence_reason: NotRequired[str]
    summary: str
    review_criteria: NotRequired[list[tuple]]
    optimization_domain: NotRequired[str]


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
