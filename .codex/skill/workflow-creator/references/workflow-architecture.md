# Workflow Architecture Reference

## When To Load This Reference
Load this file when shaping a new workflow, refactoring a workflow into subgraphs, or checking whether the repo default loop-and-score pattern should apply.

## Primary Repo Examples
- `Workflows/template-investigation-workflow/Workflow.md`
- `Workflows/template-investigation-workflow/entry.py`
- `Workflows/template-investigation-reviewer-workflow/Workflow.md`
- `Workflows/template-investigation-reviewer-workflow/entry.py`
- `core/quality_loop.py`
- `tests/test_template_investigation_workflow.py`

Treat these two template workflows as the repository's canonical examples for workflow quality. New non-trivial parent/reviewer loops should inherit their discipline around reviewer isolation, blocker normalization, minimum review depth, and artifact-by-artifact verification.

## Decomposition Rules
### Keep One Workflow
- Keep a single workflow when the work is short, deterministic, and unlikely to be reused.
- Keep a single workflow when local helper nodes are clearer than creating another workflow folder.
- Keep a single workflow when no strict review loop or independent capability boundary is needed.

### Split Into Parent Plus Reviewer Subgraph
- Split into a parent workflow plus an internal reviewer workflow when the main artifact needs hard review before handoff.
- Split when the reviewer logic should evolve separately from the main worker logic.
- Split when the parent should own looping, artifacts, and final status while review stays isolated.
- Split when approval must depend on score, blockers, or minimum review depth.

### Split Into Multiple Reusable Workflows
- Split into multiple workflows when a child capability is likely to serve more than one parent workflow.
- Split when the child has its own state contract, tests, or LLM profile.
- Split when the child can stay `exposed: false` but still deserves separate ownership.
- Split only if the new boundary improves clarity more than it increases coordination cost.

### Do Not Split
- Do not create a separate workflow for a tiny helper with no independent lifecycle.
- Do not split when the child would only wrap one local function.
- Do not split when the state handoff becomes noisier than the logic itself.

## Reuse Checklist
1. Search `Workflows/*/Workflow.md` for matching names and capabilities before adding a new folder.
2. Inspect both exposed workflows and internal workflows because reviewer or analyzer subgraphs are often reusable patterns.
3. Compare the expected input and output state shape before reusing a workflow.
4. Reuse via `context.get_workflow_graph("<workflow-name>")` when the child graph should be embedded as a subgraph.
5. Create a new workflow only when reuse would force awkward state adaptation or mismatched review criteria.

## Default Loop And Score Guardrails
- Use a strict reviewer for non-trivial investigation, planning, analysis, or implementation-prep flows.
- Default target score to `>= 90`.
- Default minimum review rounds to `2`.
- Default maximum review rounds to `3` unless the task clearly needs a different ceiling.
- Keep the parent workflow in charge of loop status and artifact reporting.
- Keep reviewer output in markdown, not JSON.
- Filter process-only review asks if the workflow is supposed to judge technical quality rather than process hygiene.
- Make final approval impossible before the minimum round count is satisfied.
- When in doubt, copy the structure and rigor of `template-investigation-workflow` plus `template-investigation-reviewer-workflow` before inventing a lighter custom loop.

Use the same core criteria unless the task genuinely needs different ones:
- Focus
- Evidence & Ownership
- Architecture
- Clean Code
- Optimization
- Verification

## Natural-Language Prompt Templates
### Worker Or Investigation Template
```text
You are <workflow-name>. <Perform the main task>. Write a markdown brief using this exact section order:
<section list>.

Stay concrete, evidence-driven, and strict about scope. <tool guidance>.
If previous review feedback exists, address it explicitly. Do not use JSON.

Task prompt:
<task_prompt>

Round goal:
<round_goal>

Minimum review rounds before final approval can stick: <min_rounds>

Suggested starting docs:
<bullets>

Suggested starting source files:
<bullets>

Suggested starting tests:
<bullets>

Current repo snapshot:
<snapshot>

Previous artifact:
<artifact or "None. This is the first round.">

Previous reviewer feedback:
<feedback or "None. This is the first round.">

Previous reviewer checklist:
<checklist or "None.">

Return only the next artifact document. The next document must add real evidence, not just rephrase the prior round.
```

### Reviewer Template
```text
You are a strict senior engineer reviewing a <artifact type>. Score it hard against focus, evidence and ownership, architecture, clean code thinking, optimization awareness, and verification quality.

Return markdown using this exact shape:
# <Review Title>
Decision: APPROVE or REVISE
Overall Score: NN/100
## Criterion Scores
## Blocking Issues
## Improvement Checklist
## Senior Engineer Notes

Use one bullet per criterion in the form `- Criterion: score/max - rationale`.
If there are no blocking issues, write exactly `- None.`.
If there are no further changes requested, write exactly `- [x] No further ...`.
Minimum final-approval depth is <min_rounds> review rounds. If the current round is below that floor, require one more independent verification pass.
Do not use JSON.
```

## Subgraph Wiring Pattern
Use a parent workflow to keep loop ownership and embed the reviewer as a subgraph:

```python
reviewer_graph = context.get_workflow_graph("my-reviewer-workflow")

graph.add_node("work", work_node)
graph.add_node("request_review", request_review_node)
graph.add_node("my-reviewer-workflow", reviewer_graph)
graph.add_node("capture_review_result", capture_review_result_node)

graph.add_edge(START, "work")
graph.add_edge("work", "request_review")
graph.add_edge("request_review", "my-reviewer-workflow")
graph.add_edge("my-reviewer-workflow", "capture_review_result")
graph.add_conditional_edges("capture_review_result", route_fn, {"work": "work", END: END})
```

Keep the reviewer workflow `exposed: false` unless users should call it directly.

## Quality Loop Pattern
Use `QualityLoopSpec` and `evaluate_quality_loop(...)` for explicit pass, continue, and stop behavior:

```python
LOOP_SPEC = QualityLoopSpec(
    loop_id="my-workflow-review",
    threshold=90,
    max_rounds=3,
    min_rounds=2,
    require_blocker_free=True,
    require_missing_section_free=False,
    require_explicit_approval=True,
    min_score_delta=1,
    stagnation_limit=2,
)
```

Keep loop state explicit in the parent workflow:
- current work artifact
- current review artifact
- score
- blocker list
- improvement checklist
- approved flag
- round counters
- loop status and reason

## Testing Expectations
- Add a regression test that the parent workflow registers the reviewer subgraph.
- Add or keep a fallback test that completes without LLM access.
- Add a loop test that proves the workflow iterates until the score reaches the threshold.
- Add a test that process-only review feedback does not block technically strong work when the reviewer is supposed to judge technical quality.
- Add a test that at least two review rounds are enforced when that is the architectural contract.
- Verify that artifacts for each round are written when the workflow persists reports.
