---
name: optimize-investigation-reviewer-workflow
entry: entry.py
version: 1.0.0
exposed: false
llm_profile: reviewer
capabilities:
  - optimization investigation review
  - performance investigation scoring
  - optimization root cause review
  - profiling evidence review
---
Internal reviewer subgraph for optimization investigation workflows. Scores
optimization investigation briefs against domain-specific criteria passed via
state, filters process-only asks, applies the verification-depth gate, and
decides whether the parent investigation loop should continue. Supports all
optimization domains (game thread, streaming, rendering) through dynamic
criteria read from `state["review_criteria"]`.
