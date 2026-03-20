---
name: template-investigation-reviewer-workflow
entry: entry.py
version: 1.0.0
exposed: false
llm_profile: reviewer
capabilities:
  - reviewer workflow template
  - senior investigation review
  - investigation scoring
  - root cause review
  - clean code review
  - architecture review
  - optimization review
---
Internal reviewer subgraph for `template-investigation-workflow`. This
workflow acts like a demanding senior engineer: it scores investigation briefs,
filters process-only asks, applies the verification-depth gate, and decides
whether the parent investigation loop should continue. Treat this reviewer as
the quality template for other strict reviewer workflows that need blocker
normalization, explicit approval gating, and a mandatory second-pass
verification round.
