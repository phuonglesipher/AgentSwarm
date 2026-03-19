---
name: root-project-investigation-reviewer-workflow
entry: entry.py
version: 1.0.0
exposed: false
llm_profile: reviewer
capabilities:
  - senior investigation review
  - investigation scoring
  - root cause review
  - clean code review
  - architecture review
  - optimization review
---
Internal reviewer subgraph for `root-project-investigation-workflow`. This
workflow acts like a demanding senior engineer: it scores investigation briefs,
filters process-only asks, applies the verification-depth gate, and decides
whether the parent investigation loop should continue.
