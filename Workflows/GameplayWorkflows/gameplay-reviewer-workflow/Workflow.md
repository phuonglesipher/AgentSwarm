---
name: gameplay-reviewer-workflow
entry: entry.py
version: 1.0.0
exposed: false
llm_profile: reviewer
capabilities:
  - gameplay implementation plan review
  - gameplay feature review
  - gameplay maintenance review
  - gameplay bugfix plan scoring
  - gameplay review loop
---
Internal reviewer subgraph for `gameplay-engineer-workflow`. This workflow
scores gameplay implementation plans against scope clarity, task typing,
existing references, implementation sequencing, regression coverage, risks, and
acceptance criteria. Approval is intentionally two-pass: the first review can
surface issues, but final approval cannot stick before a second verification
round re-checks player-visible outcome, grounded ownership, speculation control,
and adjacent-path regression coverage. It keeps the review strict enough for
gameplay work while staying focused on technical quality instead of process
overhead.
