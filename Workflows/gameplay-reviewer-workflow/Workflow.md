---
name: gameplay-reviewer-workflow
entry: entry.py
version: 1.0.0
exposed: false
llm_profile: reviewer
capabilities:
  - gameplay plan review
  - gameplay implementation feedback
  - plan scoring
  - unit test coverage review
---
This workflow reviews gameplay engineering plans. It scores the plan, verifies
that unit tests are included, highlights missing sections, and sends actionable
feedback back to the gameplay-engineer-workflow until the planning score is 100.
