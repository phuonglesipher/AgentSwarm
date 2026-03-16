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
feedback back to the gameplay-engineer-workflow.

The reviewer uses a fixed rubric:
- `Overview` 10 points: player-facing goal and scope are clear.
- `Task Type` 10 points: bugfix vs feature is named and justified.
- `Existing Docs` 10 points: referenced docs or baseline assumptions are explicit.
- `Implementation Steps` 25 points: steps are ordered, concrete, and name gameplay touch points.
- `Unit Tests` 20 points: automated checks and expected assertions are specified.
- `Risks` 10 points: likely regressions plus mitigation or fallback are documented.
- `Acceptance Criteria` 15 points: player-visible outcomes and regression checks are explicit.

Approval requires `>= 90/100` and zero blocking issues. Each review result
returns per-section scores, blocking issues, and an improvement checklist so the
gameplay-engineer-workflow knows exactly what to revise.
