---
name: gameplay-engineer-workflow
entry: entry.py
version: 1.0.0
exposed: true
capabilities:
  - gameplay implementation
  - gameplay bug fixing
  - 3c feature development
  - combat feature development
  - gameplay planning
tools:
  - find-gameplay-docs
  - load-markdown-context
---
This workflow is responsible for gameplay engineering work. It handles gameplay
feature development and bug fixing across 3C and combat content. The workflow
classifies a task as bug-fix or new feature, checks the existing gameplay and
design docs, builds a design doc when one does not exist, writes a planning doc,
requests structured review feedback from the gameplay-reviewer-workflow, revises
the plan using per-section scores, blockers, and checklist items until the plan
is approved by the reviewer rubric, then implements code, self-tests, verifies
compilation, and prepares commit and pull request artifacts.
