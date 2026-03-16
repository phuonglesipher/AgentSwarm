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
  - find-gameplay-blueprints
  - find-gameplay-docs
  - load-blueprint-context
  - load-markdown-context
  - find-gameplay-code
  - load-source-context
---
This workflow is responsible for gameplay engineering work. It handles gameplay
feature development and bug fixing across 3C and combat content. The workflow
classifies a task as bug-fix or new feature, checks the existing gameplay and
design docs, inspects relevant host-project source and test files, and also
searches for relevant Blueprint assets plus readable companion exports.

Feature work follows an architecture-review branch: it builds a design doc,
writes an architecture and implementation plan, requests structured review
feedback from the gameplay-reviewer-workflow, and revises the plan using
per-section scores, blockers, and checklist items until the plan is approved.

Bug-fix work follows a context-first branch: it gathers a bug investigation
brief, classifies the likely implementation medium as C++/code, Blueprint, or
mixed, skips the architecture review loop, and moves directly into
implementation.

For Blueprint-driven work, the workflow reads companion text when available and
generates concrete Blueprint fix instructions. When the project only exposes
binary `.uasset` data, the workflow reports that limitation clearly and emits a
manual-editor handoff artifact instead of pretending the asset was edited
directly. Code-side outputs are still written into host-project source/test
roots when available, then self-tested, verified for compilation, and prepared
as commit and pull request artifacts.
