---
name: workflow-creator
description: Design or refactor AgentSwarm workflows for this repository. Use when Codex needs to turn a new request into one or more workflow graphs under `Workflows/`, decide whether to keep the work in one workflow or split it into reusable subgraphs, reuse existing workflows before creating new ones, or enforce the repo's loop-and-score architecture with strict senior review, natural-language prompts, and non-JSON state handoffs.
---

# Workflow Creator

## Overview
Create workflows that fit this repo's architecture instead of treating every request as a brand-new graph. Decide whether to reuse an existing workflow, embed an internal reviewer subgraph, or add a new reusable child workflow, then wire a score-gated loop that keeps iterating until the work is technically strong enough.

## Start With Workflow Triage
1. Inspect `Workflows/*/Workflow.md` and the relevant `entry.py` files before creating anything new.
2. Reuse an existing workflow if its capability and state contract already fit with only light adaptation.
3. Split the request into multiple workflows only when one part is reusable, independently testable, or deserves its own lifecycle.
4. Keep tiny helpers as local functions or graph nodes; do not promote every step into a workflow.

## Apply The Repo Default
For non-trivial investigation, planning, design, or implementation-prep flows, default to:
1. A parent workflow that owns task progress, loop state, and artifacts.
2. A strict reviewer workflow, usually `exposed: false`, wired as a subgraph with `context.get_workflow_graph(...)`.
3. A score-based loop that returns to the parent work state until the reviewer score reaches the approval threshold.
4. At least two review rounds so the second pass acts as an independent verification round.

## Keep Prompts Natural
Write prompts as natural-language instructions and pass only the few state fields the next step needs:
- current task prompt
- current round or loop status
- previous artifact
- previous reviewer feedback
- previous reviewer checklist or blocking issues
- targeted repo context or file shortlist

Do not dump the full workflow state as JSON into prompts. Do not require JSON responses unless the workflow truly needs structured machine output.

## Use Tool-Capable Investigation When It Improves Accuracy
If the worker step benefits from direct repo inspection, prefer a tool-capable Codex mode and keep the tool usage read-only unless the workflow is explicitly meant to edit files. Ask the investigator to gather fresh evidence, not just rewrite the previous round.

## Load The Reference When Needed
Read [workflow-architecture.md](references/workflow-architecture.md) when you need:
- decomposition rules for single workflow vs parent plus subgraphs
- default loop and scoring guardrails
- natural-language investigator and reviewer templates
- code patterns for wiring reusable subgraphs
- regression test expectations
