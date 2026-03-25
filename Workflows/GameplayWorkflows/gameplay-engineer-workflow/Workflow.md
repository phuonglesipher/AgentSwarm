---
name: gameplay-engineer-workflow
entry: entry.py
version: 1.0.0
exposed: true
capabilities:
  - gameplay bug fixing
  - gameplay feature delivery
  - gameplay maintenance and hardening
  - combat bug resolution
  - traversal feature implementation
  - gameplay state debugging
  - gameplay-only code and blueprint fixes
---
This workflow is the gameplay-only delivery owner. It classifies gameplay
requests into bugfix, feature, or maintenance tracks; investigates grounded
ownership; plans and reviews higher-risk gameplay changes; implements code-side
fixes when appropriate; and falls back to manual Blueprint handoff when the
change cannot be applied safely from the workspace.

Investigation is no longer a one-pass heuristic. The workflow now keeps a
two-pass minimum gameplay investigation loop before planning or implementation
approval can stick, and that loop scores the live runtime owner, current-vs-
legacy split, causal hypothesis, validation path, and noise control explicitly.
The goal is to turn extra context into better signal instead of allowing a
high-token first pass to rush into a low-quality plan.
