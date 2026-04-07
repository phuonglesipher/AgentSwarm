---
name: triage-performance-workflow
entry: entry.py
version: 1.0.0
exposed: true
tools:
  - optick-analyze
  - utrace-analyze
capabilities:
  - performance triage
  - FPS drop investigation
  - profiling analysis
  - bottleneck domain detection
  - frame timing classification
  - performance optimization routing
  - utrace analysis
  - GPU profiling
---
Triage workflow for performance investigations. Analyzes profiling captures
(Optick .opt files or Unreal Insights .utrace files) to classify which
subsystem(s) are bottlenecked — game thread, rendering, streaming, or GPU —
then delegates to the appropriate specialized optimization workflow(s).
Uses DecisionEngine for LLM-driven domain classification based on bound
analysis, GPU pipeline data, and contextual bookmarks.
