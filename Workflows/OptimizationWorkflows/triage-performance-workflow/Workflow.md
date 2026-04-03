---
name: triage-performance-workflow
entry: entry.py
version: 1.0.0
exposed: true
tools:
  - optick-analyze
capabilities:
  - performance triage
  - FPS drop investigation
  - profiling analysis
  - bottleneck domain detection
  - frame timing classification
  - performance optimization routing
---
Triage workflow for performance investigations. Analyzes profiling captures
(Optick .opt files) to classify which subsystem(s) are bottlenecked — game
thread, rendering, or streaming — then delegates to the appropriate
specialized optimization workflow(s). Use when the performance domain is
unknown and profiling data needs classification before deep investigation.
