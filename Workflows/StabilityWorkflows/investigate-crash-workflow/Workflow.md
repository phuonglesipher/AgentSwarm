---
name: investigate-crash-workflow
entry: entry.py
version: 1.0.0
exposed: true
tools:
  - crash-analyze-report
capabilities:
  - crash investigation
  - crash dump analysis
  - call stack analysis
  - access violation debugging
  - GPU crash investigation
  - TDR timeout analysis
  - stability debugging
  - memory corruption investigation
  - garbage collection crash analysis
  - platform crash investigation
---
Triage workflow for crash and stability issues. Investigates root cause,
classifies the responsible domain, and produces a structured handoff
report for the appropriate engineer workflow. Loops between analysis pass
and strict senior reviewer until evidence-based root cause identification
reaches quality bar.
