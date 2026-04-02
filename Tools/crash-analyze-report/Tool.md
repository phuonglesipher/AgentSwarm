---
name: crash-analyze-report
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - crash report generation
  - crash handoff report
  - crash analysis formatting
---
Generates a structured crash analysis report from investigation
findings. The report serves as handoff artifact for downstream
engineer workflows to act on without re-investigation.
