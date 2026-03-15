---
name: find-gameplay-docs
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - gameplay doc lookup
  - design doc lookup
  - repo markdown search
---
This tool finds the most relevant gameplay and design markdown documents for a
task prompt. It returns a short tool message summary and a structured artifact
containing the repo-relative markdown paths.
