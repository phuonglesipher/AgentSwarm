---
name: find-gameplay-code
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - gameplay source lookup
  - gameplay test lookup
  - host project code search
---
This tool finds the most relevant host-project source and test files for a
gameplay task prompt. It returns repo-relative file paths so workflows can load
code context before planning or generating changes.
