---
name: load-source-context
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - source snippet loading
  - test snippet loading
  - code context assembly
---
This tool loads repo-relative source or test files and returns a concatenated
code context string suitable for planning, review, and code generation.
