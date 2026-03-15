---
name: load-markdown-context
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - markdown snippet loading
  - repo context assembly
  - design context loading
---
This tool loads repo-relative markdown files and returns a concatenated context
string suitable for workflow planning and design synthesis. The tool message
summarizes what was loaded, and the artifact contains the structured context.
