---
name: minidump-analyze
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - minidump parsing
  - crash call stack extraction
  - exception analysis
  - UE module identification
---
Parses Windows minidump (.dmp) files and returns structured crash data
including exception info, call stacks, loaded modules, and thread state
for crash investigation.
