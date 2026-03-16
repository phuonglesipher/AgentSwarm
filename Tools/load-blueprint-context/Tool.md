---
name: load-blueprint-context
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - blueprint companion text loading
  - unreal asset context assembly
  - binary asset metadata reporting
---
This tool loads text context for repo-relative Blueprint assets when a
companion text file is available. If no companion text can be found, it returns
metadata explaining that the asset is binary in source control and requires
manual Unreal Editor changes.
