---
name: find-gameplay-blueprints
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - gameplay blueprint lookup
  - unreal asset discovery
  - blueprint companion text lookup
---
This tool finds the most relevant host-project Blueprint assets for a gameplay
task prompt. It searches likely Unreal content roots, scores Blueprint asset
paths and nearby companion text, and returns repo-relative asset and companion
paths so workflows can decide whether context is available in text form.
