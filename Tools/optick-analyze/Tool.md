---
name: optick-analyze
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - optick capture parsing
  - frame timing analysis
  - performance bottleneck detection
---
Parses Optick .opt capture files and returns structured performance
data including frame timings, per-thread breakdowns, and hottest
scopes for LLM-driven performance analysis.
