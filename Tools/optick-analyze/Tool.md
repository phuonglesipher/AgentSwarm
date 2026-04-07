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
  - batch folder analysis
---
Parses Optick .opt capture files and returns structured performance
data including frame timings, per-thread breakdowns, and hottest
scopes for LLM-driven performance analysis.

Accepts a single .opt file or a directory. When given a directory,
all .opt files are analyzed sequentially (newest first) with memory
freed between files to prevent OOM on large captures.
