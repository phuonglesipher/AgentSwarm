---
name: utrace-analyze
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - utrace binary parsing
  - CPU profiler scope analysis
  - GPU profiler pass analysis
  - memory allocation tracking
  - frame timing and spike detection
  - counter extraction
  - session diagnostics
---

Parses Unreal Insights .utrace binary files and returns structured
performance data including CPU profiler scopes, GPU pass timing,
memory allocations, counters, frame timing, and session diagnostics
for LLM-driven performance analysis.

Supports UE 5.x trace format (protocol versions 5-7, transport
versions 1-4). Handles large traces with configurable memory event
capping. Can analyze specific channels (cpu, gpu, memory, counters)
or provide a summary overview.
