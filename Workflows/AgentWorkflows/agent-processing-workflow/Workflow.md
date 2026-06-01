---
name: agent-processing-workflow
entry: entry.py
version: 1.0.0
exposed: true
capabilities:
  - agent task processing
  - codex execution
  - code implementation
  - repository changes
---
Minimal AgentTask processing workflow. It passes the prepared task prompt to
Codex through the configured ai-gateway profile and returns a concise execution
summary for the runner.
