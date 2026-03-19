---
name: root-project-investigation-workflow
entry: entry.py
version: 1.0.0
exposed: true
capabilities:
  - root project investigation
  - root cause analysis
  - senior engineering review
  - investigation scoring
  - clean code review
  - architecture review
  - optimization review
---
This workflow investigates the host project root through a tight two-step loop:
an `investigate` pass followed by a strict `review` pass. The investigation node
uses a tool-capable Codex run when available so it can inspect the host project,
read the relevant files, and strengthen the brief with direct evidence before
writing the markdown handoff. The review node acts like a demanding senior
engineer and scores the investigation against focus, evidence, architecture,
clean code reasoning, optimization awareness, and verification quality.

If the score is below `90/100`, the workflow loops back into investigation with
the prior investigation brief and the latest reviewer feedback. When the score
reaches `>= 90`, the workflow exits with a final report and round-by-round
artifacts. Review stays strict: final approval must be blocker-free and cannot
stick before at least two review rounds have completed, so the second pass can
act as an independent verification round. Prompts stay in natural language and
pass only specific fields from the previous state instead of serializing the
whole state as JSON.
