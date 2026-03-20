---
name: template-investigation-workflow
entry: entry.py
version: 1.0.0
exposed: true
capabilities:
  - investigation workflow template
  - host project investigation
  - root cause analysis
  - senior engineering review
  - investigation scoring
  - clean code review
  - architecture review
  - optimization review
---
This workflow is the repository's canonical investigation-quality template. It
investigates the host project through a tight two-step loop: an `investigate`
pass followed by a strict reviewer subgraph. The investigation node uses a
tool-capable Codex run when available so it can inspect the host project, read
the relevant files, and strengthen the brief with direct evidence before
writing the markdown handoff. Senior scoring and approval now live in the
dedicated `template-investigation-reviewer-workflow` subgraph so the review
logic can evolve independently while the parent workflow keeps control of the
investigation loop.

If the score is below `90/100`, the workflow loops back into investigation with
the prior investigation brief and the latest reviewer feedback. When the score
reaches `>= 90`, the workflow exits with a final report and round-by-round
artifacts. Review stays strict: final approval must be blocker-free and cannot
stick before at least two review rounds have completed, so the second pass can
act as an independent verification round. Prompts stay in natural language and
pass only specific fields from the previous state instead of serializing the
whole state as JSON. New non-trivial workflows should match this quality bar on
loop ownership, blocker handling, reviewer depth, and regression coverage
unless there is a deliberate architectural reason to diverge.
