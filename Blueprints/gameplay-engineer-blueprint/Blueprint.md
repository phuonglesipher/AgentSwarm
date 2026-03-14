---
name: gameplay-engineer-blueprint
entry: entry.py
version: 1.0.0
exposed: true
capabilities:
  - gameplay implementation
  - gameplay bug fixing
  - 3c feature development
  - combat feature development
  - gameplay planning
---
This blueprint is responsible for gameplay engineering work. It handles gameplay
feature development and bug fixing across 3C and combat content. The workflow
classifies a task as bug-fix or new feature, checks the existing gameplay and
design docs, builds a design doc when one does not exist, writes a planning doc,
requests review feedback from the gameplay-reviewer-blueprint until the score is
100, then implements code, self-tests, verifies compilation, and prepares commit
and pull request artifacts.
