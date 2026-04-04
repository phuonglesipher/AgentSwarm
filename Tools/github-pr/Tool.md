---
name: github-pr
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - github pull request creation
  - github pull request listing
  - github pull request detail retrieval
---
Interacts with GitHub pull requests via the gh CLI. Supports creating
new PRs, listing existing PRs with filters, and viewing PR details
including reviews and comments.
