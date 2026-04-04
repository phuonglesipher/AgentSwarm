---
name: github-issues
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - github issue listing
  - github issue detail retrieval
  - github issue commenting
  - github issue status management
  - github label management
---
Interacts with GitHub issues via the gh CLI. Supports listing issues
with filters, viewing issue details, adding comments, closing or
reopening issues, and managing labels.
