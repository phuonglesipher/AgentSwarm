---
name: drive-download
entry: entry.py
version: 1.0.0
output_mode: message
state_keys_shared:
  - messages
capabilities:
  - google drive file download
  - google drive folder download
  - file format filtering
---

Downloads files from Google Drive given a share link, direct link, or
file/folder ID. Supports filtering by file extension and saves files to
the run artifact directory or a custom destination. Works with public
links out of the box; supports private files when a service account key
is provided.
