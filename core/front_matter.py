from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    cleaned = value.strip().strip('"').strip("'")
    lowered = cleaned.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return cleaned


def parse_markdown_front_matter(markdown_path: Path) -> tuple[dict[str, Any], str]:
    content = markdown_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{markdown_path} must start with front matter delimited by ---")

    front_matter: dict[str, Any] = {}
    list_key: str | None = None
    index = 1
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "---":
            index += 1
            break
        if not stripped:
            index += 1
            continue
        if stripped.startswith("- "):
            if list_key is None:
                raise ValueError(f"List item found before a key in {markdown_path}")
            front_matter.setdefault(list_key, []).append(stripped[2:].strip())
            index += 1
            continue
        if ":" not in stripped:
            raise ValueError(f"Invalid front matter line in {markdown_path}: {stripped}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            front_matter[key] = _parse_scalar(value)
            list_key = None
        else:
            front_matter[key] = []
            list_key = key
        index += 1

    description = "\n".join(lines[index:]).strip()
    return front_matter, description
