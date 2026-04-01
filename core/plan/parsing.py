from __future__ import annotations

import re


def extract_heading_block(document: str, heading: str) -> list[str]:
    """Extract the content lines under a ``## heading`` block."""
    lines = document.splitlines()
    active = False
    collected: list[str] = []
    for line in lines:
        if line.strip().lower() == f"## {heading.lower()}":
            active = True
            continue
        if active and line.startswith("## "):
            break
        if active and line.strip():
            collected.append(line.rstrip())
    return collected


def detect_missing_headings(
    document: str,
    expected: tuple[str, ...],
    *,
    min_content_length: int = 50,
) -> list[str]:
    """Return headings that are absent or have too little content."""
    if not expected:
        return []
    doc_lower = document.lower()
    missing: list[str] = []
    for heading in expected:
        if f"## {heading.lower()}" not in doc_lower:
            missing.append(heading)
            continue
        block = extract_heading_block(document, heading)
        content = "\n".join(block).strip()
        if len(content) < min_content_length:
            missing.append(heading)
    return missing


def extract_plan_summary(document: str, *, max_length: int = 200) -> str:
    """Extract a one-line summary from the first heading block or first line."""
    lines = document.strip().splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            text = stripped.lstrip("- ").strip()
            if text:
                return text[:max_length]
        if stripped:
            return stripped[:max_length]
    return ""
