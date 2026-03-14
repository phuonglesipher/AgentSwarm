from __future__ import annotations

import re
import unicodedata


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return stripped.lower()


def tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", normalize_text(value)))


def slugify(value: str, fallback: str = "task") -> str:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(value))
    if not tokens:
        return fallback
    return "-".join(tokens[:8])
