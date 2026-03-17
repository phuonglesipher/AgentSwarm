from __future__ import annotations

import re
import unicodedata


COMMON_SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "before",
    "bug",
    "bugfix",
    "build",
    "collect",
    "debug",
    "debugging",
    "engineer",
    "feature",
    "files",
    "fix",
    "for",
    "from",
    "gameplay",
    "host",
    "how",
    "ignore",
    "implement",
    "implemented",
    "implementation",
    "inspect",
    "inside",
    "investigate",
    "investigation",
    "modify",
    "most",
    "need",
    "owned",
    "project",
    "prompt",
    "relevant",
    "search",
    "should",
    "stays",
    "task",
    "that",
    "the",
    "their",
    "this",
    "through",
    "true",
    "under",
    "unless",
    "work",
    "would",
}


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return stripped.lower()


def tokenize(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", normalize_text(value)))


def keyword_tokens(value: str) -> set[str]:
    return {
        token
        for token in tokenize(value)
        if len(token) >= 3 and token not in COMMON_SEARCH_STOPWORDS and not token.isdigit()
    }


def slugify(value: str, fallback: str = "task") -> str:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(value))
    if not tokens:
        return fallback
    return "-".join(tokens[:8])
