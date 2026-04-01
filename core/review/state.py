from __future__ import annotations

from typing import Any


def apply_field_aliases(
    output: dict[str, Any],
    aliases: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """Duplicate canonical fields under alias names for backward compatibility.

    Each alias is a (canonical_name, alias_name) tuple. If the canonical field
    exists in output, its value is copied to the alias field as well.
    """
    if not aliases:
        return output
    result = dict(output)
    for canonical_name, alias_name in aliases:
        if canonical_name in result:
            result[alias_name] = result[canonical_name]
    return result
