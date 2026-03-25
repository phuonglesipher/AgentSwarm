from __future__ import annotations

from typing import Iterable


_NULL_LIKE_PROMPT_VALUES = frozenset(
    {
        "none",
        "none.",
        "- none",
        "- none.",
        "(none)",
        "n/a",
        "n/a.",
        "- n/a",
        "- n/a.",
        "null",
        "null.",
        "- null",
        "- null.",
    }
)


def _coerce_prompt_text(value: object | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_null_like_prompt_text(value: object | None) -> bool:
    clean_value = _coerce_prompt_text(value)
    if not clean_value:
        return True
    normalized = " ".join(clean_value.split()).lower()
    return normalized in _NULL_LIKE_PROMPT_VALUES


def build_prompt_brief(
    *,
    opening: object | None,
    sections: Iterable[tuple[object | None, object | None]],
    closing: object | None = None,
) -> str:
    blocks: list[str] = []
    clean_opening = "" if _is_null_like_prompt_text(opening) else _coerce_prompt_text(opening)
    if clean_opening:
        blocks.append(clean_opening)
    for title, body in sections:
        clean_title = _coerce_prompt_text(title)
        if _is_null_like_prompt_text(body):
            continue
        clean_body = _coerce_prompt_text(body)
        if not clean_title and not clean_body:
            continue
        if clean_title and clean_body:
            blocks.append(f"## {clean_title}\n{clean_body}")
        elif clean_body:
            blocks.append(clean_body)
    clean_closing = "" if _is_null_like_prompt_text(closing) else _coerce_prompt_text(closing)
    if clean_closing:
        blocks.append(clean_closing)
    return "\n\n".join(block for block in blocks if block).strip()


def build_llm_request(
    *,
    instructions: object | None,
    input_text: object | None,
    require_structured_output: bool,
) -> str:
    sections: list[str] = []
    clean_instructions = _coerce_prompt_text(instructions)
    clean_input = "" if _is_null_like_prompt_text(input_text) else _coerce_prompt_text(input_text)
    if clean_instructions:
        sections.append(clean_instructions)
    if clean_input:
        sections.append(f"Here is the current working context:\n\n{clean_input}")
    if require_structured_output:
        sections.append(
            "Respond through the configured structured output channel and do not add extra commentary."
        )
    return "\n\n".join(sections).strip()
