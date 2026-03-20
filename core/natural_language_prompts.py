from __future__ import annotations

from typing import Iterable


def build_prompt_brief(
    *,
    opening: str,
    sections: Iterable[tuple[str, str]],
    closing: str | None = None,
) -> str:
    blocks: list[str] = [opening.strip()]
    for title, body in sections:
        clean_title = str(title).strip()
        clean_body = str(body).strip()
        if not clean_title and not clean_body:
            continue
        if clean_title and clean_body:
            blocks.append(f"## {clean_title}\n{clean_body}")
        elif clean_title:
            blocks.append(f"## {clean_title}")
        else:
            blocks.append(clean_body)
    if closing and closing.strip():
        blocks.append(closing.strip())
    return "\n\n".join(block for block in blocks if block).strip()


def build_llm_request(
    *,
    instructions: str,
    input_text: str,
    require_structured_output: bool,
) -> str:
    sections: list[str] = []
    clean_instructions = instructions.strip()
    clean_input = input_text.strip()
    if clean_instructions:
        sections.append(clean_instructions)
    if clean_input:
        sections.append(f"Here is the current working context:\n\n{clean_input}")
    if require_structured_output:
        sections.append(
            "Respond through the configured structured output channel and do not add extra commentary."
        )
    return "\n\n".join(sections).strip()
