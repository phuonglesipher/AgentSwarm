from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata
from core.text_utils import tokenize


def _find_relevant_docs(project_root: Path, task_prompt: str) -> list[str]:
    search_roots = [
        project_root / "docs" / "engineer" / "gameplay",
        project_root / "docs" / "designer",
    ]
    prompt_tokens = tokenize(task_prompt)
    scored: list[tuple[int, Path]] = []

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            content = path.read_text(encoding="utf-8")
            score = len(prompt_tokens & tokenize(f"{path.name} {content}"))
            if score:
                scored.append((score, path))

    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [str(path.relative_to(project_root)) for _, path in scored[:3]]


def build_tool(context: ToolContext, metadata: ToolMetadata):
    project_root = context.project_root

    @tool(metadata.name, response_format="content_and_artifact")
    def find_gameplay_docs(task_prompt: str) -> tuple[str, dict[str, list[str]]]:
        """Find up to three gameplay/design markdown docs relevant to a task prompt."""

        doc_hits = _find_relevant_docs(project_root, task_prompt)
        if doc_hits:
            return (
                f"Matched {len(doc_hits)} doc(s): {', '.join(doc_hits)}",
                {"doc_hits": doc_hits},
            )
        return ("No gameplay or design docs matched the task prompt.", {"doc_hits": []})

    return find_gameplay_docs
