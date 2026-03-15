from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata
from core.text_utils import tokenize


def _resolve_doc_roots(context: ToolContext, scope: str) -> list[Path]:
    scope_root = context.resolve_scope_root("agentswarm" if scope == "agentswarm" else "host_project")
    candidate_roots = [scope_root / relative_root for relative_root in context.config.doc_roots]
    if any(path.exists() for path in candidate_roots):
        return candidate_roots
    return [
        scope_root / "docs" / "engineer" / "gameplay",
        scope_root / "docs" / "designer",
        scope_root / "docs",
        scope_root / "design",
    ]


def _find_relevant_docs(context: ToolContext, task_prompt: str, scope: str) -> tuple[str, list[str]]:
    scope_root = context.resolve_scope_root("agentswarm" if scope == "agentswarm" else "host_project")
    search_roots = _resolve_doc_roots(context, scope)
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
    return str(scope_root), [str(path.relative_to(scope_root)) for _, path in scored[:3]]


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def find_gameplay_docs(task_prompt: str, scope: str = "host_project") -> tuple[str, dict[str, object]]:
        """Find up to three gameplay/design markdown docs relevant to a task prompt."""

        resolved_scope_root, doc_hits = _find_relevant_docs(context, task_prompt, scope)
        if doc_hits:
            return (
                f"Matched {len(doc_hits)} doc(s): {', '.join(doc_hits)}",
                {
                    "doc_hits": doc_hits,
                    "scope": scope,
                    "scope_root": resolved_scope_root,
                },
            )
        return (
            "No gameplay or design docs matched the task prompt.",
            {
                "doc_hits": [],
                "scope": scope,
                "scope_root": resolved_scope_root,
            },
        )

    return find_gameplay_docs
