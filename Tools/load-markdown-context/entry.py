from __future__ import annotations

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata


def _load_markdown_context(project_root, doc_paths: list[str], max_chars: int) -> str:
    blocks: list[str] = []
    for relative_path in doc_paths:
        path = project_root / relative_path
        if not path.exists():
            continue
        snippet = path.read_text(encoding="utf-8")[:max_chars].strip()
        blocks.append(f"# {relative_path}\n{snippet}")
    return "\n\n".join(blocks)


def build_tool(context: ToolContext, metadata: ToolMetadata):
    project_root = context.project_root

    @tool(metadata.name, response_format="content_and_artifact")
    def load_markdown_context(doc_paths: list[str], max_chars: int = 2000) -> tuple[str, dict[str, str | list[str]]]:
        """Load repo-relative markdown snippets and combine them into one context string."""

        doc_context = _load_markdown_context(project_root, doc_paths, max_chars)
        if doc_context:
            return (
                f"Loaded markdown context from {len(doc_paths)} doc(s).",
                {"doc_context": doc_context, "doc_paths": doc_paths},
            )
        return (
            "No markdown context was loaded because no matching docs were available.",
            {"doc_context": "", "doc_paths": doc_paths},
        )

    return load_markdown_context
