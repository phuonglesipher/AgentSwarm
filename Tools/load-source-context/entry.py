from __future__ import annotations

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata


def _load_source_context(scope_root, file_paths: list[str], max_chars: int) -> str:
    blocks: list[str] = []
    for relative_path in file_paths:
        path = scope_root / relative_path
        if not path.exists():
            continue
        snippet = path.read_text(encoding="utf-8", errors="ignore")[:max_chars].strip()
        blocks.append(f"# {relative_path}\n{snippet}")
    return "\n\n".join(blocks)


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def load_source_context(
        file_paths: list[str],
        max_chars: int = 4000,
        scope: str = "host_project",
    ) -> tuple[str, dict[str, str | list[str]]]:
        """Load repo-relative source snippets and combine them into one context string."""

        scope_root = context.resolve_scope_root("agentswarm" if scope == "agentswarm" else "host_project")
        code_context = _load_source_context(scope_root, file_paths, max_chars)
        if code_context:
            return (
                f"Loaded source context from {len(file_paths)} file(s).",
                {"code_context": code_context, "file_paths": file_paths, "scope": scope},
            )
        return (
            "No source context was loaded because no matching files were available.",
            {"code_context": "", "file_paths": file_paths, "scope": scope},
        )

    return load_source_context
