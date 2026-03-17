from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata
from core.text_utils import keyword_tokens, tokenize


def _resolve_doc_roots(context: ToolContext, scope: str) -> list[Path]:
    scope_root = context.resolve_scope_root("agentswarm" if scope == "agentswarm" else "host_project")
    candidate_roots = [scope_root / relative_root for relative_root in context.config.doc_roots]
    if any(path.exists() for path in candidate_roots):
        return candidate_roots
    return [
        scope_root / "docs" / "engineer",
        scope_root / "docs" / "designer",
        scope_root / "docs",
        scope_root / "design",
    ]


def _score_doc(path: Path, scope_root: Path, prompt_tokens: set[str]) -> int:
    try:
        relative_path = path.relative_to(scope_root).as_posix()
    except ValueError:
        return 0

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0

    path_tokens = keyword_tokens(relative_path)
    content_tokens = keyword_tokens(content)
    score = (len(prompt_tokens & path_tokens) * 4) + len(prompt_tokens & content_tokens)
    if score <= 0:
        return 0

    relative_lower = relative_path.lower()
    if "/architecture/" in f"/{relative_lower}/":
        score += 3
    if any(segment in relative_lower for segment in ("stun", "combat", "character", "enemy", "damage", "retribution")):
        score += 2
    if any(
        segment in relative_lower
        for segment in ("vfx/", "/vfx", "art/", "/art", "plans/", "/plans", "_workflow", "migration", "cinematic")
    ):
        score -= 3
    return score


def _find_relevant_docs(context: ToolContext, task_prompt: str, scope: str) -> tuple[str, list[str]]:
    scope_root = context.resolve_scope_root("agentswarm" if scope == "agentswarm" else "host_project")
    search_roots = _resolve_doc_roots(context, scope)
    prompt_tokens = keyword_tokens(task_prompt) or tokenize(task_prompt)
    scored_by_path: dict[str, tuple[int, Path]] = {}

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            score = _score_doc(path, scope_root, prompt_tokens)
            if score:
                relative_path = path.relative_to(scope_root).as_posix()
                previous = scored_by_path.get(relative_path)
                if previous is None or score > previous[0]:
                    scored_by_path[relative_path] = (score, path)

    scored = list(scored_by_path.values())
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return str(scope_root), [path.relative_to(scope_root).as_posix() for _, path in scored[:3]]


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
