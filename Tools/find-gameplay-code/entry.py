from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata
from core.text_utils import tokenize


SOURCE_EXTENSIONS = {
    ".py",
    ".lua",
    ".gd",
    ".cs",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".inl",
}
TEST_FILE_NAMES = ("test_", "_test", "spec_", "_spec")


def _resolve_code_roots(context: ToolContext, scope: str, *, kind: str) -> list[Path]:
    scope_root = context.resolve_scope_root("agentswarm" if scope == "agentswarm" else "host_project")
    relative_roots = context.config.test_roots if kind == "test" else context.config.source_roots
    candidate_roots = [scope_root / relative_root for relative_root in relative_roots]
    if any(path.exists() for path in candidate_roots):
        return candidate_roots
    return [scope_root]


def _should_skip_path(path: Path, scope_root: Path, exclude_roots: tuple[str, ...]) -> bool:
    try:
        relative = path.relative_to(scope_root).as_posix()
    except ValueError:
        return True

    relative_lower = relative.lower()
    for excluded in exclude_roots:
        normalized = excluded.replace("\\", "/").strip("/").lower()
        if not normalized:
            continue
        if relative_lower == normalized or relative_lower.startswith(f"{normalized}/"):
            return True
    return False


def _score_code_path(path: Path, prompt_tokens: set[str]) -> int:
    try:
        preview = path.read_text(encoding="utf-8", errors="ignore")[:6000]
    except OSError:
        return 0
    haystack = f"{path.as_posix()} {preview}"
    return len(prompt_tokens & tokenize(haystack))


def _find_code_hits(context: ToolContext, task_prompt: str, scope: str) -> tuple[str, list[str], list[str]]:
    scope_root = context.resolve_scope_root("agentswarm" if scope == "agentswarm" else "host_project")
    prompt_tokens = tokenize(task_prompt)
    exclude_roots = context.config.exclude_roots

    source_scored: list[tuple[int, Path]] = []
    for root in _resolve_code_roots(context, scope, kind="source"):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            if _should_skip_path(path, scope_root, exclude_roots):
                continue
            score = _score_code_path(path, prompt_tokens)
            if score:
                source_scored.append((score, path))

    test_scored: list[tuple[int, Path]] = []
    for root in _resolve_code_roots(context, scope, kind="test"):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            if _should_skip_path(path, scope_root, exclude_roots):
                continue
            lower_name = path.name.lower()
            if not any(marker in lower_name for marker in TEST_FILE_NAMES):
                continue
            score = _score_code_path(path, prompt_tokens)
            if score:
                test_scored.append((score, path))

    source_scored.sort(key=lambda item: (-item[0], item[1].name))
    test_scored.sort(key=lambda item: (-item[0], item[1].name))
    source_hits = [path.relative_to(scope_root).as_posix() for _, path in source_scored[:5]]
    test_hits = [path.relative_to(scope_root).as_posix() for _, path in test_scored[:5]]
    return str(scope_root), source_hits, test_hits


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def find_gameplay_code(task_prompt: str, scope: str = "host_project") -> tuple[str, dict[str, object]]:
        """Find host-project source and test files relevant to a gameplay task."""

        resolved_scope_root, source_hits, test_hits = _find_code_hits(context, task_prompt, scope)
        summary_parts: list[str] = []
        if source_hits:
            summary_parts.append(f"{len(source_hits)} source file(s)")
        if test_hits:
            summary_parts.append(f"{len(test_hits)} test file(s)")
        if summary_parts:
            return (
                f"Matched {', '.join(summary_parts)} for gameplay code context.",
                {
                    "source_hits": source_hits,
                    "test_hits": test_hits,
                    "scope": scope,
                    "scope_root": resolved_scope_root,
                },
            )
        return (
            "No relevant host-project source or test files matched the task prompt.",
            {
                "source_hits": [],
                "test_hits": [],
                "scope": scope,
                "scope_root": resolved_scope_root,
            },
        )

    return find_gameplay_code
