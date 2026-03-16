from __future__ import annotations

import re
from pathlib import Path

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata
from core.text_utils import tokenize


ASSET_EXTENSION = ".uasset"
TEXT_EXTENSIONS = {
    ".copy",
    ".csv",
    ".export",
    ".ini",
    ".json",
    ".md",
    ".txt",
    ".utxt",
    ".yaml",
    ".yml",
}
BLUEPRINT_NAME_PREFIXES = (
    "abp_",
    "bp_",
    "bpi_",
    "wbp_",
)
BLUEPRINT_PATH_HINTS = (
    "blueprint",
    "blueprints",
    "widget",
    "widgets",
    "umg",
)


def _resolve_content_roots(context: ToolContext, scope: str) -> tuple[Path, list[Path]]:
    scope_root = context.resolve_scope_root("agentswarm" if scope == "agentswarm" else "host_project")
    roots: list[Path] = []

    content_root = scope_root / "Content"
    if content_root.exists():
        roots.append(content_root)

    plugins_root = scope_root / "Plugins"
    if plugins_root.exists():
        for plugin_file in sorted(plugins_root.rglob("*.uplugin")):
            plugin_content = plugin_file.parent / "Content"
            if plugin_content.exists():
                roots.append(plugin_content)

    if roots:
        unique_roots = sorted({path.resolve() for path in roots}, key=lambda item: item.as_posix().lower())
        return scope_root, unique_roots
    return scope_root, [scope_root]


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


def _expanded_tokens(value: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return tokenize(spaced.replace("\\", "/"))


def _safe_read_text(path: Path, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def _score_companion_name(asset_path: Path, companion_path: Path) -> tuple[int, str]:
    asset_name = asset_path.name.lower()
    asset_stem = asset_path.stem.lower()
    companion_name = companion_path.name.lower()
    companion_stem = companion_path.stem.lower()

    if companion_stem == asset_stem:
        return (40, companion_name)
    if companion_name.startswith(f"{asset_name}."):
        return (35, companion_name)
    if companion_name.startswith(f"{asset_stem}."):
        return (30, companion_name)
    if companion_stem.startswith(f"{asset_stem}_") or companion_stem.startswith(f"{asset_stem}-"):
        return (25, companion_name)
    return (10, companion_name)


def _find_companion_paths(asset_path: Path) -> list[Path]:
    try:
        sibling_paths = [path for path in asset_path.parent.iterdir() if path.is_file()]
    except OSError:
        return []

    candidates: list[Path] = []
    for path in sibling_paths:
        if path == asset_path or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        score, _ = _score_companion_name(asset_path, path)
        if score > 10:
            candidates.append(path)

    return sorted(candidates, key=lambda item: (-_score_companion_name(asset_path, item)[0], item.name.lower()))


def _is_blueprint_candidate(asset_path: Path, companion_paths: list[Path]) -> bool:
    relative_lower = asset_path.as_posix().lower()
    stem_lower = asset_path.stem.lower()
    if companion_paths:
        return True
    if any(stem_lower.startswith(prefix) for prefix in BLUEPRINT_NAME_PREFIXES):
        return True
    return any(f"/{hint}/" in f"/{relative_lower}/" for hint in BLUEPRINT_PATH_HINTS)


def _score_asset_match(
    asset_path: Path,
    scope_root: Path,
    companion_paths: list[Path],
    prompt_tokens: set[str],
) -> int:
    try:
        relative_asset_path = asset_path.relative_to(scope_root).as_posix()
    except ValueError:
        return 0

    path_matches = len(prompt_tokens & _expanded_tokens(relative_asset_path))
    companion_matches = 0
    for companion_path in companion_paths[:2]:
        companion_preview = _safe_read_text(companion_path, 6000)
        try:
            relative_companion_path = companion_path.relative_to(scope_root).as_posix()
        except ValueError:
            relative_companion_path = companion_path.as_posix()
        companion_matches += len(prompt_tokens & _expanded_tokens(f"{relative_companion_path} {companion_preview}"))

    base_score = path_matches + (companion_matches * 2)
    if base_score == 0:
        return 0

    if companion_paths:
        base_score += 2
    if _is_blueprint_candidate(asset_path, companion_paths):
        base_score += 1
    return base_score


def _find_blueprint_hits(
    context: ToolContext,
    task_prompt: str,
    scope: str,
    max_hits: int,
) -> tuple[str, list[dict[str, object]]]:
    scope_root, content_roots = _resolve_content_roots(context, scope)
    prompt_tokens = _expanded_tokens(task_prompt)
    if not prompt_tokens:
        return str(scope_root), []

    exclude_roots = context.config.exclude_roots
    scored_hits: list[tuple[int, str, dict[str, object]]] = []

    for root in content_roots:
        if not root.exists():
            continue
        for asset_path in root.rglob(f"*{ASSET_EXTENSION}"):
            if not asset_path.is_file():
                continue
            if _should_skip_path(asset_path, scope_root, exclude_roots):
                continue

            companion_paths = _find_companion_paths(asset_path)
            if not _is_blueprint_candidate(asset_path, companion_paths):
                continue

            score = _score_asset_match(asset_path, scope_root, companion_paths, prompt_tokens)
            if score <= 0:
                continue

            relative_asset_path = asset_path.relative_to(scope_root).as_posix()
            relative_companion_paths = [path.relative_to(scope_root).as_posix() for path in companion_paths]
            hit = {
                "asset_path": relative_asset_path,
                "companion_paths": relative_companion_paths,
                "best_companion_path": relative_companion_paths[0] if relative_companion_paths else "",
                "has_companion_text": bool(relative_companion_paths),
                "match_score": score,
            }
            scored_hits.append((score, relative_asset_path.lower(), hit))

    scored_hits.sort(key=lambda item: (-item[0], item[1]))
    return str(scope_root), [hit for _, _, hit in scored_hits[:max_hits]]


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def find_gameplay_blueprints(
        task_prompt: str,
        scope: str = "host_project",
        max_hits: int = 5,
    ) -> tuple[str, dict[str, object]]:
        """Find host-project Blueprint assets and nearby companion text relevant to a gameplay task."""

        resolved_scope_root, blueprint_hits = _find_blueprint_hits(context, task_prompt, scope, max(1, max_hits))
        if blueprint_hits:
            hits_with_companions = sum(1 for hit in blueprint_hits if hit["has_companion_text"])
            return (
                (
                    f"Matched {len(blueprint_hits)} Blueprint asset(s); "
                    f"{hits_with_companions} include companion text."
                ),
                {
                    "blueprint_hits": blueprint_hits,
                    "scope": scope,
                    "scope_root": resolved_scope_root,
                },
            )
        return (
            "No relevant Blueprint assets or companion text matched the task prompt.",
            {
                "blueprint_hits": [],
                "scope": scope,
                "scope_root": resolved_scope_root,
            },
        )

    return find_gameplay_blueprints
