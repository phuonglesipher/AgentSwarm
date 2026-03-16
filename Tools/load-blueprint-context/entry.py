from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata


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


def _safe_read_text(path: Path, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars].strip()
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
    if not asset_path.exists():
        return []

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


def _load_blueprint_context_blocks(
    scope_root: Path,
    asset_paths: list[str],
    max_chars: int,
) -> tuple[str, list[dict[str, object]]]:
    blocks: list[str] = []
    asset_contexts: list[dict[str, object]] = []

    for relative_asset_path in asset_paths:
        asset_path = scope_root / relative_asset_path
        asset_exists = asset_path.is_file()
        asset_record: dict[str, object] = {
            "asset_path": relative_asset_path,
            "asset_exists": asset_exists,
            "companion_paths": [],
            "loaded_companion_path": "",
            "binary_asset": asset_path.suffix.lower() == ".uasset",
            "context_available": False,
            "manual_editor_changes_required": True,
            "status": "asset_missing",
            "notes": "Blueprint asset path was not found in the selected scope.",
        }

        if not asset_exists:
            asset_contexts.append(asset_record)
            continue

        companion_paths = _find_companion_paths(asset_path)
        relative_companion_paths = [path.relative_to(scope_root).as_posix() for path in companion_paths]
        asset_record["companion_paths"] = relative_companion_paths

        loaded_companion_path = ""
        loaded_text = ""
        for companion_path in companion_paths:
            candidate_text = _safe_read_text(companion_path, max_chars)
            if candidate_text:
                loaded_companion_path = companion_path.relative_to(scope_root).as_posix()
                loaded_text = candidate_text
                break

        if loaded_text:
            blocks.append(
                f"# {relative_asset_path}\n"
                f"Companion: {loaded_companion_path}\n"
                f"{loaded_text}"
            )
            asset_record.update(
                {
                    "loaded_companion_path": loaded_companion_path,
                    "context_available": True,
                    "manual_editor_changes_required": False,
                    "status": "loaded_companion_text",
                    "notes": "Loaded Blueprint context from companion text.",
                }
            )
        elif relative_companion_paths:
            asset_record.update(
                {
                    "status": "companion_text_empty",
                    "notes": (
                        "Companion text files were found but did not contain readable context. "
                        "The asset remains effectively binary for planning purposes."
                    ),
                }
            )
        else:
            asset_record.update(
                {
                    "status": "binary_only",
                    "notes": (
                        "No companion text was found. This Blueprint asset is binary in source control "
                        "and requires manual Unreal Editor changes."
                    ),
                }
            )

        asset_contexts.append(asset_record)

    return "\n\n".join(blocks), asset_contexts


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def load_blueprint_context(
        asset_paths: list[str],
        max_chars: int = 4000,
        scope: str = "host_project",
    ) -> tuple[str, dict[str, object]]:
        """Load Blueprint companion text when available, otherwise report binary-asset metadata."""

        scope_root = context.resolve_scope_root("agentswarm" if scope == "agentswarm" else "host_project")
        blueprint_context, asset_contexts = _load_blueprint_context_blocks(
            scope_root,
            asset_paths,
            max_chars,
        )

        loaded_count = sum(1 for item in asset_contexts if item["status"] == "loaded_companion_text")
        binary_count = sum(
            1
            for item in asset_contexts
            if item["status"] in {"binary_only", "companion_text_empty", "asset_missing"}
        )

        if blueprint_context:
            return (
                (
                    f"Loaded Blueprint context for {loaded_count} asset(s); "
                    f"{binary_count} still require manual follow-up."
                ),
                {
                    "blueprint_context": blueprint_context,
                    "asset_contexts": asset_contexts,
                    "asset_paths": asset_paths,
                    "scope": scope,
                },
            )
        return (
            "No Blueprint companion text was loaded; the selected assets are binary-only or missing.",
            {
                "blueprint_context": "",
                "asset_contexts": asset_contexts,
                "asset_paths": asset_paths,
                "scope": scope,
            },
        )

    return load_blueprint_context
