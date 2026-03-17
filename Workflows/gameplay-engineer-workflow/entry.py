from __future__ import annotations

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
import unittest

from langgraph.graph import END, START, MessagesState, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_ids import to_graph_node_name
from core.graph_logging import log_graph_payload_event, trace_graph_node, trace_route_decision
from core.llm import LLMError
from core.models import WorkflowContext, WorkflowMetadata
from core.tool_graph import build_tool_call_message, find_latest_tool_message
from core.text_utils import keyword_tokens, normalize_text, slugify, tokenize


class SectionReview(TypedDict):
    section: str
    score: int
    max_score: int
    status: str
    rationale: str
    action_items: list[str]


class EngineerState(MessagesState):
    prompt: str
    task_prompt: str
    task_id: str
    run_dir: str
    task_type: str
    task_type_reason: str
    execution_track: str
    requires_architecture_review: bool
    doc_hits: list[str]
    doc_scope: str
    doc_context: str
    source_hits: list[str]
    test_hits: list[str]
    code_scope: str
    code_context: str
    blueprint_hits: list[str]
    blueprint_text_hits: list[str]
    blueprint_scope: str
    blueprint_context: str
    blueprint_editable_targets: list[str]
    implementation_medium: str
    implementation_medium_reason: str
    blueprint_fix_strategy: str
    blueprint_manual_action_required: bool
    workspace_source_file: str
    workspace_test_file: str
    workspace_write_enabled: bool
    workspace_write_summary: str
    design_doc: str
    bug_context_doc: str
    architecture_plan_doc: str
    plan_doc: str
    review_round: int
    review_score: int
    review_feedback: str
    missing_sections: list[str]
    review_section_reviews: list[SectionReview]
    review_blocking_issues: list[str]
    review_improvement_actions: list[str]
    review_approved: bool
    code_attempt: int
    generated_code: str
    generated_tests: str
    implementation_notes: str
    compile_ok: bool
    tests_ok: bool
    test_output: str
    artifact_dir: str
    final_report: dict[str, Any]
    summary: str
    score: NotRequired[int]
    feedback: NotRequired[str]
    section_reviews: NotRequired[list[SectionReview]]
    blocking_issues: NotRequired[list[str]]
    improvement_actions: NotRequired[list[str]]
    approved: NotRequired[bool]
    pending_tool_name: str
    pending_tool_call_id: str


REQUIRED_PLAN_SECTIONS = {
    "Overview": "Summarize the player-facing gameplay goal and the systems in scope.",
    "Task Type": "Describe whether this is a bug-fix or a new feature and why.",
    "Existing Docs": "Reference gameplay and design docs that informed the work.",
    "Implementation Steps": "Describe the implementation sequence, touch points, and regression safeguards.",
    "Unit Tests": "List the automated tests that must exist before code is considered done.",
    "Risks": "Record the likely implementation risks and fallback plans.",
    "Acceptance Criteria": "State the observable gameplay outcome and regression checks for completion.",
}
PLAN_SECTION_ORDER = list(REQUIRED_PLAN_SECTIONS)
REVIEW_APPROVAL_SCORE = 90
MAX_REVIEW_ROUNDS = 3
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
TEST_FILE_NAMES = ("test_", "_test", "tests", "spec_", "_spec")
BLUEPRINT_ASSET_EXTENSION = ".uasset"
BLUEPRINT_TEXT_EXTENSIONS = {
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


def _fallback_task_classification(task_prompt: str) -> tuple[str, str]:
    bugfix_keywords = {"fix", "bug", "issue", "error", "crash"}
    normalized = normalize_text(task_prompt)
    task_type = "bugfix" if bugfix_keywords & tokenize(normalized) else "feature"
    reason = "Detected bug-fix vocabulary in the task prompt." if task_type == "bugfix" else "Defaulted to feature flow."
    return task_type, reason


def _expanded_tokens(value: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    normalized = re.sub(r"[_./\\-]+", " ", spaced)
    return tokenize(normalized)


def _short_slug(value: str, *, fallback: str, max_length: int) -> str:
    slug = slugify(value, fallback=fallback)
    if len(slug) <= max_length:
        return slug

    digest = hashlib.sha1(normalize_text(value).encode("utf-8")).hexdigest()[:8]
    keep_length = max(1, max_length - len(digest) - 1)
    trimmed = slug[:keep_length].rstrip("-") or fallback
    return f"{trimmed}-{digest}"


def _artifact_task_dir_name(task_id: str) -> str:
    normalized = str(task_id).strip()
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:6]
    match = re.match(r"task-(\d+)(?:-(.*))?$", normalized)
    if match:
        task_index = match.group(1)
        suffix_source = match.group(2) or normalized
        suffix = _short_slug(suffix_source, fallback=f"task-{task_index}", max_length=14)
        return f"task-{task_index}-{suffix}-{digest}"
    suffix = _short_slug(normalized, fallback="task", max_length=14)
    return f"{suffix}-{digest}"


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


def _safe_read_text(path: Path, max_chars: int) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
    except OSError:
        return ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _extract_relative_path_candidates(raw_path: str, *, allowed_suffixes: set[str] | None = None) -> list[str]:
    value = str(raw_path).strip().replace("\\", "/")
    if not value:
        return []

    candidates: list[str] = []

    def add(candidate: str) -> None:
        normalized = candidate.strip().strip("`'\"").strip("()[]{}").replace("\\", "/")
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    add(value)
    add(re.sub(r"^[*\-\s]+", "", value))

    markdown_match = re.search(r"\[[^\]]+\]\(([^)]+)\)", value)
    if markdown_match:
        add(markdown_match.group(1))

    for separator in [" - ", " — ", " – ", "\n", "\t", " ("]:
        if separator in value:
            add(value.split(separator, 1)[0])

    suffixes = allowed_suffixes or (SOURCE_EXTENSIONS | BLUEPRINT_TEXT_EXTENSIONS | {BLUEPRINT_ASSET_EXTENSION, ".md"})
    suffix_pattern = "|".join(re.escape(suffix) for suffix in sorted(suffixes, key=len, reverse=True))
    if suffix_pattern:
        path_pattern = re.compile(
            rf"((?:[A-Za-z]:)?(?:[A-Za-z0-9_. -]+/)+[A-Za-z0-9_. -]+(?:{suffix_pattern}))",
            re.IGNORECASE,
        )
        for match in path_pattern.finditer(value):
            add(match.group(1))

    return candidates


def _resolve_existing_relative_path(
    scope_root: Path,
    raw_path: str,
    exclude_roots: tuple[str, ...],
    *,
    allowed_suffixes: set[str] | None = None,
) -> str:
    resolved_root = scope_root.resolve()
    for candidate_value in _extract_relative_path_candidates(raw_path, allowed_suffixes=allowed_suffixes):
        candidate = Path(candidate_value)
        try:
            resolved_candidate = candidate.resolve() if candidate.is_absolute() else (resolved_root / candidate).resolve()
            relative_path = resolved_candidate.relative_to(resolved_root).as_posix()
        except (OSError, ValueError):
            continue

        if allowed_suffixes is not None and Path(relative_path).suffix.lower() not in allowed_suffixes:
            continue
        if not (resolved_root / relative_path).exists():
            continue
        if _should_skip_path(resolved_root / relative_path, resolved_root, exclude_roots):
            continue
        return relative_path
    return ""


def _normalize_relative_hits(
    scope_root: Path,
    raw_hits: Any,
    exclude_roots: tuple[str, ...],
    *,
    allowed_suffixes: set[str] | None = None,
    max_hits: int = 5,
) -> list[str]:
    if not isinstance(raw_hits, list):
        return []
    hits: list[str] = []
    for item in raw_hits:
        relative_path = _resolve_existing_relative_path(
            scope_root,
            str(item),
            exclude_roots,
            allowed_suffixes=allowed_suffixes,
        )
        if relative_path and relative_path not in hits:
            hits.append(relative_path)
        if len(hits) >= max_hits:
            break
    return hits


def _resolve_code_roots(scope_root: Path, relative_roots: tuple[str, ...]) -> list[Path]:
    candidate_roots = [scope_root / relative_root for relative_root in relative_roots]
    existing = [path for path in candidate_roots if path.exists()]
    return existing or [scope_root]


def _score_code_path(path: Path, scope_root: Path, prompt_tokens: set[str]) -> int:
    try:
        relative_path = path.relative_to(scope_root).as_posix()
    except ValueError:
        return 0
    haystack = f"{relative_path} {_safe_read_text(path, 6000)}"
    return len(prompt_tokens & _expanded_tokens(haystack))


def _find_local_code_hits(
    *,
    task_prompt: str,
    scope_root: Path,
    source_roots: tuple[str, ...],
    test_roots: tuple[str, ...],
    exclude_roots: tuple[str, ...],
    max_hits: int = 5,
) -> tuple[list[str], list[str]]:
    prompt_tokens = keyword_tokens(task_prompt) or _expanded_tokens(task_prompt)
    if not prompt_tokens:
        return [], []

    source_scored: list[tuple[int, str]] = []
    for root in _resolve_code_roots(scope_root, source_roots):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            if _should_skip_path(path, scope_root, exclude_roots):
                continue
            relative_lower = path.relative_to(scope_root).as_posix().lower()
            if "/tests/" in f"/{relative_lower}/":
                continue
            score = _score_code_path(path, scope_root, prompt_tokens)
            if score <= 0:
                continue
            source_scored.append((score, path.relative_to(scope_root).as_posix()))

    test_scored: list[tuple[int, str]] = []
    for root in _resolve_code_roots(scope_root, test_roots):
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
            relative_lower = path.relative_to(scope_root).as_posix().lower()
            if not any(marker in lower_name for marker in TEST_FILE_NAMES) and "/tests/" not in f"/{relative_lower}/":
                continue
            score = _score_code_path(path, scope_root, prompt_tokens)
            if score <= 0:
                continue
            test_scored.append((score, path.relative_to(scope_root).as_posix()))

    source_scored.sort(key=lambda item: (-item[0], item[1].lower()))
    test_scored.sort(key=lambda item: (-item[0], item[1].lower()))
    source_hits = [path for _, path in source_scored[:max_hits]]
    test_hits = [path for _, path in test_scored[:max_hits]]
    return source_hits, test_hits


def _resolve_content_roots(scope_root: Path) -> list[Path]:
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

    unique_roots = sorted({path.resolve() for path in roots}, key=lambda item: item.as_posix().lower())
    return unique_roots or [scope_root]


def _find_companion_paths(asset_path: Path) -> list[Path]:
    try:
        sibling_paths = [path for path in asset_path.parent.iterdir() if path.is_file()]
    except OSError:
        return []

    asset_name = asset_path.name.lower()
    asset_stem = asset_path.stem.lower()
    candidates: list[tuple[int, str, Path]] = []
    for path in sibling_paths:
        if path == asset_path or path.suffix.lower() not in BLUEPRINT_TEXT_EXTENSIONS:
            continue
        companion_name = path.name.lower()
        companion_stem = path.stem.lower()
        score = 0
        if companion_stem == asset_stem:
            score = 40
        elif companion_name.startswith(f"{asset_name}."):
            score = 35
        elif companion_name.startswith(f"{asset_stem}."):
            score = 30
        elif companion_stem.startswith(f"{asset_stem}_") or companion_stem.startswith(f"{asset_stem}-"):
            score = 25
        if score > 0:
            candidates.append((score, companion_name, path))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [path for _, _, path in candidates]


def _is_blueprint_candidate(asset_path: Path, companion_paths: list[Path]) -> bool:
    relative_lower = asset_path.as_posix().lower()
    stem_lower = asset_path.stem.lower()
    if companion_paths:
        return True
    if any(stem_lower.startswith(prefix) for prefix in BLUEPRINT_NAME_PREFIXES):
        return True
    return any(f"/{hint}/" in f"/{relative_lower}/" for hint in BLUEPRINT_PATH_HINTS)


def _score_blueprint_asset(
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
        try:
            relative_companion_path = companion_path.relative_to(scope_root).as_posix()
        except ValueError:
            relative_companion_path = companion_path.as_posix()
        companion_matches += len(
            prompt_tokens & _expanded_tokens(f"{relative_companion_path} {_safe_read_text(companion_path, 6000)}")
        )

    base_score = path_matches + (companion_matches * 2)
    if base_score == 0:
        return 0
    if companion_paths:
        base_score += 2
    if _is_blueprint_candidate(asset_path, companion_paths):
        base_score += 1
    return base_score


def _find_local_blueprint_hits(
    *,
    task_prompt: str,
    scope_root: Path,
    exclude_roots: tuple[str, ...],
    max_hits: int = 5,
) -> tuple[list[str], list[str]]:
    prompt_tokens = keyword_tokens(task_prompt) or _expanded_tokens(task_prompt)
    if not prompt_tokens:
        return [], []

    scored_hits: list[tuple[int, str, str, list[str]]] = []
    for root in _resolve_content_roots(scope_root):
        if not root.exists():
            continue
        for asset_path in root.rglob(f"*{BLUEPRINT_ASSET_EXTENSION}"):
            if not asset_path.is_file():
                continue
            if _should_skip_path(asset_path, scope_root, exclude_roots):
                continue
            companion_paths = _find_companion_paths(asset_path)
            if not _is_blueprint_candidate(asset_path, companion_paths):
                continue
            score = _score_blueprint_asset(asset_path, scope_root, companion_paths, prompt_tokens)
            if score <= 0:
                continue
            relative_asset_path = asset_path.relative_to(scope_root).as_posix()
            relative_companions = [path.relative_to(scope_root).as_posix() for path in companion_paths]
            scored_hits.append((score, relative_asset_path.lower(), relative_asset_path, relative_companions))

    scored_hits.sort(key=lambda item: (-item[0], item[1]))
    blueprint_hits: list[str] = []
    blueprint_text_hits: list[str] = []
    for _, _, asset_path, companion_paths in scored_hits[:max_hits]:
        blueprint_hits.append(asset_path)
        for companion_path in companion_paths:
            if companion_path not in blueprint_text_hits:
                blueprint_text_hits.append(companion_path)
    return blueprint_hits, blueprint_text_hits


def _summarize_text_context(
    scope_root: Path,
    relative_paths: list[str],
    *,
    max_files: int = 4,
    max_chars_per_file: int = 900,
    max_total_chars: int = 4000,
) -> str:
    sections: list[str] = []
    total_chars = 0
    for relative_path in relative_paths[:max_files]:
        file_text = _safe_read_text(scope_root / relative_path, max_chars_per_file).strip()
        if not file_text:
            continue
        section = f"# {relative_path}\n{file_text}"
        if total_chars and total_chars + len(section) > max_total_chars:
            break
        sections.append(section)
        total_chars += len(section)
    return "\n\n".join(sections)


def _summarize_blueprint_context(
    scope_root: Path,
    blueprint_hits: list[str],
    blueprint_text_hits: list[str],
) -> str:
    text_summary = _summarize_text_context(
        scope_root,
        blueprint_text_hits,
        max_files=3,
        max_chars_per_file=900,
        max_total_chars=3200,
    )
    if text_summary:
        return text_summary
    if blueprint_hits:
        sections = [
            f"# {item}\nCompanion text not available for direct inspection."
            for item in blueprint_hits[:3]
        ]
        return "\n\n".join(sections)
    return ""


def _compose_design_doc(
    task_prompt: str,
    task_type: str,
    implementation_medium: str,
    implementation_medium_reason: str,
    doc_hits: list[str],
    doc_context: str,
    source_hits: list[str],
    test_hits: list[str],
    code_context: str,
    blueprint_hits: list[str],
    blueprint_text_hits: list[str],
    blueprint_context: str,
    workspace_source_file: str,
    workspace_test_file: str,
) -> str:
    lines = [
        "# Gameplay Design Context",
        "",
        f"Task Prompt: {task_prompt}",
        f"Task Type: {task_type}",
        f"Implementation Medium: {implementation_medium}",
        f"Medium Rationale: {implementation_medium_reason}",
        "",
        "## Existing References",
    ]
    if doc_hits:
        lines.extend([f"- {item}" for item in doc_hits])
    else:
        lines.extend(
            [
                "- No existing gameplay or design doc matched the task closely enough.",
                "- A fresh design baseline was created from the incoming prompt.",
            ]
        )

    lines.extend(
        [
            "",
            "## Design Intent",
            "- Keep the gameplay change readable for engineering and design partners.",
            "- Describe expected player-facing behavior first, then implementation notes.",
        ]
    )
    lines.extend(["", "## Codebase Touch Points"])
    if source_hits:
        lines.extend([f"- Source: {item}" for item in source_hits])
    else:
        lines.append("- No relevant host-project source files were matched.")
    if test_hits:
        lines.extend([f"- Test: {item}" for item in test_hits])
    else:
        lines.append("- No relevant host-project test files were matched.")
    if blueprint_hits or blueprint_text_hits:
        lines.extend(["", "## Blueprint Touch Points"])
        if blueprint_hits:
            lines.extend([f"- Asset: {item}" for item in blueprint_hits])
        if blueprint_text_hits:
            lines.extend([f"- Companion text: {item}" for item in blueprint_text_hits])
    if workspace_source_file or workspace_test_file:
        lines.extend(
            [
                "",
                "## Planned Workspace Targets",
                f"- Source target: {workspace_source_file or 'artifact-only'}",
                f"- Test target: {workspace_test_file or 'artifact-only'}",
            ]
        )
    if doc_context:
        lines.extend(["", "## Reference Snippets", doc_context])
    if code_context:
        lines.extend(["", "## Source Snippets", code_context])
    if blueprint_context:
        lines.extend(["", "## Blueprint Context", blueprint_context])
    return "\n".join(lines)


def _fallback_implementation_medium(
    task_prompt: str,
    source_hits: list[str],
    test_hits: list[str],
    blueprint_hits: list[str],
    blueprint_text_hits: list[str],
) -> tuple[str, str]:
    normalized = normalize_text(task_prompt)
    tokens = tokenize(normalized)
    mentions_blueprint = any(token in tokens for token in {"blueprint", "uasset", "bp"})
    mentions_cpp = "c++" in normalized or "cpp" in tokens
    has_code = bool(source_hits or test_hits)
    has_blueprints = bool(blueprint_hits or blueprint_text_hits)

    if has_code and has_blueprints:
        return "mixed", "Matched both source code files and Blueprint assets related to the task."
    if has_blueprints and (mentions_blueprint or not has_code):
        return "blueprint", "Matched Blueprint assets and did not find stronger code-side evidence."
    if has_code:
        reason = "Matched source/test files for a code-side gameplay fix."
        if mentions_cpp:
            reason = "The prompt references C++/cpp work and matched source/test files."
        return "cpp", reason
    if has_blueprints:
        return "blueprint", "Only Blueprint assets were matched for this gameplay task."
    if mentions_blueprint and not mentions_cpp:
        return "blueprint", "The prompt explicitly references Blueprint work."
    return "cpp", "Defaulted to the code-side implementation track because no Blueprint evidence was found."


def _compose_bug_context_doc(
    task_prompt: str,
    implementation_medium: str,
    implementation_medium_reason: str,
    doc_hits: list[str],
    source_hits: list[str],
    test_hits: list[str],
    blueprint_hits: list[str],
    blueprint_text_hits: list[str],
    blueprint_fix_strategy: str,
    doc_context: str,
    code_context: str,
    blueprint_context: str,
) -> str:
    lines = [
        "# Gameplay Bug Context",
        "",
        f"Task Prompt: {task_prompt}",
        f"Execution Track: bugfix",
        f"Implementation Medium: {implementation_medium}",
        f"Medium Rationale: {implementation_medium_reason}",
        f"Blueprint Fix Strategy: {blueprint_fix_strategy}",
        "",
        "## Investigation Summary",
        "- Gather the current bug symptoms, the likely failing state transition, and the code or asset ownership before editing.",
        "- Prefer the narrowest safe fix that restores intended gameplay behavior and keeps adjacent states stable.",
        "",
        "## Existing References",
    ]
    if doc_hits:
        lines.extend(f"- {item}" for item in doc_hits)
    else:
        lines.append("- No matching gameplay or design docs were found.")

    lines.extend(["", "## Code Signals"])
    if source_hits:
        lines.extend(f"- Source: {item}" for item in source_hits)
    else:
        lines.append("- No source files were matched.")
    if test_hits:
        lines.extend(f"- Test: {item}" for item in test_hits)
    else:
        lines.append("- No test files were matched.")

    lines.extend(["", "## Blueprint Signals"])
    if blueprint_hits:
        lines.extend(f"- Asset: {item}" for item in blueprint_hits)
    else:
        lines.append("- No Blueprint assets were matched.")
    if blueprint_text_hits:
        lines.extend(f"- Companion text: {item}" for item in blueprint_text_hits)
    else:
        lines.append("- No readable Blueprint companion exports were matched.")

    lines.extend(
        [
            "",
            "## Fix Plan",
            "- Confirm the exact repro path and the first state or node where behavior diverges from intent.",
            "- Apply the narrowest change in the identified medium, then validate the primary bug path and neighboring transitions.",
            "- Preserve or add regression coverage when the fix lands in source code.",
        ]
    )
    if doc_context:
        lines.extend(["", "## Reference Snippets", doc_context])
    if code_context:
        lines.extend(["", "## Source Snippets", code_context])
    if blueprint_context:
        lines.extend(["", "## Blueprint Context", blueprint_context])
    return "\n".join(lines)


def _compose_blueprint_fix_instructions(
    task_prompt: str,
    implementation_medium: str,
    blueprint_hits: list[str],
    blueprint_text_hits: list[str],
    blueprint_fix_strategy: str,
    bug_context_doc: str,
    blueprint_context: str,
) -> str:
    lines = [
        "# Blueprint Fix Instructions",
        "",
        f"Task Prompt: {task_prompt}",
        f"Implementation Medium: {implementation_medium}",
        f"Fix Strategy: {blueprint_fix_strategy}",
        "",
        "## Target Assets",
    ]
    if blueprint_hits:
        lines.extend(f"- {item}" for item in blueprint_hits)
    else:
        lines.append("- No concrete Blueprint asset path was matched; start from the bug context and search the owning gameplay asset.")

    lines.extend(["", "## Readable Blueprint Context"])
    if blueprint_text_hits:
        lines.extend(f"- Companion export: {item}" for item in blueprint_text_hits)
    else:
        lines.append("- No readable Blueprint companion export was available. Treat the `.uasset` as editor-only data.")

    lines.extend(
        [
            "",
            "## Recommended Change",
            "- Open the matched Blueprint asset and inspect the event graph, state transition, or gameplay function named by the bug context.",
            "- Align the failing branch, gate, or timing condition with the intended gameplay behavior described in the task prompt.",
            "- Preserve adjacent transitions and add debug breadcrumbs or comments in any exported companion text when the project supports that workflow.",
            "",
            "## Validation",
            "- Reproduce the original gameplay bug before editing so the failing behavior is explicit.",
            "- Re-run the same gameplay path after the change and verify adjacent states, inputs, and timing windows still behave correctly.",
            "- If source code also changed, run the automated tests for the code-side portion of the fix.",
        ]
    )
    if bug_context_doc:
        lines.extend(["", "## Bug Context", bug_context_doc])
    if blueprint_context:
        lines.extend(["", "## Blueprint Context", blueprint_context])
    return "\n".join(lines)


def _compose_initial_plan(task_prompt: str, task_type: str, doc_hits: list[str]) -> str:
    return _render_plan_doc(
        _build_plan_sections(
            task_prompt=task_prompt,
            task_type=task_type,
            doc_hits=doc_hits,
        )
    )


def _parse_plan_sections(plan_doc: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    buffer: list[str] = []

    for line in plan_doc.splitlines():
        if line.startswith("## "):
            if current_section is not None:
                sections[current_section] = [item for item in buffer if item.strip()]
            heading = line[3:].strip()
            current_section = heading if heading in PLAN_SECTION_ORDER else None
            buffer = []
            continue

        if current_section is not None:
            buffer.append(line.rstrip())

    if current_section is not None:
        sections[current_section] = [item for item in buffer if item.strip()]
    return sections


def _dedupe_section_lines(lines: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        ordered.append(line)
    return ordered


def _default_section_lines(section: str, task_prompt: str, task_type: str, doc_hits: list[str]) -> list[str]:
    if section == "Overview":
        return [
            f"- Gameplay task: {task_prompt}",
            "- Player-facing goal: make the target gameplay flow behave predictably without regressing nearby interactions.",
        ]
    if section == "Task Type":
        task_reason = (
            "restore intended gameplay behavior and remove a regression or bug."
            if task_type == "bugfix"
            else "add or extend gameplay behavior without breaking existing expectations."
        )
        return [
            f"- {task_type}",
            f"- Classification reason: treat this as a {task_type} because the requested work should {task_reason}",
        ]
    if section == "Existing Docs":
        if doc_hits:
            return [f"- {item}" for item in doc_hits] + [
                "- Use the referenced docs as the implementation baseline and call out any conflicts before coding.",
            ]
        return [
            "- No matching docs found.",
            "- Proceed from the incoming task prompt and document any assumptions before implementation starts.",
        ]
    if section == "Implementation Steps":
        return [
            "- Inspect the current gameplay flow and identify the systems, states, or assets touched by the task.",
            "- Implement the behavior change in an ordered sequence so state transitions remain predictable.",
            "- Add logging, assertions, or debug breadcrumbs that make gameplay regressions fast to detect.",
        ]
    if section == "Unit Tests":
        return [
            "- Add or update automated tests that cover the requested gameplay path and its expected state transition.",
            "- Verify the regression case described in the task prompt and assert the intended gameplay outcome.",
        ]
    if section == "Risks":
        return [
            "- Risk: adjacent gameplay states, timings, or animation hooks may regress when this change lands.",
            "- Mitigation: add validation, fallback guards, or targeted debug logging before shipping the change.",
        ]
    return [
        f"- Players should observe the intended gameplay behavior from: {task_prompt}",
        "- Regression checks for adjacent states, timing windows, and nearby inputs should still pass before the task is complete.",
    ]


def _build_plan_sections(
    task_prompt: str,
    task_type: str,
    doc_hits: list[str],
    current_sections: dict[str, list[str]] | None = None,
    review_section_reviews: list[SectionReview] | None = None,
) -> dict[str, list[str]]:
    current_sections = current_sections or {}
    review_section_map = {
        review["section"]: review
        for review in review_section_reviews or []
    }
    built_sections: dict[str, list[str]] = {}

    for section in PLAN_SECTION_ORDER:
        lines = list(current_sections.get(section, []))
        review = review_section_map.get(section)

        if not lines:
            lines.extend(_default_section_lines(section, task_prompt, task_type, doc_hits))

        if review is not None and review["status"] != "pass":
            lines.extend(_default_section_lines(section, task_prompt, task_type, doc_hits))
            lines.extend(f"- Reviewer follow-up: {item}" for item in review["action_items"])

        built_sections[section] = _dedupe_section_lines(lines)

    return built_sections


def _render_plan_doc(sections: dict[str, list[str]]) -> str:
    lines = ["# Gameplay Implementation Plan", ""]
    for section in PLAN_SECTION_ORDER:
        lines.append(f"## {section}")
        section_lines = sections.get(section, [])
        if section_lines:
            lines.extend(section_lines)
        else:
            lines.append(f"- {REQUIRED_PLAN_SECTIONS[section]}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _workspace_roots_exist(scope_root: Path, relative_roots: tuple[str, ...]) -> bool:
    return any((scope_root / relative_root).exists() for relative_root in relative_roots)


def _select_workspace_root(scope_root: Path, relative_roots: tuple[str, ...], *, fallback: str) -> Path:
    for relative_root in relative_roots:
        candidate = Path(relative_root)
        if (scope_root / candidate).exists():
            return candidate
    if relative_roots:
        return Path(relative_roots[0])
    return Path(fallback)


def _build_workspace_target(
    scope_root: Path,
    task_id: str,
    hits: list[str],
    relative_roots: tuple[str, ...],
    *,
    prefix: str,
) -> str:
    if hits:
        return Path(hits[0]).as_posix()
    task_slug = _short_slug(task_id, fallback="gameplay-task", max_length=18).replace("-", "_")
    base_root = _select_workspace_root(
        scope_root,
        relative_roots,
        fallback="tests" if prefix.startswith("test") else "src",
    )
    return (base_root / f"{prefix}_{task_slug}.py").as_posix()


def _resolve_workspace_targets(
    task_id: str,
    scope_root: Path,
    source_hits: list[str],
    test_hits: list[str],
    source_roots: tuple[str, ...],
    test_roots: tuple[str, ...],
    *,
    allow_workspace_writes: bool,
) -> tuple[str, str]:
    if not allow_workspace_writes:
        return "", ""

    source_file = _build_workspace_target(
        scope_root,
        task_id,
        source_hits,
        source_roots,
        prefix="agentswarm_gameplay_change",
    )
    test_file = _build_workspace_target(
        scope_root,
        task_id,
        test_hits,
        test_roots,
        prefix="test_agentswarm_gameplay_change",
    )

    # Keep the generated files inside the selected workspace scope.
    for relative_path in [source_file, test_file]:
        resolved_path = (scope_root / relative_path).resolve()
        resolved_path.relative_to(scope_root.resolve())

    return source_file, test_file


def _write_workspace_file(scope_root: Path, relative_path: str, content: str) -> str:
    resolved_root = scope_root.resolve()
    target_path = (resolved_root / relative_path).resolve()
    target_path.relative_to(resolved_root)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return str(target_path)


def _mirror_existing_workspace_bundle(
    *,
    scope_root: Path,
    source_relative_path: str,
    test_relative_path: str,
) -> dict[str, str] | None:
    if not source_relative_path or not test_relative_path:
        return None

    source_code = _read_text(scope_root / source_relative_path)
    test_code = _read_text(scope_root / test_relative_path)
    if not source_code.strip() or not test_code.strip():
        return None

    return {
        "source_code": source_code,
        "test_code": test_code,
        "implementation_notes": (
            "Deterministic fallback mirrored the matched host-project source and test files so the "
            "workflow could validate existing workspace targets without inventing a shadow module."
        ),
    }


def _revise_plan(
    plan_doc: str,
    task_prompt: str,
    task_type: str,
    doc_hits: list[str],
    review_section_reviews: list[SectionReview],
) -> str:
    current_sections = _parse_plan_sections(plan_doc)
    revised_sections = _build_plan_sections(
        task_prompt=task_prompt,
        task_type=task_type,
        doc_hits=doc_hits,
        current_sections=current_sections,
        review_section_reviews=review_section_reviews,
    )
    return _render_plan_doc(revised_sections)


def _fallback_code_bundle(task_prompt: str, task_type: str, attempt: int) -> dict[str, str]:
    normalized_prompt = normalize_text(task_prompt)
    task_slug = slugify(task_prompt, fallback="gameplay-task").replace("-", "_")
    expected_unit_tests = ["compiles", "returns_task_metadata", "captures_task_type", "records_review_score"]

    if attempt == 1:
        source_code = "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def build_gameplay_change_summary() -> dict:",
                "    return {",
                f'        "task_id": "{task_slug}",',
                f'        "task_type": "{task_type}",',
                f'        "prompt": "{normalized_prompt}",',
                '        "implementation_status": "draft",',
                "    }",
            ]
        )
    else:
        source_code = "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def build_gameplay_change_summary() -> dict:",
                "    return {",
                f'        "task_id": "{task_slug}",',
                f'        "task_type": "{task_type}",',
                f'        "prompt": "{normalized_prompt}",',
                '        "implementation_status": "ready-for-review",',
                f'        "unit_tests": {expected_unit_tests},',
                "    }",
            ]
        )

    test_code = "\n".join(
        [
            "from gameplay_change import build_gameplay_change_summary",
            "",
            "def test_build_gameplay_change_summary():",
            "    summary = build_gameplay_change_summary()",
            '    assert summary["task_type"] in {"bugfix", "feature"}',
            '    assert summary["implementation_status"] == "ready-for-review"',
            '    assert "unit_tests" in summary',
        ]
    )
    notes = "Deterministic fallback code bundle was generated."
    return {
        "source_code": source_code,
        "test_code": test_code,
        "implementation_notes": notes,
    }


def _build_tool_call_id(state: EngineerState, tool_name: str) -> str:
    message_count = len(state.get("messages", []))
    return f"{state['task_id']}-{tool_name}-{message_count + 1}"


def _extract_tool_artifact(state: EngineerState, tool_name: str) -> dict[str, Any]:
    tool_message = find_latest_tool_message(
        list(state.get("messages", [])),
        tool_name=tool_name,
        tool_call_id=state.get("pending_tool_call_id") or None,
    )
    if tool_message is None:
        raise RuntimeError(f"Expected ToolMessage for {tool_name}, but no matching tool result was found.")
    if isinstance(tool_message.artifact, dict):
        return tool_message.artifact
    return {}


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_runtime_relative_path(relative_path: str, fallback: str) -> Path:
    value = str(relative_path or "").strip().replace("\\", "/")
    if not value:
        return Path(fallback)

    candidate = Path(value)
    if candidate.is_absolute():
        return Path(fallback)

    parts: list[str] = []
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            return Path(fallback)
        parts.append(part)
    if not parts:
        return Path(fallback)
    return Path(*parts)


def _run_compile_and_tests(
    source_code: str,
    test_code: str,
    *,
    source_relative_path: str = "gameplay_change.py",
    test_relative_path: str = "test_gameplay_change.py",
) -> tuple[bool, bool, str]:
    source_runtime_path = _normalize_runtime_relative_path(source_relative_path, "gameplay_change.py")
    test_runtime_path = _normalize_runtime_relative_path(test_relative_path, "test_gameplay_change.py")

    try:
        compile(source_code, source_runtime_path.as_posix(), "exec")
    except SyntaxError as exc:
        return False, False, f"Compile error in {source_runtime_path.name}: {exc}"

    try:
        compile(test_code, test_runtime_path.as_posix(), "exec")
    except SyntaxError as exc:
        return False, False, f"Compile error in {test_runtime_path.name}: {exc}"

    aliased_module_names = [
        "gameplay_change",
        "solution",
        "main",
        "gameplay_change_summary",
    ]
    previous_modules = {name: sys.modules.get(name) for name in aliased_module_names}
    generated_test_module_name = "_generated_test_gameplay_change"
    previous_generated_test_module = sys.modules.get(generated_test_module_name)
    original_sys_path = list(sys.path)
    previous_source_env = os.environ.get("GAMEPLAY_CHANGE_SOURCE_PATH")

    try:
        with tempfile.TemporaryDirectory(prefix="gameplay-selftest-") as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / source_runtime_path
            test_path = temp_path / test_runtime_path
            source_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(source_code, encoding="utf-8")
            test_path.write_text(test_code, encoding="utf-8")

            sys.path.insert(0, str(temp_path))
            os.environ["GAMEPLAY_CHANGE_SOURCE_PATH"] = str(source_path)

            source_module = _load_module_from_path("gameplay_change", source_path)
            for alias in aliased_module_names:
                sys.modules[alias] = source_module

            test_module = _load_module_from_path(generated_test_module_name, test_path)
            function_tests = [
                value
                for name, value in vars(test_module).items()
                if name.startswith("test_") and callable(value)
            ]
            for test_func in function_tests:
                test_func()

            suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
            unittest_cases = suite.countTestCases()
            if unittest_cases:
                stream = io.StringIO()
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
                if not result.wasSuccessful():
                    return True, False, stream.getvalue().strip()

            total_tests = len(function_tests) + unittest_cases
            if total_tests == 0:
                return True, False, "No generated test functions or unittest cases were found."

    except AssertionError as exc:
        message = str(exc) or "Generated assertion failed."
        return True, False, f"Unit test failed: {message}"
    except SyntaxError as exc:
        file_name = Path(exc.filename or "").name or "generated file"
        return False, False, f"Compile error in {file_name}: {exc}"
    except Exception as exc:
        return True, False, f"Unit test execution failed: {exc}"
    finally:
        sys.path[:] = original_sys_path
        if previous_source_env is None:
            os.environ.pop("GAMEPLAY_CHANGE_SOURCE_PATH", None)
        else:
            os.environ["GAMEPLAY_CHANGE_SOURCE_PATH"] = previous_source_env
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module
        if previous_generated_test_module is None:
            sys.modules.pop(generated_test_module_name, None)
        else:
            sys.modules[generated_test_module_name] = previous_generated_test_module

    return True, True, f"Compile and {total_tests} generated test(s) passed."


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    default_llm = context.llm
    graph_name = metadata.name
    reviewer_graph = context.get_workflow_graph("gameplay-reviewer-workflow")
    tool_subgraphs = context.register_tools(metadata.tools, EngineerState)
    doc_search_tool_name = context.get_tool("find-gameplay-docs").metadata.qualified_name
    doc_context_tool_name = context.get_tool("load-markdown-context").metadata.qualified_name
    tool_node_names = {
        tool_name: to_graph_node_name(tool_name)
        for tool_name in tool_subgraphs
    }

    def classify_request(state: EngineerState) -> dict[str, Any]:
        artifact_dir = Path(state["run_dir"]) / "tasks" / _artifact_task_dir_name(state["task_id"]) / metadata.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        task_type, task_type_reason = _fallback_task_classification(state["task_prompt"])

        if default_llm.is_enabled():
            schema = {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "enum": ["bugfix", "feature"]},
                    "reason": {"type": "string"},
                },
                "required": ["task_type", "reason"],
                "additionalProperties": False,
            }
            try:
                result = default_llm.generate_json(
                    instructions=(
                        "You are gameplay-engineer-workflow. Decide whether the task is a gameplay bugfix or a new gameplay feature. "
                        "Use bugfix only when the request is about fixing an issue, regression, crash, or unintended behavior."
                    ),
                    input_text=f"Task prompt:\n{state['task_prompt']}",
                    schema_name="gameplay_task_classification",
                    schema=schema,
                )
                task_type = result["task_type"]
                task_type_reason = result["reason"]
            except LLMError:
                pass

        return {
            "task_type": task_type,
            "task_type_reason": task_type_reason,
            "execution_track": task_type,
            "requires_architecture_review": task_type == "feature",
            "doc_hits": [],
            "doc_scope": "host_project",
            "doc_context": "",
            "source_hits": [],
            "test_hits": [],
            "code_scope": "host_project",
            "code_context": "",
            "blueprint_hits": [],
            "blueprint_text_hits": [],
            "blueprint_scope": "host_project",
            "blueprint_context": "",
            "blueprint_editable_targets": [],
            "implementation_medium": "",
            "implementation_medium_reason": "",
            "blueprint_fix_strategy": "not-applicable",
            "blueprint_manual_action_required": False,
            "workspace_source_file": "",
            "workspace_test_file": "",
            "workspace_write_enabled": False,
            "workspace_write_summary": "",
            "review_round": 0,
            "review_score": 0,
            "review_feedback": "",
            "missing_sections": [],
            "review_section_reviews": [],
            "review_blocking_issues": [],
            "review_improvement_actions": [],
            "review_approved": False,
            "code_attempt": 0,
            "design_doc": "",
            "bug_context_doc": "",
            "architecture_plan_doc": "",
            "plan_doc": "",
            "generated_code": "",
            "generated_tests": "",
            "implementation_notes": "",
            "compile_ok": False,
            "tests_ok": False,
            "test_output": "",
            "artifact_dir": str(artifact_dir),
        }

    def prepare_doc_search(state: EngineerState) -> dict[str, Any]:
        tool_name = doc_search_tool_name
        call_id = _build_tool_call_id(state, tool_name)
        return {
            "pending_tool_name": tool_name,
            "pending_tool_call_id": call_id,
            "messages": [
                build_tool_call_message(
                    tool_name,
                    {"task_prompt": state["task_prompt"], "scope": "host_project"},
                    call_id,
                    content="Find the gameplay and design docs most relevant to this task.",
                )
            ],
        }

    def route_tool_request(state: EngineerState) -> str:
        tool_name = state.get("pending_tool_name", "").strip()
        if tool_name in tool_subgraphs:
            return tool_node_names[tool_name]
        return "tool_request_error"

    def tool_request_error(state: EngineerState) -> dict[str, Any]:
        raise RuntimeError(
            f"{metadata.name} requested an unregistered tool: {state.get('pending_tool_name') or '(empty)'}"
        )

    def capture_doc_hits(state: EngineerState) -> dict[str, Any]:
        artifact = _extract_tool_artifact(state, doc_search_tool_name)
        raw_hits = artifact.get("doc_hits", [])
        doc_hits = [str(item) for item in raw_hits] if isinstance(raw_hits, list) else []
        doc_scope = str(artifact.get("scope") or "host_project")
        return {
            "doc_hits": doc_hits,
            "doc_scope": doc_scope,
            "pending_tool_name": "",
            "pending_tool_call_id": "",
        }

    def prepare_doc_context_lookup(state: EngineerState) -> dict[str, Any]:
        tool_name = doc_context_tool_name
        call_id = _build_tool_call_id(state, tool_name)
        return {
            "pending_tool_name": tool_name,
            "pending_tool_call_id": call_id,
            "messages": [
                build_tool_call_message(
                    tool_name,
                    {"doc_paths": state["doc_hits"], "max_chars": 2000, "scope": state["doc_scope"]},
                    call_id,
                    content="Load markdown snippets for the matched gameplay and design docs.",
                )
            ],
        }

    def capture_doc_context(state: EngineerState) -> dict[str, Any]:
        artifact = _extract_tool_artifact(state, doc_context_tool_name)
        doc_context = str(artifact.get("doc_context") or "")
        return {
            "doc_context": doc_context,
            "pending_tool_name": "",
            "pending_tool_call_id": "",
        }

    def simulate_engineer_investigation(state: EngineerState) -> dict[str, Any]:
        scope_root = context.resolve_scope_root("host_project")
        exclude_roots = context.config.exclude_roots
        source_hits: list[str] = []
        test_hits: list[str] = []
        blueprint_hits: list[str] = []
        blueprint_text_hits: list[str] = []
        code_context = ""
        blueprint_context = ""

        if default_llm.is_enabled():
            schema = {
                "type": "object",
                "properties": {
                    "source_hits": {"type": "array", "items": {"type": "string"}},
                    "test_hits": {"type": "array", "items": {"type": "string"}},
                    "blueprint_hits": {"type": "array", "items": {"type": "string"}},
                    "blueprint_text_hits": {"type": "array", "items": {"type": "string"}},
                    "code_context": {"type": "string"},
                    "blueprint_context": {"type": "string"},
                },
                "required": [
                    "source_hits",
                    "test_hits",
                    "blueprint_hits",
                    "blueprint_text_hits",
                    "code_context",
                    "blueprint_context",
                ],
                "additionalProperties": False,
            }
            try:
                result = default_llm.generate_json(
                    instructions=(
                        "You are gameplay-engineer-workflow. Simulate how a senior gameplay engineer would investigate "
                        "this task from the host project root. Search the host project yourself and return only the most "
                        "relevant host-project filesystem paths plus concise context summaries. Prefer gameplay-owned code "
                        "and assets. Avoid generated, third-party, marketplace, or unrelated framework files unless they are "
                        "directly responsible for the bug or feature. Return relative filesystem paths such as "
                        "`Source/...`, `Plugins/...`, `Content/...`, `src/...`, or `tests/...`, not Unreal `/Game/...` refs."
                    ),
                    input_text=(
                        f"Host project root: {context.host_root}\n"
                        f"Source roots: {context.config.source_roots}\n"
                        f"Test roots: {context.config.test_roots}\n"
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Task type: {state['task_type']}\n"
                        f"Task type reason: {state['task_type_reason']}\n\n"
                        f"Relevant gameplay docs:\n{state['doc_hits']}\n\n"
                        f"Doc context:\n{state['doc_context'] or 'No matching docs.'}\n\n"
                        "Do not modify files. Investigate like a gameplay engineer preparing the next implementation prompt."
                    ),
                    schema_name="gameplay_engineering_context",
                    schema=schema,
                )
                source_hits = _normalize_relative_hits(
                    scope_root,
                    result.get("source_hits"),
                    exclude_roots,
                    allowed_suffixes=SOURCE_EXTENSIONS,
                )
                test_hits = _normalize_relative_hits(
                    scope_root,
                    result.get("test_hits"),
                    exclude_roots,
                    allowed_suffixes=SOURCE_EXTENSIONS,
                )
                blueprint_hits = _normalize_relative_hits(
                    scope_root,
                    result.get("blueprint_hits"),
                    exclude_roots,
                    allowed_suffixes={BLUEPRINT_ASSET_EXTENSION},
                )
                blueprint_text_hits = _normalize_relative_hits(
                    scope_root,
                    result.get("blueprint_text_hits"),
                    exclude_roots,
                    allowed_suffixes=BLUEPRINT_TEXT_EXTENSIONS,
                )
                code_context = str(result.get("code_context") or "").strip()
                blueprint_context = str(result.get("blueprint_context") or "").strip()
            except (AssertionError, KeyError, TypeError, ValueError, LLMError):
                source_hits = []
                test_hits = []
                blueprint_hits = []
                blueprint_text_hits = []
                code_context = ""
                blueprint_context = ""

        if not any([source_hits, test_hits, blueprint_hits, blueprint_text_hits]):
            source_hits, test_hits = _find_local_code_hits(
                task_prompt=state["task_prompt"],
                scope_root=scope_root,
                source_roots=context.config.source_roots,
                test_roots=context.config.test_roots,
                exclude_roots=exclude_roots,
            )
            blueprint_hits, blueprint_text_hits = _find_local_blueprint_hits(
                task_prompt=state["task_prompt"],
                scope_root=scope_root,
                exclude_roots=exclude_roots,
            )

        if not code_context:
            code_context = _summarize_text_context(scope_root, [*source_hits, *test_hits])
        if not blueprint_context:
            blueprint_context = _summarize_blueprint_context(scope_root, blueprint_hits, blueprint_text_hits)

        workspace_write_enabled = _workspace_roots_exist(scope_root, context.config.source_roots)
        workspace_source_file, workspace_test_file = _resolve_workspace_targets(
            task_id=state["task_id"],
            scope_root=scope_root,
            source_hits=source_hits,
            test_hits=test_hits,
            source_roots=context.config.source_roots,
            test_roots=context.config.test_roots,
            allow_workspace_writes=workspace_write_enabled,
        )
        workspace_write_summary = (
            (
                "Workflow will update the matched host-project source/test files in place."
                if workspace_source_file in source_hits or workspace_test_file in test_hits
                else "Workflow will write generated code into the host project source/test roots."
            )
            if workspace_write_enabled
            else "No host-project source/test roots were available, so generated code will stay in workflow artifacts."
        )
        blueprint_editable_targets = list(blueprint_text_hits)

        investigation_lines = [
            "# Engineer Investigation",
            "",
            f"Task Prompt: {state['task_prompt']}",
            f"Task Type: {state['task_type']}",
            "",
            "## Source Hits",
            *([f"- {item}" for item in source_hits] or ["- None."]),
            "",
            "## Test Hits",
            *([f"- {item}" for item in test_hits] or ["- None."]),
            "",
            "## Blueprint Hits",
            *([f"- {item}" for item in blueprint_hits] or ["- None."]),
            "",
            "## Blueprint Companion Text",
            *([f"- {item}" for item in blueprint_text_hits] or ["- None."]),
            "",
            "## Code Context",
            code_context or "No concise code context was produced.",
            "",
            "## Blueprint Context",
            blueprint_context or "No readable Blueprint context was produced.",
        ]
        artifact_dir = Path(state["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "engineer_investigation.md").write_text("\n".join(investigation_lines), encoding="utf-8")

        return {
            "source_hits": source_hits,
            "test_hits": test_hits,
            "code_scope": "host_project",
            "code_context": code_context,
            "blueprint_hits": blueprint_hits,
            "blueprint_text_hits": blueprint_text_hits,
            "blueprint_scope": "host_project",
            "blueprint_context": blueprint_context,
            "blueprint_editable_targets": blueprint_editable_targets,
            "workspace_source_file": workspace_source_file,
            "workspace_test_file": workspace_test_file,
            "workspace_write_enabled": workspace_write_enabled,
            "workspace_write_summary": workspace_write_summary,
            "pending_tool_name": "",
            "pending_tool_call_id": "",
        }

    def assess_implementation_strategy(state: EngineerState) -> dict[str, Any]:
        implementation_medium, implementation_medium_reason = _fallback_implementation_medium(
            task_prompt=state["task_prompt"],
            source_hits=state["source_hits"],
            test_hits=state["test_hits"],
            blueprint_hits=state["blueprint_hits"],
            blueprint_text_hits=state["blueprint_text_hits"],
        )

        if default_llm.is_enabled():
            schema = {
                "type": "object",
                "properties": {
                    "implementation_medium": {"type": "string", "enum": ["cpp", "blueprint", "mixed"]},
                    "reason": {"type": "string"},
                },
                "required": ["implementation_medium", "reason"],
                "additionalProperties": False,
            }
            try:
                result = default_llm.generate_json(
                    instructions=(
                        "You are gameplay-engineer-workflow. Decide whether the gameplay work should primarily be "
                        "handled in C++/code, Blueprint, or a mixed implementation. Prefer mixed only when both "
                        "code and Blueprint evidence are materially relevant."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Task type: {state['task_type']}\n\n"
                        f"Source hits:\n{state['source_hits']}\n\n"
                        f"Test hits:\n{state['test_hits']}\n\n"
                        f"Blueprint hits:\n{state['blueprint_hits']}\n\n"
                        f"Blueprint context:\n{state['blueprint_context'] or 'No readable Blueprint context.'}\n\n"
                        f"Code context:\n{state['code_context'] or 'No matching source files.'}"
                    ),
                    schema_name="gameplay_implementation_medium",
                    schema=schema,
                )
                implementation_medium = str(result["implementation_medium"])
                implementation_medium_reason = str(result["reason"])
            except LLMError:
                pass

        blueprint_fix_strategy = "not-applicable"
        blueprint_manual_action_required = False
        if implementation_medium in {"blueprint", "mixed"}:
            blueprint_manual_action_required = True
            blueprint_fix_strategy = (
                "adjacent_patch_artifact"
                if state["blueprint_editable_targets"]
                else "instructions_only"
            )

        return {
            "implementation_medium": implementation_medium,
            "implementation_medium_reason": implementation_medium_reason,
            "blueprint_fix_strategy": blueprint_fix_strategy,
            "blueprint_manual_action_required": blueprint_manual_action_required,
        }

    def route_execution_track(state: EngineerState) -> str:
        if state["execution_track"] == "feature":
            return "build_design_doc"
        return "build_bug_context_doc"

    def build_design_doc(state: EngineerState) -> dict[str, Any]:
        design_doc = _compose_design_doc(
            state["task_prompt"],
            state["task_type"],
            state["implementation_medium"],
            state["implementation_medium_reason"],
            state["doc_hits"],
            state["doc_context"],
            state["source_hits"],
            state["test_hits"],
            state["code_context"],
            state["blueprint_hits"],
            state["blueprint_text_hits"],
            state["blueprint_context"],
            state["workspace_source_file"],
            state["workspace_test_file"],
        )
        if default_llm.is_enabled():
            try:
                design_doc = default_llm.generate_text(
                    instructions=(
                        "You are gameplay-engineer-workflow. Write a concise markdown design context document for a gameplay task. "
                        "Include sections: Overview, Existing References, Player-Facing Behavior, Technical Notes, Risks. "
                        "Ground the design in the provided docs when they exist."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Task type: {state['task_type']}\n"
                        f"Classification reason: {state['task_type_reason']}\n\n"
                        f"Doc hits:\n{state['doc_hits']}\n\n"
                        f"Doc context:\n{state['doc_context'] or 'No matching docs.'}\n\n"
                        f"Source hits:\n{state['source_hits']}\n\n"
                        f"Test hits:\n{state['test_hits']}\n\n"
                        f"Blueprint hits:\n{state['blueprint_hits']}\n\n"
                        f"Implementation medium: {state['implementation_medium']}\n"
                        f"Implementation medium reason: {state['implementation_medium_reason']}\n\n"
                        f"Workspace targets:\nSource={state['workspace_source_file'] or 'artifact-only'}\n"
                        f"Test={state['workspace_test_file'] or 'artifact-only'}\n\n"
                        f"Code context:\n{state['code_context'] or 'No matching source files.'}\n\n"
                        f"Blueprint context:\n{state['blueprint_context'] or 'No readable Blueprint context.'}"
                    ),
                )
            except LLMError:
                pass

        artifact_dir = Path(state["artifact_dir"])
        (artifact_dir / "design_doc.md").write_text(design_doc, encoding="utf-8")
        return {"design_doc": design_doc}

    def build_bug_context_doc(state: EngineerState) -> dict[str, Any]:
        bug_context_doc = _compose_bug_context_doc(
            task_prompt=state["task_prompt"],
            implementation_medium=state["implementation_medium"],
            implementation_medium_reason=state["implementation_medium_reason"],
            doc_hits=state["doc_hits"],
            source_hits=state["source_hits"],
            test_hits=state["test_hits"],
            blueprint_hits=state["blueprint_hits"],
            blueprint_text_hits=state["blueprint_text_hits"],
            blueprint_fix_strategy=state["blueprint_fix_strategy"],
            doc_context=state["doc_context"],
            code_context=state["code_context"],
            blueprint_context=state["blueprint_context"],
        )
        if default_llm.is_enabled():
            try:
                bug_context_doc = default_llm.generate_text(
                    instructions=(
                        "You are gameplay-engineer-workflow. Write a concise markdown bug investigation brief. "
                        "Include sections: Bug Summary, Current Signals, Likely Ownership, Fix Strategy, Validation. "
                        "Focus on the reproduction context and whether the bug belongs to code, Blueprint, or both."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Task type reason: {state['task_type_reason']}\n\n"
                        f"Implementation medium: {state['implementation_medium']}\n"
                        f"Implementation medium reason: {state['implementation_medium_reason']}\n\n"
                        f"Source hits:\n{state['source_hits']}\n\n"
                        f"Test hits:\n{state['test_hits']}\n\n"
                        f"Blueprint hits:\n{state['blueprint_hits']}\n\n"
                        f"Blueprint fix strategy: {state['blueprint_fix_strategy']}\n\n"
                        f"Doc context:\n{state['doc_context'] or 'No matching docs.'}\n\n"
                        f"Code context:\n{state['code_context'] or 'No matching source files.'}\n\n"
                        f"Blueprint context:\n{state['blueprint_context'] or 'No readable Blueprint context.'}"
                    ),
                )
            except LLMError:
                pass

        artifact_dir = Path(state["artifact_dir"])
        (artifact_dir / "bug_context.md").write_text(bug_context_doc, encoding="utf-8")
        return {"bug_context_doc": bug_context_doc}

    def plan_work(state: EngineerState) -> dict[str, Any]:
        plan_doc = _compose_initial_plan(state["task_prompt"], state["task_type"], state["doc_hits"])
        if default_llm.is_enabled():
            try:
                plan_doc = default_llm.generate_text(
                    instructions=(
                        "You are gameplay-engineer-workflow. Produce a markdown architecture and implementation plan for gameplay work. "
                        "The document must contain these exact sections: Overview, Task Type, Existing Docs, Implementation Steps, "
                        "Unit Tests, Risks, Acceptance Criteria. Treat it as the architecture approval document for new gameplay features. "
                        "Each section must have concrete bullets that are specific enough "
                        f"to pass a reviewer rubric with an approval bar of {REVIEW_APPROVAL_SCORE}/100."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Task type: {state['task_type']}\n"
                        f"Task type reason: {state['task_type_reason']}\n\n"
                        f"Design doc:\n{state['design_doc']}\n\n"
                        f"Relevant docs:\n{state['doc_context'] or 'No matching docs.'}\n\n"
                        f"Relevant source files:\n{state['source_hits']}\n\n"
                        f"Relevant test files:\n{state['test_hits']}\n\n"
                        f"Blueprint hits:\n{state['blueprint_hits']}\n\n"
                        f"Implementation medium: {state['implementation_medium']}\n"
                        f"Blueprint fix strategy: {state['blueprint_fix_strategy']}\n\n"
                        f"Workspace write mode: {state['workspace_write_summary']}\n"
                        f"Workspace targets: source={state['workspace_source_file'] or 'artifact-only'}, "
                        f"test={state['workspace_test_file'] or 'artifact-only'}\n\n"
                        f"Code context:\n{state['code_context'] or 'No matching source files.'}\n\n"
                        f"Blueprint context:\n{state['blueprint_context'] or 'No readable Blueprint context.'}"
                    ),
                )
            except LLMError:
                pass

        artifact_dir = Path(state["artifact_dir"])
        (artifact_dir / "plan_doc.md").write_text(plan_doc, encoding="utf-8")
        (artifact_dir / "architecture_plan.md").write_text(plan_doc, encoding="utf-8")
        return {"plan_doc": plan_doc, "architecture_plan_doc": plan_doc}

    def request_review(state: EngineerState) -> dict[str, Any]:
        review_round = state["review_round"] + 1
        return {"review_round": review_round}

    def enter_review_subgraph(state: EngineerState) -> dict[str, Any]:
        review_request = {
            "task_prompt": state["task_prompt"],
            "plan_doc": state["plan_doc"],
            "review_round": state["review_round"],
            "task_id": state["task_id"],
            "run_dir": state["run_dir"],
        }
        log_graph_payload_event(
            state=state,
            graph_name=graph_name,
            node_name="gameplay-reviewer-workflow",
            phase="SUBGRAPH_ENTER",
            payload_label="input",
            payload=review_request,
        )
        return {}

    def capture_review_result(state: EngineerState) -> dict[str, Any]:
        review_result = {
            "score": state["score"],
            "feedback": state["feedback"],
            "missing_sections": state["missing_sections"],
            "section_reviews": state.get("section_reviews", []),
            "blocking_issues": state.get("blocking_issues", []),
            "improvement_actions": state.get("improvement_actions", []),
            "approved": state["approved"],
            "summary": state["summary"],
        }
        log_graph_payload_event(
            state=state,
            graph_name=graph_name,
            node_name="gameplay-reviewer-workflow",
            phase="SUBGRAPH_EXIT",
            payload_label="output",
            payload=review_result,
        )
        artifact_dir = Path(state["artifact_dir"])
        section_reviews = list(state.get("section_reviews", []))
        blocking_issues = list(state.get("blocking_issues", []))
        improvement_actions = list(state.get("improvement_actions", []))
        feedback_lines = [
            f"# Review Round {state['review_round']}",
            "",
            f"- Score: {state['score']}",
            f"- Approved: {state['approved']}",
            "",
            "## Blocking Issues",
            *([f"- {item}" for item in blocking_issues] or ["- None."]),
            "",
            "## Improvement Checklist",
            *([f"- {item}" for item in improvement_actions] or ["- None."]),
            "",
            "## Section Scores",
            *(
                [
                    (
                        f"- {review['section']}: {review['score']}/{review['max_score']} "
                        f"({review['status']}) - {review['rationale']}"
                    )
                    for review in section_reviews
                ]
                or ["- Reviewer did not return per-section scores."]
            ),
            "",
            "## Full Feedback",
            state["feedback"],
        ]
        (artifact_dir / f"review_round_{state['review_round']}.md").write_text(
            "\n".join(feedback_lines),
            encoding="utf-8",
        )
        return {
            "review_score": state["score"],
            "review_feedback": state["feedback"],
            "missing_sections": state["missing_sections"],
            "review_section_reviews": section_reviews,
            "review_blocking_issues": blocking_issues,
            "review_improvement_actions": improvement_actions,
            "review_approved": state["approved"],
        }

    def review_gate(state: EngineerState) -> str:
        if not state["review_approved"]:
            if state["review_round"] >= MAX_REVIEW_ROUNDS:
                return "prepare_review_blocked_delivery"
            return "revise_plan"
        return "implement_code"

    def revise_plan(state: EngineerState) -> dict[str, Any]:
        revised_plan = _revise_plan(
            plan_doc=state["plan_doc"],
            task_prompt=state["task_prompt"],
            task_type=state["task_type"],
            doc_hits=state["doc_hits"],
            review_section_reviews=state["review_section_reviews"],
        )
        if default_llm.is_enabled():
            try:
                revised_plan = default_llm.generate_text(
                    instructions=(
                        "You are gameplay-engineer-workflow. Rewrite the full markdown implementation plan after reviewer feedback. "
                        "Keep the exact sections Overview, Task Type, Existing Docs, Implementation Steps, Unit Tests, Risks, "
                        "Acceptance Criteria, and make sure all reviewer blockers and checklist items are addressed clearly enough "
                        f"to reach the reviewer approval bar of {REVIEW_APPROVAL_SCORE}/100."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Task type: {state['task_type']}\n"
                        f"Task type reason: {state['task_type_reason']}\n\n"
                        f"Current plan:\n{state['plan_doc']}\n\n"
                        f"Relevant source files:\n{state['source_hits']}\n\n"
                        f"Relevant test files:\n{state['test_hits']}\n\n"
                        f"Workspace write mode: {state['workspace_write_summary']}\n"
                        f"Workspace targets: source={state['workspace_source_file'] or 'artifact-only'}, "
                        f"test={state['workspace_test_file'] or 'artifact-only'}\n\n"
                        f"Code context:\n{state['code_context'] or 'No matching source files.'}\n\n"
                        f"Blueprint context:\n{state['blueprint_context'] or 'No readable Blueprint context.'}\n\n"
                        f"Per-section review results:\n{state['review_section_reviews']}\n\n"
                        f"Blocking issues:\n{state['review_blocking_issues']}\n\n"
                        f"Improvement checklist:\n{state['review_improvement_actions']}\n\n"
                        f"Reviewer feedback:\n{state['review_feedback']}\n\n"
                        f"Design doc:\n{state['design_doc']}"
                    ),
                )
            except LLMError:
                pass

        artifact_dir = Path(state["artifact_dir"])
        (artifact_dir / "plan_doc.md").write_text(revised_plan, encoding="utf-8")
        (artifact_dir / "architecture_plan.md").write_text(revised_plan, encoding="utf-8")
        return {"plan_doc": revised_plan, "architecture_plan_doc": revised_plan}

    def _persist_generated_bundle(state: EngineerState, bundle: dict[str, str]) -> str:
        artifact_dir = Path(state["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "gameplay_change.py").write_text(bundle["source_code"], encoding="utf-8")
        (artifact_dir / "test_gameplay_change.py").write_text(bundle["test_code"], encoding="utf-8")

        if not state["workspace_write_enabled"]:
            summary = "Generated code was saved to workflow artifacts only."
            (artifact_dir / "workspace_write_manifest.md").write_text(summary, encoding="utf-8")
            return summary

        workspace_root = context.resolve_scope_root("agentswarm" if state["code_scope"] == "agentswarm" else "host_project")
        source_target = _write_workspace_file(workspace_root, state["workspace_source_file"], bundle["source_code"])
        test_target = _write_workspace_file(workspace_root, state["workspace_test_file"], bundle["test_code"])
        manifest_lines = [
            "# Workspace Write Manifest",
            "",
            f"- Source file: {source_target}",
            f"- Test file: {test_target}",
        ]
        (artifact_dir / "workspace_write_manifest.md").write_text("\n".join(manifest_lines), encoding="utf-8")
        return f"Generated code updated host project files: {source_target}, {test_target}."

    def _persist_blueprint_instructions(state: EngineerState, instructions_doc: str) -> str:
        artifact_dir = Path(state["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "blueprint_fix_instructions.md"
        artifact_path.write_text(instructions_doc, encoding="utf-8")

        if not state["blueprint_editable_targets"]:
            return (
                "Blueprint fix instructions were saved to workflow artifacts only. "
                "Manual Unreal Editor changes are still required because no readable Blueprint export was available."
            )

        workspace_root = context.resolve_scope_root(
            "agentswarm" if state["blueprint_scope"] == "agentswarm" else "host_project"
        )
        patch_paths: list[str] = []
        for relative_path in state["blueprint_editable_targets"]:
            target_path = Path(relative_path)
            patch_relative_path = (
                target_path.parent / f"{target_path.stem}.agentswarm_fix.md"
            ).as_posix()
            patch_paths.append(_write_workspace_file(workspace_root, patch_relative_path, instructions_doc))

        manifest_lines = [
            "# Blueprint Fix Manifest",
            "",
            f"- Artifact instructions: {artifact_path}",
            *[f"- Adjacent patch note: {item}" for item in patch_paths],
        ]
        (artifact_dir / "blueprint_fix_manifest.md").write_text("\n".join(manifest_lines), encoding="utf-8")
        return (
            "Blueprint fix instructions were saved to workflow artifacts and adjacent patch notes: "
            + ", ".join(patch_paths)
            + ". Manual Unreal Editor application is still required for the underlying `.uasset`."
        )

    def _generate_code_bundle(state: EngineerState, error_context: str = "") -> dict[str, str]:
        workspace_root = context.resolve_scope_root("agentswarm" if state["code_scope"] == "agentswarm" else "host_project")
        fallback = _mirror_existing_workspace_bundle(
            scope_root=workspace_root,
            source_relative_path=state["workspace_source_file"],
            test_relative_path=state["workspace_test_file"],
        ) or _fallback_code_bundle(state["task_prompt"], state["task_type"], state["code_attempt"] + 1)
        codegen_llm = context.get_llm("codegen")
        if not codegen_llm.is_enabled():
            return fallback

        current_source_target = _safe_read_text(workspace_root / state["workspace_source_file"], 8000).strip()
        current_test_target = _safe_read_text(workspace_root / state["workspace_test_file"], 8000).strip()

        schema = {
            "type": "object",
            "properties": {
                "source_code": {"type": "string"},
                "test_code": {"type": "string"},
                "implementation_notes": {"type": "string"},
            },
            "required": ["source_code", "test_code", "implementation_notes"],
            "additionalProperties": False,
        }
        try:
            result = codegen_llm.generate_json(
                instructions=(
                    "You are gameplay-engineer-workflow code generation. Return the full updated contents of the "
                    "target Python source file and the target Python test file. Do not use markdown fences. "
                    "If workspace targets point at existing host-project files, update those files in place and preserve "
                    "their gameplay-facing API unless the task explicitly requires an API change. Do not create a "
                    "parallel helper module, summary wrapper, or shadow implementation when a concrete host-project "
                    "target exists. The test file must be plain Python with assert statements or unittest and no "
                    "external dependencies. Ground the implementation in the provided host-project code context and "
                    "workspace targets."
                ),
                input_text=(
                    f"Task prompt:\n{state['task_prompt']}\n\n"
                    f"Execution track: {state['execution_track']}\n"
                    f"Task type: {state['task_type']}\n\n"
                    f"Implementation medium: {state['implementation_medium']}\n"
                    f"Implementation medium reason: {state['implementation_medium_reason']}\n\n"
                    f"Design doc:\n{state['design_doc']}\n\n"
                    f"Bug context:\n{state['bug_context_doc'] or 'No dedicated bug context doc.'}\n\n"
                    f"Implementation plan:\n{state['plan_doc']}\n\n"
                    f"Relevant source files:\n{state['source_hits']}\n\n"
                    f"Relevant test files:\n{state['test_hits']}\n\n"
                    f"Relevant Blueprint assets:\n{state['blueprint_hits']}\n\n"
                    f"Workspace write mode: {state['workspace_write_summary']}\n"
                    f"Workspace targets: source={state['workspace_source_file'] or 'artifact-only'}, "
                    f"test={state['workspace_test_file'] or 'artifact-only'}\n\n"
                    f"Current source target contents:\n{current_source_target or 'No existing source target contents.'}\n\n"
                    f"Current test target contents:\n{current_test_target or 'No existing test target contents.'}\n\n"
                    f"Code context:\n{state['code_context'] or 'No matching source files.'}\n\n"
                    f"Blueprint context:\n{state['blueprint_context'] or 'No readable Blueprint context.'}\n\n"
                    f"Reviewer feedback:\n{state['review_feedback']}\n\n"
                    f"Previous self-test output:\n{error_context or 'None'}"
                ),
                schema_name="gameplay_code_bundle",
                schema=schema,
            )
        except LLMError:
            return fallback

        source_code = str(result["source_code"]).strip()
        test_code = str(result["test_code"]).strip()
        if not source_code or not test_code:
            return fallback
        return {
            "source_code": source_code,
            "test_code": test_code,
            "implementation_notes": str(result["implementation_notes"]).strip() or "Codex generated the code bundle.",
        }

    def _generate_blueprint_fix_doc(state: EngineerState) -> str:
        instructions_doc = _compose_blueprint_fix_instructions(
            task_prompt=state["task_prompt"],
            implementation_medium=state["implementation_medium"],
            blueprint_hits=state["blueprint_hits"],
            blueprint_text_hits=state["blueprint_text_hits"],
            blueprint_fix_strategy=state["blueprint_fix_strategy"],
            bug_context_doc=state["bug_context_doc"],
            blueprint_context=state["blueprint_context"],
        )
        if default_llm.is_enabled():
            try:
                instructions_doc = default_llm.generate_text(
                    instructions=(
                        "You are gameplay-engineer-workflow. Write a markdown Blueprint fix handoff. "
                        "Be explicit about target assets, the likely graph or state to inspect, the intended change, "
                        "and the manual validation steps. Do not claim the `.uasset` was edited directly."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Execution track: {state['execution_track']}\n"
                        f"Implementation medium: {state['implementation_medium']}\n"
                        f"Fix strategy: {state['blueprint_fix_strategy']}\n\n"
                        f"Bug context:\n{state['bug_context_doc'] or 'No dedicated bug context doc.'}\n\n"
                        f"Blueprint hits:\n{state['blueprint_hits']}\n\n"
                        f"Readable Blueprint context:\n{state['blueprint_context'] or 'No readable Blueprint context.'}\n\n"
                        f"Code context:\n{state['code_context'] or 'No matching source files.'}"
                    ),
                )
            except LLMError:
                pass
        return instructions_doc

    def implement_code(state: EngineerState) -> dict[str, Any]:
        code_attempt = state["code_attempt"] + 1
        implementation_notes: list[str] = []
        workspace_summaries: list[str] = []
        generated_code = ""
        generated_tests = ""

        if state["implementation_medium"] != "blueprint":
            bundle = _generate_code_bundle(state)
            generated_code = bundle["source_code"]
            generated_tests = bundle["test_code"]
            implementation_notes.append(bundle["implementation_notes"])
            workspace_summaries.append(_persist_generated_bundle(state, bundle))

        if state["implementation_medium"] in {"blueprint", "mixed"}:
            blueprint_doc = _generate_blueprint_fix_doc(state)
            workspace_summaries.append(_persist_blueprint_instructions(state, blueprint_doc))
            implementation_notes.append(
                "Blueprint fix instructions were generated for manual application in the Unreal Editor."
            )

        compile_ok = False
        tests_ok = False
        test_output = ""
        if generated_code and generated_tests:
            test_output = "Awaiting source-code self-test."
        elif state["implementation_medium"] == "blueprint":
            test_output = (
                "No automated compile/test harness is available for binary Blueprint assets. "
                "Manual Unreal Editor validation is required."
            )

        return {
            "code_attempt": code_attempt,
            "generated_code": generated_code,
            "generated_tests": generated_tests,
            "implementation_notes": " ".join(note for note in implementation_notes if note).strip(),
            "workspace_write_summary": " ".join(summary for summary in workspace_summaries if summary).strip(),
            "compile_ok": compile_ok,
            "tests_ok": tests_ok,
            "test_output": test_output,
        }

    def post_implementation_gate(state: EngineerState) -> str:
        if state["generated_code"].strip() and state["generated_tests"].strip():
            return "self_test"
        return "prepare_delivery"

    def self_test(state: EngineerState) -> dict[str, Any]:
        compile_ok, tests_ok, test_output = _run_compile_and_tests(
            state["generated_code"],
            state["generated_tests"],
            source_relative_path=state["workspace_source_file"] or "gameplay_change.py",
            test_relative_path=state["workspace_test_file"] or "test_gameplay_change.py",
        )
        artifact_dir = Path(state["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "self_test.txt").write_text(test_output, encoding="utf-8")
        return {
            "compile_ok": compile_ok,
            "tests_ok": tests_ok,
            "test_output": test_output,
        }

    def test_gate(state: EngineerState) -> str:
        if state["compile_ok"] and state["tests_ok"]:
            return "prepare_delivery"
        return "repair_code"

    def repair_code(state: EngineerState) -> dict[str, Any]:
        bundle = _generate_code_bundle(state, error_context=state["test_output"])
        code_attempt = state["code_attempt"] + 1
        workspace_summaries = [_persist_generated_bundle(state, bundle)]
        implementation_notes = [bundle["implementation_notes"]]
        if state["implementation_medium"] == "mixed":
            blueprint_doc = _generate_blueprint_fix_doc(state)
            workspace_summaries.append(_persist_blueprint_instructions(state, blueprint_doc))
            implementation_notes.append(
                "Blueprint fix instructions were regenerated so the mixed implementation handoff stays aligned."
            )
        return {
            "code_attempt": code_attempt,
            "generated_code": bundle["source_code"],
            "generated_tests": bundle["test_code"],
            "implementation_notes": " ".join(note for note in implementation_notes if note).strip(),
            "workspace_write_summary": " ".join(summary for summary in workspace_summaries if summary).strip(),
        }

    def prepare_delivery(state: EngineerState) -> dict[str, Any]:
        artifact_dir = Path(state["artifact_dir"])
        commit_message = f"feat(gameplay): deliver {state['task_id']}"
        if state["task_type"] == "bugfix":
            commit_message = f"fix(gameplay): resolve {state['task_id']}"
        pr_title = f"[Gameplay] {state['task_prompt']}"
        delivery_status = "completed"
        if state["implementation_medium"] == "blueprint":
            delivery_status = "manual-validation-required"
        pr_body = "\n".join(
            [
                "# Pull Request Draft",
                "",
                f"Workflow: {metadata.name}",
                f"Task: {state['task_prompt']}",
                f"Execution track: {state['execution_track']}",
                f"Requires architecture review: {state['requires_architecture_review']}",
                f"Review score: {state['review_score']}",
                f"Review approved: {state['review_approved']}",
                f"Task type rationale: {state['task_type_reason']}",
                f"Implementation medium: {state['implementation_medium']}",
                f"Implementation medium rationale: {state['implementation_medium_reason']}",
                f"Blueprint fix strategy: {state['blueprint_fix_strategy']}",
                f"Blueprint manual action required: {state['blueprint_manual_action_required']}",
                f"Implementation notes: {state['implementation_notes']}",
                f"Workspace write mode: {state['workspace_write_summary']}",
                f"Workspace source target: {state['workspace_source_file'] or 'artifact-only'}",
                f"Workspace test target: {state['workspace_test_file'] or 'artifact-only'}",
                "",
                "## Validation",
                f"- {state['test_output']}",
            ]
        )
        (artifact_dir / "commit_message.txt").write_text(commit_message, encoding="utf-8")
        (artifact_dir / "pull_request.md").write_text(pr_body, encoding="utf-8")

        final_report = {
            "status": delivery_status,
            "task_type": state["task_type"],
            "execution_track": state["execution_track"],
            "requires_architecture_review": state["requires_architecture_review"],
            "review_rounds": state["review_round"],
            "review_score": state["review_score"],
            "review_approved": state["review_approved"],
            "compile_ok": state["compile_ok"],
            "tests_ok": state["tests_ok"],
            "implementation_medium": state["implementation_medium"],
            "implementation_medium_reason": state["implementation_medium_reason"],
            "blueprint_hits": state["blueprint_hits"],
            "blueprint_fix_strategy": state["blueprint_fix_strategy"],
            "blueprint_manual_action_required": state["blueprint_manual_action_required"],
            "commit_message": commit_message,
            "pr_title": pr_title,
            "artifact_dir": str(artifact_dir),
            "workspace_write_enabled": state["workspace_write_enabled"],
            "workspace_source_file": state["workspace_source_file"],
            "workspace_test_file": state["workspace_test_file"],
            "workspace_write_summary": state["workspace_write_summary"],
            "llm_profile": context.llm_manager.describe(metadata.llm_profile),
            "codegen_profile": context.llm_manager.describe("codegen"),
        }
        if delivery_status == "manual-validation-required":
            summary = (
                f"{metadata.name} prepared a manual Blueprint validation handoff for the {state['execution_track']} "
                f"track after {state['review_round']} review round(s) and {state['code_attempt']} implementation attempt(s)."
            )
        else:
            summary = (
                f"{metadata.name} completed the {state['execution_track']} track in "
                f"{state['review_round']} review round(s) and {state['code_attempt']} implementation attempt(s)."
            )
        return {"final_report": final_report, "summary": summary}

    def prepare_review_blocked_delivery(state: EngineerState) -> dict[str, Any]:
        artifact_dir = Path(state["artifact_dir"])
        missing_sections = list(state["missing_sections"]) or ["Reviewer did not provide structured missing sections."]
        blocking_issues = list(state["review_blocking_issues"]) or ["Reviewer did not provide structured blocking issues."]
        improvement_actions = list(state["review_improvement_actions"]) or ["Reviewer did not provide structured action items."]
        review_abort = "\n".join(
            [
                "# Review Blocked",
                "",
                f"Workflow stopped after {state['review_round']} review round(s).",
                "",
                "## Latest Feedback",
                state["review_feedback"],
                "",
                "## Blocking Issues",
                *[f"- {issue}" for issue in blocking_issues],
                "",
                "## Improvement Checklist",
                *[f"- {item}" for item in improvement_actions],
                "",
                "## Missing Sections",
                *[f"- {section}" for section in missing_sections],
            ]
        )
        (artifact_dir / "review_abort.md").write_text(review_abort, encoding="utf-8")

        final_report = {
            "status": "review-blocked",
            "task_type": state["task_type"],
            "execution_track": state["execution_track"],
            "requires_architecture_review": state["requires_architecture_review"],
            "review_rounds": state["review_round"],
            "review_score": state["review_score"],
            "review_approved": state["review_approved"],
            "compile_ok": False,
            "tests_ok": False,
            "implementation_medium": state["implementation_medium"],
            "implementation_medium_reason": state["implementation_medium_reason"],
            "blueprint_hits": state["blueprint_hits"],
            "blueprint_fix_strategy": state["blueprint_fix_strategy"],
            "blueprint_manual_action_required": state["blueprint_manual_action_required"],
            "artifact_dir": str(artifact_dir),
            "workspace_write_enabled": state["workspace_write_enabled"],
            "workspace_source_file": state["workspace_source_file"],
            "workspace_test_file": state["workspace_test_file"],
            "workspace_write_summary": state["workspace_write_summary"],
            "llm_profile": context.llm_manager.describe(metadata.llm_profile),
            "codegen_profile": context.llm_manager.describe("codegen"),
            "missing_sections": missing_sections,
            "blocking_issues": blocking_issues,
            "improvement_actions": improvement_actions,
            "review_feedback": state["review_feedback"],
        }
        summary = (
            f"{metadata.name} stopped after {state['review_round']} review round(s) because the plan never reached "
            f"approval. See review_abort.md for the latest reviewer feedback."
        )
        return {"final_report": final_report, "summary": summary}

    graph = StateGraph(EngineerState)
    graph.add_node(
        "classify_request",
        trace_graph_node(graph_name=graph_name, node_name="classify_request", node_fn=classify_request),
    )
    graph.add_node(
        "prepare_doc_search",
        trace_graph_node(graph_name=graph_name, node_name="prepare_doc_search", node_fn=prepare_doc_search),
    )
    graph.add_node(
        "tool_request_error",
        trace_graph_node(graph_name=graph_name, node_name="tool_request_error", node_fn=tool_request_error),
    )
    graph.add_node(
        "capture_doc_hits",
        trace_graph_node(graph_name=graph_name, node_name="capture_doc_hits", node_fn=capture_doc_hits),
    )
    graph.add_node(
        "prepare_doc_context_lookup",
        trace_graph_node(
            graph_name=graph_name,
            node_name="prepare_doc_context_lookup",
            node_fn=prepare_doc_context_lookup,
        ),
    )
    graph.add_node(
        "capture_doc_context",
        trace_graph_node(graph_name=graph_name, node_name="capture_doc_context", node_fn=capture_doc_context),
    )
    graph.add_node(
        "simulate_engineer_investigation",
        trace_graph_node(
            graph_name=graph_name,
            node_name="simulate_engineer_investigation",
            node_fn=simulate_engineer_investigation,
        ),
    )
    graph.add_node(
        "assess_implementation_strategy",
        trace_graph_node(
            graph_name=graph_name,
            node_name="assess_implementation_strategy",
            node_fn=assess_implementation_strategy,
        ),
    )
    graph.add_node(
        "build_design_doc",
        trace_graph_node(graph_name=graph_name, node_name="build_design_doc", node_fn=build_design_doc),
    )
    graph.add_node(
        "build_bug_context_doc",
        trace_graph_node(graph_name=graph_name, node_name="build_bug_context_doc", node_fn=build_bug_context_doc),
    )
    graph.add_node(
        "plan_work",
        trace_graph_node(graph_name=graph_name, node_name="plan_work", node_fn=plan_work),
    )
    graph.add_node(
        "request_review",
        trace_graph_node(graph_name=graph_name, node_name="request_review", node_fn=request_review),
    )
    graph.add_node(
        "enter_review_subgraph",
        trace_graph_node(graph_name=graph_name, node_name="enter_review_subgraph", node_fn=enter_review_subgraph),
    )
    graph.add_node("gameplay-reviewer-workflow", reviewer_graph)
    graph.add_node(
        "capture_review_result",
        trace_graph_node(graph_name=graph_name, node_name="capture_review_result", node_fn=capture_review_result),
    )
    graph.add_node(
        "revise_plan",
        trace_graph_node(graph_name=graph_name, node_name="revise_plan", node_fn=revise_plan),
    )
    graph.add_node(
        "implement_code",
        trace_graph_node(graph_name=graph_name, node_name="implement_code", node_fn=implement_code),
    )
    graph.add_node(
        "post_implementation_gate",
        trace_graph_node(
            graph_name=graph_name,
            node_name="post_implementation_gate",
            node_fn=lambda state: {},
        ),
    )
    graph.add_node(
        "self_test",
        trace_graph_node(graph_name=graph_name, node_name="self_test", node_fn=self_test),
    )
    graph.add_node(
        "repair_code",
        trace_graph_node(graph_name=graph_name, node_name="repair_code", node_fn=repair_code),
    )
    graph.add_node(
        "prepare_delivery",
        trace_graph_node(graph_name=graph_name, node_name="prepare_delivery", node_fn=prepare_delivery),
    )
    graph.add_node(
        "prepare_review_blocked_delivery",
        trace_graph_node(
            graph_name=graph_name,
            node_name="prepare_review_blocked_delivery",
            node_fn=prepare_review_blocked_delivery,
        ),
    )
    for tool_name, tool_subgraph in tool_subgraphs.items():
        graph.add_node(tool_node_names[tool_name], tool_subgraph)

    graph.add_edge(START, "classify_request")
    graph.add_edge("classify_request", "prepare_doc_search")
    graph.add_conditional_edges(
        "prepare_doc_search",
        trace_route_decision(graph_name=graph_name, router_name="route_tool_request", route_fn=route_tool_request),
        {
            **{
                tool_node_names[tool_name]: tool_node_names[tool_name]
                for tool_name in tool_subgraphs
            },
            "tool_request_error": "tool_request_error",
        },
    )
    graph.add_edge("tool_request_error", END)
    graph.add_edge(tool_node_names[doc_search_tool_name], "capture_doc_hits")
    graph.add_edge("capture_doc_hits", "prepare_doc_context_lookup")
    graph.add_conditional_edges(
        "prepare_doc_context_lookup",
        trace_route_decision(graph_name=graph_name, router_name="route_tool_request", route_fn=route_tool_request),
        {
            **{
                tool_node_names[tool_name]: tool_node_names[tool_name]
                for tool_name in tool_subgraphs
            },
            "tool_request_error": "tool_request_error",
        },
    )
    graph.add_edge(tool_node_names[doc_context_tool_name], "capture_doc_context")
    graph.add_edge("capture_doc_context", "simulate_engineer_investigation")
    graph.add_edge("simulate_engineer_investigation", "assess_implementation_strategy")
    graph.add_conditional_edges(
        "assess_implementation_strategy",
        trace_route_decision(graph_name=graph_name, router_name="route_execution_track", route_fn=route_execution_track),
        {
            "build_design_doc": "build_design_doc",
            "build_bug_context_doc": "build_bug_context_doc",
        },
    )
    graph.add_edge("build_design_doc", "plan_work")
    graph.add_edge("plan_work", "request_review")
    graph.add_edge("request_review", "enter_review_subgraph")
    graph.add_edge("enter_review_subgraph", "gameplay-reviewer-workflow")
    graph.add_edge("gameplay-reviewer-workflow", "capture_review_result")
    graph.add_conditional_edges(
        "capture_review_result",
        trace_route_decision(graph_name=graph_name, router_name="review_gate", route_fn=review_gate),
        {
            "revise_plan": "revise_plan",
            "prepare_review_blocked_delivery": "prepare_review_blocked_delivery",
            "implement_code": "implement_code",
        },
    )
    graph.add_edge("revise_plan", "request_review")
    graph.add_edge("build_bug_context_doc", "implement_code")
    graph.add_edge("implement_code", "post_implementation_gate")
    graph.add_conditional_edges(
        "post_implementation_gate",
        trace_route_decision(
            graph_name=graph_name,
            router_name="post_implementation_gate",
            route_fn=post_implementation_gate,
        ),
        {
            "self_test": "self_test",
            "prepare_delivery": "prepare_delivery",
        },
    )
    graph.add_conditional_edges(
        "self_test",
        trace_route_decision(graph_name=graph_name, router_name="test_gate", route_fn=test_gate),
        {
            "repair_code": "repair_code",
            "prepare_delivery": "prepare_delivery",
        },
    )
    graph.add_edge("repair_code", "self_test")
    graph.add_edge("prepare_delivery", END)
    graph.add_edge("prepare_review_blocked_delivery", END)
    return graph
