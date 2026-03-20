from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_logging import trace_graph_node, trace_route_decision
from core.llm import LLMError
from core.models import WorkflowContext, WorkflowMetadata
from core.natural_language_prompts import build_prompt_brief
from core.quality_loop import QualityLoopSpec, evaluate_quality_loop
from core.scoring import ScoreAssessment, ScorePolicy, evaluate_score_decision
from core.text_utils import keyword_tokens, normalize_text, slugify, tokenize


APPROVAL_SCORE = 90
MIN_INVESTIGATION_ROUNDS = 2
MAX_REVIEW_ROUNDS = 3
MAX_REPAIR_ROUNDS = 3
INVESTIGATION_SCORE = 85
MANDATORY_INVESTIGATION_VERIFICATION_ACTION = (
    "Run one more investigation pass that independently re-validates the live runtime owner, "
    "root-cause hypothesis, and concrete verification path before planning or implementation."
)
INVESTIGATION_LOOP_SPEC = QualityLoopSpec(
    loop_id="gameplay-investigation",
    threshold=INVESTIGATION_SCORE,
    max_rounds=3,
    min_rounds=MIN_INVESTIGATION_ROUNDS,
    require_blocker_free=True,
    require_missing_section_free=False,
    require_explicit_approval=True,
    min_score_delta=1,
    stagnation_limit=2,
)
INVESTIGATION_SCORE_POLICY = ScorePolicy(
    system_id="gameplay-investigation",
    threshold=INVESTIGATION_SCORE,
    require_blocker_free=True,
    require_missing_section_free=False,
    require_explicit_approval=True,
)
INVESTIGATION_SECTION_WEIGHTS = (
    ("Supporting References", 10),
    ("Runtime Owner Precision", 25),
    ("Current vs Legacy Split", 10),
    ("Ownership Summary", 10),
    ("Root Cause Hypothesis", 15),
    ("Investigation Summary", 10),
    ("Implementation Medium", 5),
    ("Validation Plan", 10),
    ("Noise Control", 5),
)
INVESTIGATION_SECTION_WEIGHT_MAP = dict(INVESTIGATION_SECTION_WEIGHTS)
INVESTIGATION_BLOCKING_SECTIONS = {
    "Runtime Owner Precision",
    "Current vs Legacy Split",
    "Root Cause Hypothesis",
    "Validation Plan",
}
SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".lua",
    ".py",
    ".rb",
    ".rs",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
}
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".yaml", ".yml", ".json"}
BLUEPRINT_ASSET_EXTENSION = ".uasset"
BLUEPRINT_TEXT_EXTENSIONS = {".copy", ".md", ".txt"}
GAMEPLAY_KEYWORDS = {
    "ability",
    "air",
    "battle",
    "blueprint",
    "boss",
    "charge",
    "character",
    "combat",
    "controller",
    "dash",
    "dodge",
    "enemy",
    "gameplay",
    "hit",
    "jump",
    "melee",
    "movement",
    "player",
    "recharge",
    "skill",
    "spawn",
    "state",
    "stun",
    "traversal",
    "weapon",
}
NON_GAMEPLAY_KEYWORDS = {
    "api",
    "auth",
    "backend",
    "billing",
    "ci",
    "database",
    "deploy",
    "deployment",
    "devops",
    "infra",
    "infrastructure",
    "legal",
    "marketing",
    "payment",
    "rendering",
    "sales",
    "server",
    "ui",
    "ux",
    "web",
}
PROCESS_ONLY_REVIEW_PATTERNS = (
    r"\breview round\b",
    r"\bround metadata\b",
    r"\bcurrent review context\b",
    r"\bprocess[- ]gate\b",
    r"\bindependent verif(?:ication|ier)\b",
    r"\bsign[- ]off\b",
    r"\bevidence artifact\b",
    r"\bverification artifact\b",
    r"\bartifact naming\b",
    r"\btraceability\b",
    r"\bapproval[- ]trace\b",
    r"\bmetadata\b",
    r"\bcurrent-round\b",
    r"\bround-\d+\b",
    r"\blog filenames?\b",
    r"\btargeted test command",
)


class InvestigationCheck(TypedDict):
    section: str
    score: int
    max_score: int
    status: str
    rationale: str
    action_items: list[str]


class EngineerState(TypedDict):
    prompt: NotRequired[str]
    task_prompt: str
    task_id: NotRequired[str]
    run_dir: NotRequired[str]
    artifact_dir: str
    task_type: str
    execution_track: str
    gameplay_scope_verdict: str
    classification_reason: str
    implementation_requested: bool
    requires_architecture_review: bool
    investigation_round: int
    investigation_score: int
    investigation_feedback: str
    investigation_missing_sections: list[str]
    investigation_blocking_issues: list[str]
    investigation_improvement_actions: list[str]
    investigation_approved: bool
    investigation_loop_status: str
    investigation_loop_reason: str
    investigation_loop_stagnated_rounds: int
    investigation_score_confidence: NotRequired[float | None]
    investigation_score_confidence_label: NotRequired[str]
    investigation_score_confidence_reason: NotRequired[str]
    investigation_focus_terms: list[str]
    investigation_avoid_terms: list[str]
    investigation_search_notes: list[str]
    investigation_root_cause: str
    investigation_validation_plan: str
    investigation_summary: str
    investigation_doc: str
    investigation_learning_summary: str
    investigation_learning_focus: str
    doc_hits: list[str]
    source_hits: list[str]
    test_hits: list[str]
    blueprint_hits: list[str]
    blueprint_text_hits: list[str]
    current_runtime_paths: list[str]
    legacy_runtime_paths: list[str]
    runtime_path_hypotheses: list[str]
    ownership_summary: str
    doc_context: str
    code_context: str
    blueprint_context: str
    implementation_medium: str
    implementation_medium_reason: str
    workspace_write_enabled: bool
    workspace_write_summary: str
    workspace_source_file: str
    workspace_test_file: str
    blueprint_fix_strategy: str
    blueprint_manual_action_required: bool
    bug_context_doc: str
    design_doc: str
    plan_doc: str
    review_round: int
    review_score: int
    review_feedback: str
    review_approved: bool
    review_loop_status: str
    review_loop_reason: str
    review_loop_should_continue: bool
    review_loop_completed: bool
    review_loop_stagnated_rounds: int
    review_blocking_issues: list[str]
    review_improvement_actions: list[str]
    compile_ok: bool
    tests_ok: bool
    self_test_output: str
    repair_round: int
    repair_loop_status: str
    repair_loop_reason: str
    code_attempt: int
    implementation_notes: str
    implementation_status: str
    final_report: dict[str, Any]
    summary: str
    active_loop_should_continue: bool
    active_loop_completed: bool
    active_loop_status: str


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_item in items:
        item = str(raw_item).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _short_slug(value: str, *, fallback: str, max_length: int = 18) -> str:
    slug = slugify(value, fallback=fallback)
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(normalize_text(value).encode("utf-8")).hexdigest()[:6]
    keep = max(1, max_length - len(digest) - 1)
    return f"{slug[:keep].rstrip('-') or fallback}-{digest}"


def _is_process_only_review_item(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in PROCESS_ONLY_REVIEW_PATTERNS)


def _filter_plan_revision_items(items: list[str]) -> list[str]:
    return _dedupe([item for item in items if not _is_process_only_review_item(item)])


def _artifact_dir(context: WorkflowContext, metadata: WorkflowMetadata, state: EngineerState) -> Path:
    existing = str(state.get("artifact_dir", "")).strip()
    if existing:
        path = Path(existing)
        path.mkdir(parents=True, exist_ok=True)
        return path

    run_dir = str(state.get("run_dir", "")).strip()
    base_dir = Path(run_dir) if run_dir else Path(context.artifact_root) / "adhoc"
    task_id = str(state.get("task_id", "")).strip() or state["task_prompt"]
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:6]
    task_dir = f"{_short_slug(task_id, fallback='task')}-{digest}"
    path = base_dir / "tasks" / task_dir / metadata.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_bullets(items: list[str], *, empty_message: str = "None.") -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return f"- {empty_message}"
    return "\n".join(f"- {item}" for item in cleaned)


def _contains_path_hint(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+", text))


def _should_skip(path: Path, scope_root: Path, exclude_roots: tuple[str, ...]) -> bool:
    try:
        relative = path.relative_to(scope_root).as_posix().lower()
    except ValueError:
        return True
    for excluded in exclude_roots:
        normalized = excluded.replace("\\", "/").strip("/").lower()
        if not normalized:
            continue
        if relative == normalized or relative.startswith(f"{normalized}/"):
            return True
    return False


def _safe_read_text(path: Path, *, limit: int = 700) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def _resolve_roots(scope_root: Path, relative_roots: tuple[str, ...]) -> list[Path]:
    candidates = [scope_root / relative_root for relative_root in relative_roots]
    existing = [path for path in candidates if path.exists()]
    return existing or [scope_root]


def _find_ranked_files(
    *,
    scope_root: Path,
    relative_roots: tuple[str, ...],
    exclude_roots: tuple[str, ...],
    query_text: str,
    allowed_suffixes: set[str],
    max_hits: int = 5,
) -> list[str]:
    query_tokens = keyword_tokens(query_text) or tokenize(query_text)
    scored: list[tuple[int, str]] = []
    for root in _resolve_roots(scope_root, relative_roots):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            if _should_skip(path, scope_root, exclude_roots):
                continue
            relative_path = path.relative_to(scope_root).as_posix()
            haystack = f"{relative_path.replace('_', ' ')}\n{_safe_read_text(path).replace('_', ' ')}"
            normalized_haystack = normalize_text(haystack)
            score = sum(1 for token in query_tokens if token in normalized_haystack)
            if score > 0:
                scored.append((score, relative_path))
    scored.sort(key=lambda item: (-item[0], item[1].lower()))
    hits: list[str] = []
    for _, relative_path in scored:
        if relative_path in hits:
            continue
        hits.append(relative_path)
        if len(hits) >= max_hits:
            break
    return hits


def _find_local_doc_hits(
    *,
    task_prompt: str,
    scope_root: Path,
    doc_roots: tuple[str, ...],
    exclude_roots: tuple[str, ...],
) -> list[str]:
    return _find_ranked_files(
        scope_root=scope_root,
        relative_roots=doc_roots,
        exclude_roots=exclude_roots,
        query_text=task_prompt,
        allowed_suffixes=DOC_EXTENSIONS,
    )


def _find_local_code_hits(
    *,
    task_prompt: str,
    scope_root: Path,
    source_roots: tuple[str, ...],
    test_roots: tuple[str, ...],
    exclude_roots: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    source_candidates = _find_ranked_files(
        scope_root=scope_root,
        relative_roots=source_roots,
        exclude_roots=exclude_roots,
        query_text=task_prompt,
        allowed_suffixes=SOURCE_EXTENSIONS,
        max_hits=12,
    )
    test_candidates = _find_ranked_files(
        scope_root=scope_root,
        relative_roots=test_roots,
        exclude_roots=exclude_roots,
        query_text=task_prompt,
        allowed_suffixes=SOURCE_EXTENSIONS,
        max_hits=12,
    )

    def is_test_path(relative_path: str) -> bool:
        lowered = relative_path.replace("\\", "/").lower()
        filename = Path(lowered).name
        return (
            "/tests/" in lowered
            or lowered.endswith("/tests")
            or filename.startswith("test_")
            or filename.endswith("tests.py")
            or filename.endswith("tests.cpp")
            or filename.endswith("tests.cs")
        )

    source_hits = [item for item in source_candidates if not is_test_path(item)]
    test_hits = _dedupe([*filter(is_test_path, test_candidates), *[item for item in source_candidates if is_test_path(item)]])
    return source_hits[:5], test_hits[:5]


def _find_local_blueprint_hits(
    *,
    task_prompt: str,
    scope_root: Path,
    exclude_roots: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    blueprint_asset_roots: tuple[str, ...] = ("Content",)
    blueprint_text_roots: tuple[str, ...] = ("Content", ".blueprints")
    blueprint_hits = _find_ranked_files(
        scope_root=scope_root,
        relative_roots=blueprint_asset_roots,
        exclude_roots=exclude_roots,
        query_text=task_prompt,
        allowed_suffixes={BLUEPRINT_ASSET_EXTENSION},
    )
    blueprint_text_hits = _find_ranked_files(
        scope_root=scope_root,
        relative_roots=blueprint_text_roots,
        exclude_roots=exclude_roots,
        query_text=task_prompt,
        allowed_suffixes=BLUEPRINT_TEXT_EXTENSIONS,
    )
    return (
        _filter_hits_to_roots(blueprint_hits, blueprint_asset_roots),
        _filter_hits_to_roots(blueprint_text_hits, blueprint_text_roots),
    )


def _extract_existing_relative_path(
    raw_value: str,
    *,
    scope_root: Path,
    exclude_roots: tuple[str, ...],
    allowed_suffixes: set[str],
) -> str | None:
    text = str(raw_value).strip().replace("\\", "/")
    if not text:
        return None
    candidates = [text]
    for index, char in enumerate(text):
        if char in {" ", "-", ":"}:
            candidates.append(text[:index].strip())
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip(" `\"'")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        path = scope_root / normalized
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        if _should_skip(path, scope_root, exclude_roots):
            continue
        return path.relative_to(scope_root).as_posix()
    return None


def _normalize_relative_hits(
    scope_root: Path,
    raw_hits: list[str],
    exclude_roots: tuple[str, ...],
    *,
    allowed_suffixes: set[str],
) -> list[str]:
    normalized: list[str] = []
    for raw_hit in raw_hits:
        candidate = _extract_existing_relative_path(
            str(raw_hit),
            scope_root=scope_root,
            exclude_roots=exclude_roots,
            allowed_suffixes=allowed_suffixes,
        )
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _filter_hits_to_roots(relative_hits: list[str], allowed_roots: tuple[str, ...]) -> list[str]:
    normalized_roots = [root.strip("/\\") for root in allowed_roots]
    allowed_prefixes = tuple(root + "/" for root in normalized_roots if root)
    return [item for item in relative_hits if item.startswith(allowed_prefixes)]


def _workspace_roots_exist(scope_root: Path, source_roots: tuple[str, ...]) -> bool:
    return any((scope_root / relative_root).exists() for relative_root in source_roots)


def _default_workspace_root(scope_root: Path, relative_roots: tuple[str, ...], *, fallback: str) -> Path:
    for relative_root in relative_roots:
        path = scope_root / relative_root
        if path.exists():
            return path
    return scope_root / fallback


def _resolve_workspace_targets(
    *,
    task_id: str,
    scope_root: Path,
    source_hits: list[str],
    test_hits: list[str],
    source_roots: tuple[str, ...],
    test_roots: tuple[str, ...],
    allow_workspace_writes: bool,
) -> tuple[str, str]:
    if not allow_workspace_writes:
        return "", ""
    writable_source_hits = _filter_hits_to_roots(source_hits, source_roots)
    writable_test_hits = _filter_hits_to_roots(test_hits, test_roots)
    source_target_from_hits = bool(writable_source_hits)
    if writable_source_hits:
        source_target = writable_source_hits[0]
    else:
        source_root = _default_workspace_root(scope_root, source_roots, fallback="src")
        module_name = _short_slug(task_id, fallback="gameplay_change").replace("-", "_")
        source_target = (source_root / f"{module_name}.py").relative_to(scope_root).as_posix()
    if source_target_from_hits and writable_test_hits:
        test_target = writable_test_hits[0]
    else:
        test_root = _default_workspace_root(scope_root, test_roots, fallback="tests")
        module_name = Path(source_target).stem
        test_target = (test_root / f"test_{module_name}.py").relative_to(scope_root).as_posix()
    return source_target, test_target


def _run_compile_and_tests(source_code: str, test_code: str) -> tuple[bool, bool, str]:
    with tempfile.TemporaryDirectory(prefix="agentswarm-gameplay-self-test-") as temp_dir:
        root = Path(temp_dir)
        source_path = root / "gameplay_change.py"
        alias_paths = [root / "solution.py", root / "main.py", root / "gameplay_change_summary.py"]
        test_path = root / "test_gameplay_change.py"
        source_path.write_text(source_code, encoding="utf-8")
        for alias_path in alias_paths:
            alias_path.write_text("from gameplay_change import *\n", encoding="utf-8")
        test_path.write_text(test_code, encoding="utf-8")
        import_matches = re.findall(r"(?:from|import)\s+([A-Za-z_][A-Za-z0-9_\.]*)", test_code)
        for module_name in import_matches:
            if module_name in {"__future__", "importlib", "pathlib"}:
                continue
            module_parts = module_name.split(".")
            if not module_parts or module_parts[-1] in {"Path"}:
                continue
            if module_name in {"solution", "main", "gameplay_change_summary", "gameplay_change"}:
                continue
            module_path = root.joinpath(*module_parts).with_suffix(".py")
            module_path.parent.mkdir(parents=True, exist_ok=True)
            current = module_path.parent
            while current != root:
                init_path = current / "__init__.py"
                if not init_path.exists():
                    init_path.write_text("", encoding="utf-8")
                current = current.parent
            module_path.write_text(source_code, encoding="utf-8")

        compile_cmd = [
            sys.executable,
            "-m",
            "py_compile",
            str(source_path),
            str(test_path),
            *[str(alias_path) for alias_path in alias_paths],
            *[str(path) for path in root.rglob("*.py") if path not in {source_path, test_path, *alias_paths}],
        ]
        compile_completed = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            cwd=root,
            check=False,
            timeout=30,
        )
        compile_output = (compile_completed.stdout + compile_completed.stderr).strip()
        compile_ok = compile_completed.returncode == 0
        if not compile_ok:
            return False, False, compile_output or "Compilation failed."

        runner = (
            "import importlib.util, inspect, json, sys\n"
            "from pathlib import Path\n"
            "root = Path(sys.argv[1])\n"
            "sys.path.insert(0, str(root))\n"
            "spec = importlib.util.spec_from_file_location('test_gameplay_change', root / 'test_gameplay_change.py')\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "results = []\n"
            "for name, fn in sorted(vars(module).items()):\n"
            "    if name.startswith('test_') and inspect.isfunction(fn):\n"
            "        fn()\n"
            "        results.append(name)\n"
            "print(json.dumps({'tests': results}))\n"
        )
        test_completed = subprocess.run(
            [sys.executable, "-c", runner, str(root)],
            capture_output=True,
            text=True,
            cwd=root,
            check=False,
            timeout=30,
        )
        test_output = (test_completed.stdout + test_completed.stderr).strip()
        return compile_ok, test_completed.returncode == 0, test_output


def _is_read_only_request(task_prompt: str) -> bool:
    lowered = normalize_text(task_prompt)
    markers = (
        "do not modify",
        "without changing files",
        "read-only",
        "investigate",
        "research and plan",
        "summarize",
        "design only",
        "before implementation",
    )
    return any(marker in lowered for marker in markers)


def _classify_task(context: WorkflowContext, task_prompt: str) -> tuple[str, str]:
    gameplay_hits = len(tokenize(task_prompt) & GAMEPLAY_KEYWORDS)
    non_gameplay_hits = len(tokenize(task_prompt) & NON_GAMEPLAY_KEYWORDS)
    fallback_task_type = "maintenance"
    lowered = normalize_text(task_prompt)
    if any(marker in lowered for marker in ("fix", "bug", "regression", "broken")):
        fallback_task_type = "bugfix"
    elif any(marker in lowered for marker in ("feature", "add ", "new ", "implement")):
        fallback_task_type = "feature"
    elif any(marker in lowered for marker in ("maintain", "maintenance", "refactor", "harden", "cleanup", "stabil")):
        fallback_task_type = "maintenance"

    if non_gameplay_hits > gameplay_hits and gameplay_hits == 0:
        fallback_task_type = "non_gameplay"
        fallback_reason = "The prompt does not appear to be gameplay-owned work."
    else:
        fallback_reason = f"The prompt is best handled as gameplay `{fallback_task_type}` work."

    if context.llm.is_enabled():
        schema = {
            "type": "object",
            "properties": {
                "task_type": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["task_type", "reason"],
            "additionalProperties": False,
        }
        try:
            result = context.llm.generate_json(
                instructions=(
                    "Classify a gameplay-engineering request. Return one of: bugfix, feature, maintenance, or non_gameplay. "
                    "Use non_gameplay only when the task is clearly outside gameplay ownership."
                ),
                input_text=build_prompt_brief(
                    opening="Decide how this gameplay-engineering request should be classified.",
                    sections=[("Request", task_prompt.strip())],
                    closing="Choose the track that best matches the real gameplay ownership and user intent.",
                ),
                schema_name="gameplay_task_classification",
                schema=schema,
            )
            task_type = str(result.get("task_type", fallback_task_type)).strip().lower()
            if task_type not in {"bugfix", "feature", "maintenance", "non_gameplay"}:
                task_type = fallback_task_type
            reason = str(result.get("reason", fallback_reason)).strip() or fallback_reason
            return task_type, reason
        except Exception:
            return fallback_task_type, fallback_reason
    return fallback_task_type, fallback_reason


def _prepare_investigation_strategy_payload(
    context: WorkflowContext,
    state: EngineerState,
) -> dict[str, Any]:
    task_prompt = state["task_prompt"]
    focus_terms = sorted(keyword_tokens(task_prompt))[:6]
    avoid_terms = ["archive", "legacy"] if "legacy" in normalize_text(task_prompt) else []
    search_notes = ["Identify the live gameplay runtime owner before implementation."]
    implementation_medium_hint = "cpp"
    implementation_medium_reason = "Gameplay ownership usually lands in code when a live runtime path is present."

    if context.llm.is_enabled():
        schema = {
            "type": "object",
            "properties": {
                "focus_terms": {"type": "array", "items": {"type": "string"}},
                "avoid_terms": {"type": "array", "items": {"type": "string"}},
                "search_notes": {"type": "array", "items": {"type": "string"}},
                "implementation_medium_hint": {"type": "string"},
                "implementation_medium_reason": {"type": "string"},
                "investigation_root_cause": {"type": "string"},
                "investigation_validation_plan": {"type": "string"},
            },
            "required": [
                "focus_terms",
                "avoid_terms",
                "search_notes",
                "implementation_medium_hint",
                "implementation_medium_reason",
                "investigation_root_cause",
                "investigation_validation_plan",
            ],
            "additionalProperties": False,
        }
        previous_learning = "\n".join(
            [
                "## Previous Learning Summary",
                state.get("investigation_learning_summary", "") or "None.",
                "",
                "## Previous Learning Focus",
                state.get("investigation_learning_focus", "") or "None.",
                "",
                "## Previous Retained Evidence",
                _format_bullets(list(state.get("current_runtime_paths", []))),
                "",
                "## Previous Open Questions",
                _format_bullets(list(state.get("investigation_improvement_actions", []))),
                "",
                "## Previous Rejected Evidence",
                _format_bullets(list(state.get("legacy_runtime_paths", []))),
            ]
        )
        try:
            result = context.llm.generate_json(
                instructions=(
                    "Plan the next gameplay investigation round. Propose the next focus terms, avoid terms, search notes, and the current "
                    "implementation-medium hypothesis."
                ),
                input_text=build_prompt_brief(
                    opening="Plan the next grounded gameplay investigation pass.",
                    sections=[
                        ("Task request", task_prompt.strip()),
                        (
                            "Current round",
                            (
                                f"This is investigation round {int(state.get('investigation_round', 0)) + 1} "
                                f"for a {state.get('task_type', 'bugfix')} task."
                            ),
                        ),
                        ("What survived from the previous pass", previous_learning),
                    ],
                    closing=(
                        "Tighten the next pass around the live runtime owner, the clearest causal hypothesis, "
                        "and the most concrete validation path."
                    ),
                ),
                schema_name="gameplay_investigation_strategy",
                schema=schema,
            )
            return {
                "focus_terms": _dedupe([str(item) for item in result.get("focus_terms", []) if str(item).strip()]) or focus_terms,
                "avoid_terms": _dedupe([str(item) for item in result.get("avoid_terms", []) if str(item).strip()]),
                "search_notes": _dedupe([str(item) for item in result.get("search_notes", []) if str(item).strip()]) or search_notes,
                "implementation_medium_hint": str(result.get("implementation_medium_hint", implementation_medium_hint)).strip() or implementation_medium_hint,
                "implementation_medium_reason": str(result.get("implementation_medium_reason", implementation_medium_reason)).strip() or implementation_medium_reason,
                "investigation_root_cause": str(result.get("investigation_root_cause", "")).strip(),
                "investigation_validation_plan": str(result.get("investigation_validation_plan", "")).strip(),
            }
        except Exception:
            pass
    return {
        "focus_terms": focus_terms,
        "avoid_terms": avoid_terms,
        "search_notes": search_notes,
        "implementation_medium_hint": implementation_medium_hint,
        "implementation_medium_reason": implementation_medium_reason,
        "investigation_root_cause": "",
        "investigation_validation_plan": "",
    }


def _collect_engineering_context(context: WorkflowContext, state: EngineerState) -> dict[str, Any]:
    scope_root = context.resolve_scope_root("host_project")
    task_prompt = "\n".join([state["task_prompt"], *state.get("investigation_focus_terms", [])])
    local_doc_hits = _find_local_doc_hits(
        task_prompt=task_prompt,
        scope_root=scope_root,
        doc_roots=context.config.doc_roots,
        exclude_roots=context.config.exclude_roots,
    )
    local_source_hits, local_test_hits = _find_local_code_hits(
        task_prompt=task_prompt,
        scope_root=scope_root,
        source_roots=context.config.source_roots,
        test_roots=context.config.test_roots,
        exclude_roots=context.config.exclude_roots,
    )
    local_blueprint_hits, local_blueprint_text_hits = _find_local_blueprint_hits(
        task_prompt=task_prompt,
        scope_root=scope_root,
        exclude_roots=context.config.exclude_roots,
    )
    payload = {
        "doc_hits": list(local_doc_hits),
        "doc_context": "",
        "source_hits": list(local_source_hits),
        "test_hits": list(local_test_hits),
        "blueprint_hits": list(local_blueprint_hits),
        "blueprint_text_hits": list(local_blueprint_text_hits),
        "current_runtime_paths": list(local_source_hits[:1]),
        "legacy_runtime_paths": [],
        "runtime_path_hypotheses": [],
        "ownership_summary": "",
        "investigation_summary": "",
        "code_context": "",
        "blueprint_context": "",
        "implementation_medium": "",
        "implementation_medium_reason": str(state.get("implementation_medium_reason", "")),
    }
    llm_source_hits_raw: list[str] = []
    llm_current_runtime_paths_raw: list[str] = []
    llm_legacy_runtime_paths_raw: list[str] = []
    if context.llm.is_enabled():
        schema = {
            "type": "object",
            "properties": {
                "doc_hits": {"type": "array", "items": {"type": "string"}},
                "doc_context": {"type": "string"},
                "source_hits": {"type": "array", "items": {"type": "string"}},
                "test_hits": {"type": "array", "items": {"type": "string"}},
                "blueprint_hits": {"type": "array", "items": {"type": "string"}},
                "blueprint_text_hits": {"type": "array", "items": {"type": "string"}},
                "current_runtime_paths": {"type": "array", "items": {"type": "string"}},
                "legacy_runtime_paths": {"type": "array", "items": {"type": "string"}},
                "runtime_path_hypotheses": {"type": "array", "items": {"type": "string"}},
                "ownership_summary": {"type": "string"},
                "investigation_summary": {"type": "string"},
                "code_context": {"type": "string"},
                "blueprint_context": {"type": "string"},
                "implementation_medium": {"type": "string"},
                "implementation_medium_reason": {"type": "string"},
            },
            "required": [
                "doc_hits",
                "doc_context",
                "source_hits",
                "test_hits",
                "blueprint_hits",
                "blueprint_text_hits",
                "current_runtime_paths",
                "legacy_runtime_paths",
                "runtime_path_hypotheses",
                "ownership_summary",
                "investigation_summary",
                "code_context",
                "blueprint_context",
                "implementation_medium",
                "implementation_medium_reason",
            ],
            "additionalProperties": False,
        }
        try:
            generated = context.llm.generate_json(
                instructions=(
                    "Inspect the grounded gameplay context and return the best current ownership summary, runtime paths, test paths, and "
                    "implementation-medium hypothesis as JSON. Prefer current runtime ownership over legacy notes."
                ),
                input_text=build_prompt_brief(
                    opening="Review the grounded gameplay evidence and identify the live owner of the work.",
                    sections=[
                        ("Task request", state["task_prompt"].strip()),
                        (
                            "Current investigation focus",
                            _format_bullets(list(state.get("investigation_focus_terms", []))),
                        ),
                        (
                            "Evidence to ignore or de-prioritize",
                            _format_bullets(list(state.get("investigation_avoid_terms", []))),
                        ),
                        ("Suggested design references", _format_bullets(local_doc_hits)),
                        ("Suggested runtime code paths", _format_bullets(local_source_hits)),
                        ("Suggested validation paths", _format_bullets(local_test_hits)),
                        ("Suggested Blueprint assets", _format_bullets(local_blueprint_hits)),
                        (
                            "Suggested Blueprint text mirrors",
                            _format_bullets(local_blueprint_text_hits),
                        ),
                    ],
                    closing=(
                        "Prefer the current runtime owner over archival or speculative references, "
                        "and keep the ownership story tight enough for implementation."
                    ),
                ),
                schema_name="gameplay_engineering_context",
                schema=schema,
            )
            llm_source_hits_raw = [str(item) for item in generated.get("source_hits", []) if str(item).strip()]
            llm_current_runtime_paths_raw = [
                str(item) for item in generated.get("current_runtime_paths", []) if str(item).strip()
            ]
            llm_legacy_runtime_paths_raw = [str(item) for item in generated.get("legacy_runtime_paths", []) if str(item).strip()]
            payload.update(generated)
        except Exception:
            pass

    payload["doc_hits"] = _normalize_relative_hits(
        scope_root,
        list(payload.get("doc_hits", [])),
        context.config.exclude_roots,
        allowed_suffixes=DOC_EXTENSIONS,
    ) or local_doc_hits
    payload["source_hits"] = _normalize_relative_hits(
        scope_root,
        list(payload.get("source_hits", [])),
        context.config.exclude_roots,
        allowed_suffixes=SOURCE_EXTENSIONS,
    ) or local_source_hits
    payload["test_hits"] = _normalize_relative_hits(
        scope_root,
        list(payload.get("test_hits", [])),
        context.config.exclude_roots,
        allowed_suffixes=SOURCE_EXTENSIONS,
    ) or local_test_hits
    payload["blueprint_hits"] = _filter_hits_to_roots(
        _normalize_relative_hits(
            scope_root,
            list(payload.get("blueprint_hits", [])),
            context.config.exclude_roots,
            allowed_suffixes={BLUEPRINT_ASSET_EXTENSION},
        ),
        ("Content",),
    ) or local_blueprint_hits
    payload["blueprint_text_hits"] = _filter_hits_to_roots(
        _normalize_relative_hits(
            scope_root,
            list(payload.get("blueprint_text_hits", [])),
            context.config.exclude_roots,
            allowed_suffixes=BLUEPRINT_TEXT_EXTENSIONS,
        ),
        ("Content", ".blueprints"),
    ) or local_blueprint_text_hits
    current_runtime_paths = _normalize_relative_hits(
        scope_root,
        list(payload.get("current_runtime_paths", [])),
        context.config.exclude_roots,
        allowed_suffixes=SOURCE_EXTENSIONS | {BLUEPRINT_ASSET_EXTENSION} | BLUEPRINT_TEXT_EXTENSIONS | DOC_EXTENSIONS,
    )
    if current_runtime_paths:
        payload["current_runtime_paths"] = current_runtime_paths
    elif llm_source_hits_raw:
        payload["current_runtime_paths"] = payload["source_hits"][:1]
    elif llm_current_runtime_paths_raw:
        payload["current_runtime_paths"] = []
    elif llm_legacy_runtime_paths_raw:
        payload["current_runtime_paths"] = []
    else:
        payload["current_runtime_paths"] = payload["source_hits"][:1] or payload["blueprint_hits"][:1] or payload["blueprint_text_hits"][:1]
    payload["legacy_runtime_paths"] = _normalize_relative_hits(
        scope_root,
        list(payload.get("legacy_runtime_paths", [])),
        context.config.exclude_roots,
        allowed_suffixes=SOURCE_EXTENSIONS | DOC_EXTENSIONS | BLUEPRINT_TEXT_EXTENSIONS,
    )
    payload["runtime_path_hypotheses"] = _dedupe(
        [str(item) for item in payload.get("runtime_path_hypotheses", []) if str(item).strip()]
    )
    if not str(payload.get("ownership_summary", "")).strip():
        owner = (
            payload["current_runtime_paths"][:1]
            or payload["source_hits"][:1]
            or payload["blueprint_hits"][:1]
            or payload["blueprint_text_hits"][:1]
        )
        validation = payload["test_hits"][:1] or payload["blueprint_text_hits"][:1] or payload["doc_hits"][:1]
        if owner:
            owner_note = f"Current gameplay ownership is anchored on {owner[0]}."
            validation_note = (
                f"Supporting validation or reference context is grounded in {validation[0]}."
                if validation
                else "Supporting validation still needs a stronger explicit anchor."
            )
            payload["ownership_summary"] = f"{owner_note} {validation_note}".strip()
    return payload


def _choose_implementation_medium(state: EngineerState, payload: dict[str, Any]) -> tuple[str, str]:
    hint = str(payload.get("implementation_medium") or state.get("implementation_medium", "") or "").strip().lower()
    if payload["source_hits"] or payload["current_runtime_paths"] and any(
        not item.endswith(BLUEPRINT_ASSET_EXTENSION) and not Path(item).suffix.lower() in BLUEPRINT_TEXT_EXTENSIONS
        for item in payload["current_runtime_paths"]
    ):
        return "cpp", str(payload.get("implementation_medium_reason") or state.get("implementation_medium_reason") or "")
    if payload["blueprint_hits"] or payload["blueprint_text_hits"]:
        return "blueprint", str(payload.get("implementation_medium_reason") or state.get("implementation_medium_reason") or "")
    if hint in {"blueprint", "bp"}:
        return "blueprint", str(payload.get("implementation_medium_reason") or state.get("implementation_medium_reason") or "")
    return "cpp", str(payload.get("implementation_medium_reason") or state.get("implementation_medium_reason") or "")


def _build_investigation_learning(state: EngineerState, payload: dict[str, Any]) -> tuple[str, str, str]:
    retained = payload["current_runtime_paths"] or payload["source_hits"] or payload["blueprint_hits"]
    rejected = payload["legacy_runtime_paths"]
    summary = (
        str(payload.get("investigation_summary", "")).strip()
        or "The round narrowed the gameplay owner and validation path."
    )
    if not payload["current_runtime_paths"]:
        reminder = "The investigation still needs a stronger runtime owner before implementation or handoff."
        if reminder.lower() not in summary.lower():
            summary = f"{summary.rstrip('.')} {reminder}".strip()
    if state.get("investigation_improvement_actions") or int(state.get("investigation_round", 1)) < MIN_INVESTIGATION_ROUNDS:
        next_pass_reminder = "This still needs another pass before implementation or handoff."
        if next_pass_reminder.lower() not in summary.lower():
            summary = f"{summary.rstrip('.')} {next_pass_reminder}".strip()
    focus = (
        f"Keep investigating {retained[0]}."
        if retained
        else "Keep searching for the live gameplay runtime owner."
    )
    learning_doc = "\n".join(
        [
            "# Investigation Learning",
            "",
            "## Summary",
            summary,
            "",
            "## Focus",
            focus,
            "",
            "## Retained Evidence",
            _format_bullets(list(retained)),
            "",
            "## Rejected Evidence",
            _format_bullets(list(rejected)),
            "",
            "## Open Questions",
            _format_bullets(list(state.get("investigation_improvement_actions", []))),
        ]
    )
    return summary, focus, learning_doc


def _compose_investigation_doc(state: EngineerState, payload: dict[str, Any], *, round_index: int) -> str:
    return "\n".join(
        [
            "# Gameplay Engineer Investigation",
            "",
            f"- Task Type: {state['task_type']}",
            f"- Investigation Round: {round_index}",
            f"- Implementation Medium: {state['implementation_medium']}",
            f"- Implementation Medium Reason: {state['implementation_medium_reason'] or 'None.'}",
            "",
            "## Focus Terms",
            _format_bullets(list(state.get("investigation_focus_terms", []))),
            "",
            "## Search Notes",
            _format_bullets(list(state.get("investigation_search_notes", []))),
            "",
            "## Doc Hits",
            _format_bullets(list(payload.get("doc_hits", []))),
            "",
            "## Source Hits",
            _format_bullets(list(payload.get("source_hits", []))),
            "",
            "## Test Hits",
            _format_bullets(list(payload.get("test_hits", []))),
            "",
            "## Blueprint Hits",
            _format_bullets(list(payload.get("blueprint_hits", []))),
            "",
            "## Blueprint Text Hits",
            _format_bullets(list(payload.get("blueprint_text_hits", []))),
            "",
            "## Current Runtime Paths",
            _format_bullets(list(payload.get("current_runtime_paths", []))),
            "",
            "## Legacy Runtime Paths",
            _format_bullets(list(payload.get("legacy_runtime_paths", []))),
            "",
            "## Runtime Path Hypotheses",
            _format_bullets(list(payload.get("runtime_path_hypotheses", []))),
            "",
            "## Ownership Summary",
            str(payload.get("ownership_summary", "")).strip() or "Ownership is still being established.",
            "",
            "## Investigation Summary",
            str(payload.get("investigation_summary", "")).strip() or "Investigation is still narrowing the likely owner.",
            "",
            "## Root Cause Direction",
            str(state.get("investigation_root_cause", "")).strip() or "No concrete root-cause statement yet.",
            "",
            "## Validation Plan",
            str(state.get("investigation_validation_plan", "")).strip() or "No explicit validation plan yet.",
            "",
            "## Code Context",
            str(payload.get("code_context", "")).strip() or "No code context yet.",
            "",
            "## Blueprint Context",
            str(payload.get("blueprint_context", "")).strip() or "No Blueprint context yet.",
        ]
    )


def _compose_bug_context_doc(context: WorkflowContext, state: EngineerState) -> str:
    blueprint_notes = [
        item
        for item in [*list(state.get("blueprint_hits", [])), *list(state.get("blueprint_text_hits", []))]
        if item.endswith(BLUEPRINT_ASSET_EXTENSION)
        or Path(item).suffix.lower() in BLUEPRINT_TEXT_EXTENSIONS
        and (item.startswith("Content/") or item.startswith(".blueprints/"))
    ]
    metadata_block = "\n".join(
        [
            "",
            f"Implementation Medium: {state['implementation_medium']}",
            f"Blueprint Fix Strategy: {state['blueprint_fix_strategy']}",
            "",
            "## Blueprint Notes",
            _format_bullets(
                blueprint_notes,
                empty_message="No Blueprint assets were matched.",
            ),
        ]
    )
    fallback = "\n".join(
        [
            "# Gameplay Bug Context",
            "",
            "## Bug Summary",
            f"- Request: {state['task_prompt']}",
            f"- Ownership: {state['ownership_summary'] or 'Still narrowing the owner.'}",
            "",
            "## Current Signals",
            _format_bullets(list(state.get("current_runtime_paths", [])), empty_message="No current runtime path confirmed yet."),
            metadata_block,
        ]
    )
    llm = context.llm
    if llm.is_enabled():
        try:
            generated = llm.generate_text(
                instructions=(
                    "Write a concise markdown bug investigation brief with the sections Bug Summary and Current Signals. "
                    "Keep it grounded in gameplay ownership and validation paths."
                ),
                input_text=build_prompt_brief(
                    opening="Prepare the gameplay bug context that the delivery workflow will carry forward.",
                    sections=[
                        ("Task request", state["task_prompt"].strip()),
                        (
                            "Grounded owner and current signals",
                            state["ownership_summary"].strip() or "Ownership is still being narrowed.",
                        ),
                        (
                            "Confirmed runtime paths",
                            _format_bullets(list(state.get("current_runtime_paths", []))),
                        ),
                        (
                            "Confirmed validation paths",
                            _format_bullets(list(state.get("test_hits", []))),
                        ),
                    ],
                    closing="Stay narrow, grounded, and ready for an implementer to act on without re-opening broad investigation.",
                ),
            )
            return f"{generated.rstrip()}\n{metadata_block}"
        except Exception:
            return fallback
    return fallback


def _compose_design_doc(context: WorkflowContext, state: EngineerState) -> str:
    fallback = "\n".join(
        [
            "# Gameplay Design Context",
            "",
            "## Overview",
            f"- Request: {state['task_prompt']}",
            "- Keep the change scoped to gameplay-only ownership and preserve adjacent gameplay states.",
            "",
            "## Existing References",
            _format_bullets(list(state.get("doc_hits", [])), empty_message="No strong design references were found."),
            "",
            "## Player-Facing Behavior",
            "- Describe the expected gameplay behavior and the neighboring path that must remain stable.",
            "",
            "## Technical Notes",
            _format_bullets(list(state.get("current_runtime_paths", [])), empty_message="No runtime owner has been grounded yet."),
            "",
            "## Risks",
            "- Risk: adjacent gameplay states could drift if the change is implemented too broadly.",
        ]
    )
    if context.llm.is_enabled():
        try:
            return context.llm.generate_text(
                instructions=(
                    "Write a concise markdown design context document with these exact sections: Overview, Existing References, "
                    "Player-Facing Behavior, Technical Notes, Risks."
                ),
                input_text=build_prompt_brief(
                    opening="Prepare the gameplay design context that will ground planning and review.",
                    sections=[
                        ("Task request", state["task_prompt"].strip()),
                        (
                            "Grounded references",
                            _format_bullets(list(state.get("doc_hits", []))),
                        ),
                        (
                            "Grounded runtime ownership",
                            _format_bullets(list(state.get("current_runtime_paths", []))),
                        ),
                    ],
                    closing="Keep the design context concrete, scoped to gameplay ownership, and explicit about nearby behavior that must remain stable.",
                ),
            )
        except Exception:
            return fallback
    return fallback


def _compose_plan_doc(context: WorkflowContext, state: EngineerState, *, revise: bool) -> str:
    task_type_reason = state["classification_reason"] or f"This work is classified as {state['task_type']}."
    filtered_review_blocking_issues = _filter_plan_revision_items(list(state.get("review_blocking_issues", [])))
    filtered_review_improvement_actions = _filter_plan_revision_items(list(state.get("review_improvement_actions", [])))
    fallback = "\n".join(
        [
            "# Gameplay Implementation Plan",
            "",
            "## Overview",
            f"- {state['task_prompt']}",
            "- Keep nearby gameplay states stable while delivering the requested behavior.",
            "",
            "## Task Type",
            f"- {state['task_type']}",
            f"- Classification reason: {task_type_reason}",
            "",
            "## Existing Docs",
            _format_bullets(list(state.get("doc_hits", [])), empty_message="No grounded docs found."),
            "",
            "## Implementation Steps",
            _format_bullets(
                [
                    f"Anchor the change on {state['current_runtime_paths'][0]}."
                    if state.get("current_runtime_paths")
                    else "Confirm the owning gameplay runtime path before coding.",
                    "Keep the change narrow and protect the neighboring gameplay transition.",
                    "Preserve existing validation hooks and logs where possible.",
                ]
            ),
            "",
            "## Unit Tests",
            _format_bullets(
                list(state.get("test_hits", []))
                or ["Add or update automated regression coverage for the owning gameplay path."]
            ),
            "",
            "## Risks",
            "- Risk: adjacent gameplay states could regress if the hook is too broad.",
            "- Mitigation: constrain the change to the grounded owner and cover the neighbor path in tests.",
            "",
            "## Acceptance Criteria",
            "- The requested gameplay outcome is visible to the player.",
            "- Neighboring gameplay paths remain unchanged and automated checks still pass.",
        ]
    )
    instruction = (
        "Rewrite the full markdown implementation plan"
        if revise
        else (
            "Produce a markdown architecture and implementation plan"
            if state["requires_architecture_review"]
            else "Produce a markdown implementation plan"
        )
    )
    if context.llm.is_enabled():
        try:
            return context.llm.generate_text(
                instructions=(
                    f"{instruction} using these exact sections: Overview, Task Type, Existing Docs, Implementation Steps, "
                    "Unit Tests, Risks, Acceptance Criteria."
                ),
                input_text=build_prompt_brief(
                    opening="Draft the next gameplay implementation plan for this workflow.",
                    sections=[
                        ("Task request", state["task_prompt"].strip()),
                        (
                            "Current task framing",
                            "\n".join(
                                [
                                    f"- Task type: {state['task_type']}",
                                    f"- Architecture review required: {state['requires_architecture_review']}",
                                ]
                            ),
                        ),
                        (
                            "Open blocking issues",
                            _format_bullets(filtered_review_blocking_issues),
                        ),
                        (
                            "Requested improvements",
                            _format_bullets(filtered_review_improvement_actions),
                        ),
                        ("Design context", state.get("design_doc", "").strip() or "None."),
                        ("Bug context", state.get("bug_context_doc", "").strip() or "None."),
                    ],
                    closing=(
                        "Produce a plan that is implementation-ready, anchored on the current gameplay owner, "
                        "and explicit about regression coverage for adjacent behavior. Keep the plan technical and "
                        "do not add review-round bookkeeping, sign-off workflow, or artifact naming requirements."
                    ),
                ),
            )
        except Exception:
            return fallback
    return fallback


def _fallback_code_bundle(state: EngineerState, scope_root: Path) -> dict[str, str]:
    source_path = scope_root / state["workspace_source_file"] if state.get("workspace_source_file") else None
    test_path = scope_root / state["workspace_test_file"] if state.get("workspace_test_file") else None
    if source_path and source_path.exists():
        source_code = source_path.read_text(encoding="utf-8")
    else:
        source_code = "\n".join(
            [
                "from pathlib import Path",
                "",
                "def build_gameplay_change_summary():",
                "    return {",
                f'        "task_type": "{state["task_type"]}",',
                '        "implementation_status": "ready-for-review",',
                '        "unit_tests": ["smoke"],',
                '        "source_file": Path(__file__).name,',
                "    }",
            ]
        )
    if test_path and test_path.exists():
        test_code = test_path.read_text(encoding="utf-8")
    else:
        test_code = "\n".join(
            [
                "import importlib",
                "",
                "def test_gameplay_change_smoke():",
                "    for name in ['solution', 'main', 'gameplay_change_summary']:",
                "        module = importlib.import_module(name)",
                "        if hasattr(module, 'build_gameplay_change_summary'):",
                "            result = module.build_gameplay_change_summary()",
                "            assert result['implementation_status'] == 'ready-for-review'",
                "            return",
                "    raise AssertionError('builder not found')",
            ]
        )
    return {
        "source_code": source_code,
        "test_code": test_code,
        "implementation_notes": "Used the grounded workspace files as the fallback gameplay code bundle.",
    }


def _compose_blueprint_handoff(
    context: WorkflowContext,
    state: EngineerState,
    *,
    scope_root: Path,
    blueprint_target: str,
) -> str:
    blueprint_text_target = next(
        (
            item
            for item in [*list(state.get("blueprint_text_hits", [])), *list(state.get("blueprint_hits", []))]
            if str(item).strip()
        ),
        "",
    )
    blueprint_text = _safe_read_text(scope_root / blueprint_text_target, limit=2000) if blueprint_text_target else ""
    fallback = "\n".join(
        [
            "# Blueprint Fix Instructions",
            "",
            "## Goal",
            f"- {state['task_prompt']}",
            "- Keep the Blueprint-only change narrow and protect adjacent gameplay states.",
            "",
            "## Scope",
            f"- Target asset: {blueprint_target or 'No concrete Blueprint target found.'}",
            f"- Text mirror: {blueprint_text_target or 'None.'}",
            "",
            "## Safe Patch Steps",
            "1. Create a backup copy of the target Blueprint in Unreal Editor before editing.",
            "2. Open the narrowest EventGraph or helper path that owns the failing gameplay transition described below.",
            "3. Adjust the transition so the requested gameplay behavior stays correct without broadening ownership into adjacent states.",
            "4. Preserve existing gates for nearby states, interrupts, and cleanup paths unless the bug context proves they are wrong.",
            "5. After the manual edit, update the text mirror so it matches the final in-editor behavior.",
            "",
            "## Verification Checklist",
            "- Reproduce the original bug and confirm the requested gameplay behavior now works.",
            "- Re-test the adjacent gameplay path that must remain unchanged.",
            "- Capture a short validation note or screenshot proving the final Blueprint wiring.",
            "",
            "## Notes For Text Mirror",
            "- Update the `.copy` or exported Blueprint text after the manual patch.",
            "- Keep the text mirror aligned with the final EventGraph so future investigations stay grounded.",
            "",
            "## Grounded Context",
            state.get("bug_context_doc", "").strip() or "No additional bug context was available.",
            "",
            "## Current Blueprint Text Mirror",
            blueprint_text or "No readable Blueprint text mirror was available in the workspace.",
        ]
    )
    handoff_llm = context.get_llm("codegen")
    if handoff_llm.is_enabled():
        try:
            return handoff_llm.generate_text(
                instructions=(
                    "Write a precise markdown manual Blueprint handoff using these exact sections: Goal, Scope, Safe Patch Steps, "
                    "Verification Checklist, Notes For Text Mirror, Grounded Context, Current Blueprint Text Mirror. "
                    "Do not claim the binary asset was edited. Keep the steps asset-specific, safe, and grounded in the provided evidence."
                ),
                input_text=build_prompt_brief(
                    opening="Prepare the manual Blueprint handoff for a gameplay-only fix.",
                    sections=[
                        ("Task request", state["task_prompt"].strip()),
                        ("Blueprint target", blueprint_target or "None."),
                        ("Blueprint text mirror target", blueprint_text_target or "None."),
                        ("Grounded bug context", state.get("bug_context_doc", "").strip() or "None."),
                        ("Blueprint-specific context", state.get("blueprint_context", "").strip() or "None."),
                        (
                            "Readable Blueprint text mirror",
                            blueprint_text or "No readable Blueprint text mirror was available in the workspace.",
                        ),
                    ],
                    closing="Keep the handoff safe, asset-specific, and explicit about what must be validated manually in Unreal Editor.",
                ),
            )
        except Exception:
            return fallback
    return fallback


def _compose_loop_feedback(
    *,
    title: str,
    round_index: int,
    score: int,
    threshold: int,
    approved: bool,
    confidence_label: str = "unmeasured",
    confidence_reason: str = "",
    confidence: float | None = None,
    blocking_issues: list[str],
    improvement_actions: list[str],
    sections: list[InvestigationCheck],
    loop_reason: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"- Round: {round_index}",
        f"- Score: {score}/100",
        f"- Threshold: {threshold}/100",
        f"- Scoring confidence: {confidence_label}{f' ({confidence:.1f}x noise floor)' if confidence is not None else ''}",
        f"- Confidence reason: {confidence_reason}",
        f"- Minimum verification depth: {MIN_INVESTIGATION_ROUNDS} round(s)",
        f"- Approved: {approved}",
        f"- Loop Reason: {loop_reason}",
        "",
        "## Section Scores",
    ]
    for item in sections:
        lines.append(f"- {item['section']}: {item['score']}/{item['max_score']} - {item['rationale']}")
    lines.extend(["", "## Blocking Issues"])
    lines.extend([f"- {item}" for item in blocking_issues] or ["- None."])
    lines.extend(["", "## Improvement Actions"])
    lines.extend([f"- {item}" for item in improvement_actions] or ["- None."])
    return "\n".join(lines)


def _to_score_assessments(checks: list[InvestigationCheck]) -> list[ScoreAssessment]:
    return [
        ScoreAssessment(
            label=item["section"],
            score=int(item["score"]),
            max_score=INVESTIGATION_SECTION_WEIGHT_MAP[item["section"]],
            status=item["status"],
            rationale=item["rationale"],
            action_items=tuple(item["action_items"]),
        )
        for item in checks
    ]


def _evaluate_investigation_quality(state: EngineerState) -> tuple[dict[str, Any], list[InvestigationCheck]]:
    checks: list[InvestigationCheck] = []
    artifact_dir_value = str(state.get("artifact_dir", "")).strip()
    score_history_dir = Path(artifact_dir_value) if artifact_dir_value else None

    def add_check(section: str, max_score: int, passed: bool, rationale: str, action: str) -> None:
        checks.append(
            {
                "section": section,
                "score": max_score if passed else max(1, round(max_score * 0.4)),
                "max_score": max_score,
                "status": "pass" if passed else "needs-work",
                "rationale": rationale,
                "action_items": [] if passed else [action],
            }
        )

    current_runtime_paths = [str(item).strip() for item in state.get("current_runtime_paths", []) if str(item).strip()]
    legacy_runtime_paths = [str(item).strip() for item in state.get("legacy_runtime_paths", []) if str(item).strip()]
    source_hits = [str(item).strip() for item in state.get("source_hits", []) if str(item).strip()]
    test_hits = [str(item).strip() for item in state.get("test_hits", []) if str(item).strip()]
    blueprint_hits = [str(item).strip() for item in state.get("blueprint_hits", []) if str(item).strip()]
    blueprint_text_hits = [str(item).strip() for item in state.get("blueprint_text_hits", []) if str(item).strip()]
    doc_hits = [str(item).strip() for item in state.get("doc_hits", []) if str(item).strip()]
    ownership_summary = str(state.get("ownership_summary", "")).strip()
    investigation_summary = str(state.get("investigation_summary", "")).strip()
    implementation_medium = str(state.get("implementation_medium", "")).strip().lower()
    implementation_medium_reason = str(state.get("implementation_medium_reason", "")).strip()
    root_cause_text = "\n".join(
        [
            str(state.get("investigation_root_cause", "")).strip(),
            *[str(item).strip() for item in state.get("runtime_path_hypotheses", []) if str(item).strip()],
        ]
    ).strip()
    validation_plan = str(state.get("investigation_validation_plan", "")).strip()
    owner_hint_text = "\n".join(
        [
            ownership_summary,
            str(state.get("code_context", "")).strip(),
            str(state.get("blueprint_context", "")).strip(),
            root_cause_text,
        ]
    )
    current_set = {item.lower() for item in current_runtime_paths}
    legacy_set = {item.lower() for item in legacy_runtime_paths}
    has_supporting_references = bool(doc_hits or source_hits or blueprint_hits or blueprint_text_hits)
    has_live_owner = bool(current_runtime_paths or blueprint_hits or blueprint_text_hits)
    has_owner_precision = has_live_owner and (
        bool(current_runtime_paths)
        or _contains_path_hint(owner_hint_text)
    )
    has_current_vs_legacy_split = (not legacy_set) or bool(current_set and not current_set.intersection(legacy_set))
    has_ownership_summary = bool(ownership_summary) and (has_live_owner or _contains_path_hint(ownership_summary))
    has_root_cause = bool(root_cause_text) or (
        has_live_owner
        and bool(investigation_summary)
        and int(state.get("investigation_round", 1)) >= MIN_INVESTIGATION_ROUNDS
    )
    has_investigation_summary = bool(investigation_summary)
    has_implementation_medium = implementation_medium in {"cpp", "blueprint"} and bool(implementation_medium_reason)
    has_validation = (
        bool(test_hits)
        or bool(validation_plan and (_contains_path_hint(validation_plan) or "manual" in normalize_text(validation_plan)))
        or bool(implementation_medium == "blueprint" and (blueprint_hits or blueprint_text_hits))
    )
    has_noise_control = (not legacy_runtime_paths) or has_live_owner

    add_check(
        "Supporting References",
        10,
        has_supporting_references,
        "Relevant docs, source files, or Blueprint assets support the investigation."
        if has_supporting_references
        else "The investigation still lacks grounded supporting references.",
        "Ground the investigation in concrete docs, source files, or Blueprint assets before planning.",
    )
    add_check(
        "Runtime Owner Precision",
        25,
        has_owner_precision,
        "The investigation isolated a current runtime owner with grounded path-level evidence."
        if has_owner_precision
        else "The live gameplay runtime owner is still too ambiguous.",
        "Identify the current runtime owner and anchor the handoff on the exact gameplay path.",
    )
    add_check(
        "Current vs Legacy Split",
        10,
        has_current_vs_legacy_split,
        "The investigation separated current runtime ownership from stale or archival references."
        if has_current_vs_legacy_split
        else "Legacy references still blur the live runtime owner.",
        "Separate the live gameplay owner from stale or archival references before implementation.",
    )
    add_check(
        "Ownership Summary",
        10,
        has_ownership_summary,
        "The ownership summary names the likely owner and why it is responsible."
        if has_ownership_summary
        else "The ownership handoff is still too thin.",
        "Tighten the ownership summary so it names the owner and the reason it owns this behavior.",
    )
    add_check(
        "Root Cause Hypothesis",
        15,
        has_root_cause,
        "The investigation names a plausible causal hypothesis for the gameplay behavior."
        if has_root_cause
        else "The investigation still needs a concrete causal hypothesis.",
        "State the likely failing transition, hook, or runtime condition before implementation.",
    )
    add_check(
        "Investigation Summary",
        10,
        has_investigation_summary,
        "The investigation summary is concrete enough to hand off."
        if has_investigation_summary
        else "The investigation summary still needs a tighter handoff.",
        "Summarize the grounded owner, likely cause, and next proof step in one handoff-ready note.",
    )
    add_check(
        "Implementation Medium",
        5,
        has_implementation_medium,
        "The investigation classified the work as code or Blueprint with rationale."
        if has_implementation_medium
        else "The implementation medium is still under-justified.",
        "Explain whether the fix belongs in code or Blueprint and why.",
    )
    add_check(
        "Validation Plan",
        10,
        has_validation,
        "A concrete automated or manual validation path is present."
        if has_validation
        else "The validation path is still too vague.",
        "Name the exact regression test, validation path, or manual Blueprint verification flow.",
    )
    add_check(
        "Noise Control",
        5,
        has_noise_control,
        "The evidence set stayed focused on the live gameplay owner."
        if has_noise_control
        else "The investigation still carries too much stale or noisy evidence.",
        "Trim stale evidence and keep the brief focused on the live gameplay owner only.",
    )

    blocking_issues = _dedupe(
        [
            f"{item['section']}: {item['action_items'][0]}"
            for item in checks
            if item["section"] in INVESTIGATION_BLOCKING_SECTIONS and item["status"] != "pass"
        ]
    )
    if int(state.get("investigation_round", 1)) < MIN_INVESTIGATION_ROUNDS:
        blocking_issues.append(
            f"Minimum verification depth is {MIN_INVESTIGATION_ROUNDS} rounds, so this investigation still needs one more independent pass."
        )
    improvement_actions = _dedupe(
        [
            action
            for item in checks
            if item["status"] != "pass"
            for action in item["action_items"]
        ]
    )
    if int(state.get("investigation_round", 1)) < MIN_INVESTIGATION_ROUNDS:
        improvement_actions.append(MANDATORY_INVESTIGATION_VERIFICATION_ACTION)
    proposed_score = sum(item["score"] for item in checks)
    score_decision = evaluate_score_decision(
        INVESTIGATION_SCORE_POLICY,
        round_index=max(1, int(state.get("investigation_round", 1))),
        assessments=_to_score_assessments(checks),
        explicit_approval=proposed_score >= INVESTIGATION_SCORE and not blocking_issues,
        blocking_issues=blocking_issues,
        improvement_actions=improvement_actions,
        artifact_dir=score_history_dir,
    )
    progress = evaluate_quality_loop(
        INVESTIGATION_LOOP_SPEC,
        round_index=max(1, int(state.get("investigation_round", 1))),
        score=score_decision.score,
        approved=score_decision.approved,
        blocking_issues=score_decision.blocking_issues,
        improvement_actions=score_decision.improvement_actions,
        previous_score=int(state.get("investigation_score", 0)) if int(state.get("investigation_round", 1)) > 1 else None,
        prior_stagnated_rounds=int(state.get("investigation_loop_stagnated_rounds", 0)) if int(state.get("investigation_round", 1)) > 1 else 0,
    )
    feedback = _compose_loop_feedback(
        title="Investigation Confidence Review",
        round_index=progress.round_index,
        score=progress.score,
        threshold=progress.threshold,
        approved=progress.approved,
        confidence_label=score_decision.confidence_label,
        confidence_reason=score_decision.confidence_reason,
        confidence=score_decision.confidence,
        blocking_issues=list(progress.blocking_issues),
        improvement_actions=list(progress.improvement_actions),
        sections=checks,
        loop_reason=progress.reason,
    )
    return (
        {
            "investigation_score": progress.score,
            "investigation_feedback": feedback,
            "investigation_missing_sections": list(progress.missing_sections),
            "investigation_blocking_issues": list(progress.blocking_issues),
            "investigation_improvement_actions": list(progress.improvement_actions),
            "investigation_approved": progress.approved,
            "investigation_loop_status": progress.status,
            "investigation_loop_reason": progress.reason,
            "investigation_loop_stagnated_rounds": progress.stagnated_rounds,
            "investigation_score_confidence": score_decision.confidence,
            "investigation_score_confidence_label": score_decision.confidence_label,
            "investigation_score_confidence_reason": score_decision.confidence_reason,
            "active_loop_should_continue": progress.should_continue,
            "active_loop_completed": progress.completed,
            "active_loop_status": progress.status,
        },
        checks,
    )


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name
    reviewer_graph = context.get_workflow_graph("gameplay-reviewer-workflow")

    def classify_request(state: EngineerState) -> dict[str, Any]:
        task_type, reason = _classify_task(context, state["task_prompt"])
        gameplay_scope_verdict = "gameplay" if task_type != "non_gameplay" else "non_gameplay"
        implementation_requested = not _is_read_only_request(state["task_prompt"]) and task_type != "non_gameplay"
        execution_track = task_type if task_type in {"bugfix", "feature", "maintenance"} else "bugfix"
        return {
            "task_type": task_type,
            "execution_track": execution_track,
            "gameplay_scope_verdict": gameplay_scope_verdict,
            "classification_reason": reason,
            "implementation_requested": implementation_requested,
            "requires_architecture_review": task_type in {"feature", "maintenance"},
            "review_round": 0,
            "review_score": 0,
            "review_feedback": "",
            "review_approved": False,
            "review_loop_status": "not-started",
            "review_loop_reason": "",
            "review_loop_should_continue": False,
            "review_loop_completed": False,
            "review_loop_stagnated_rounds": 0,
            "review_blocking_issues": [],
            "review_improvement_actions": [],
            "repair_round": 0,
            "repair_loop_status": "not-started",
            "repair_loop_reason": "",
            "code_attempt": 0,
            "implementation_status": "not-started",
            "blueprint_fix_strategy": "not-applicable",
            "blueprint_manual_action_required": False,
            "compile_ok": False,
            "tests_ok": False,
            "workspace_write_enabled": False,
            "workspace_write_summary": "",
            "workspace_source_file": "",
            "workspace_test_file": "",
            "summary": f"{metadata.name} classified the gameplay request as `{task_type}`.",
        }

    def request_investigation(state: EngineerState) -> dict[str, Any]:
        return {"summary": f"{metadata.name} requested a grounded gameplay investigation pass."}

    def prepare_investigation_strategy(state: EngineerState) -> dict[str, Any]:
        strategy = _prepare_investigation_strategy_payload(context, state)
        return {
            "investigation_round": int(state.get("investigation_round", 0)) + 1,
            "investigation_focus_terms": list(strategy["focus_terms"]),
            "investigation_avoid_terms": list(strategy["avoid_terms"]),
            "investigation_search_notes": list(strategy["search_notes"]),
            "implementation_medium_reason": str(strategy["implementation_medium_reason"]),
            "investigation_root_cause": str(strategy["investigation_root_cause"]),
            "investigation_validation_plan": str(strategy["investigation_validation_plan"]),
            "summary": f"{metadata.name} prepared gameplay investigation strategy round {int(state.get('investigation_round', 0)) + 1}.",
        }

    def simulate_engineer_investigation(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        payload = _collect_engineering_context(context, state)
        implementation_medium, implementation_reason = _choose_implementation_medium(state, payload)
        learning_summary, learning_focus, learning_doc = _build_investigation_learning(state, payload)
        investigation_doc = _compose_investigation_doc(
            {
                **state,
                "implementation_medium": implementation_medium,
                "implementation_medium_reason": implementation_reason,
            },
            payload,
            round_index=int(state["investigation_round"]),
        )
        round_index = int(state["investigation_round"])
        (artifact_dir / f"engineer_investigation_round_{round_index}.md").write_text(investigation_doc, encoding="utf-8")
        (artifact_dir / "engineer_investigation.md").write_text(investigation_doc, encoding="utf-8")
        (artifact_dir / f"investigation_learning_round_{round_index}.md").write_text(learning_doc, encoding="utf-8")
        previous_log = ""
        log_path = artifact_dir / "investigation_learning_log.md"
        if log_path.exists():
            previous_log = log_path.read_text(encoding="utf-8").strip()
        log_path.write_text("\n\n".join(item for item in [previous_log, learning_doc] if item).strip(), encoding="utf-8")
        return {
            "artifact_dir": str(artifact_dir),
            "doc_hits": list(payload["doc_hits"]),
            "source_hits": list(payload["source_hits"]),
            "test_hits": list(payload["test_hits"]),
            "blueprint_hits": list(payload["blueprint_hits"]),
            "blueprint_text_hits": list(payload["blueprint_text_hits"]),
            "current_runtime_paths": list(payload["current_runtime_paths"]),
            "legacy_runtime_paths": list(payload["legacy_runtime_paths"]),
            "runtime_path_hypotheses": list(payload["runtime_path_hypotheses"]),
            "ownership_summary": str(payload.get("ownership_summary", "")).strip(),
            "investigation_summary": str(payload.get("investigation_summary", "")).strip() or learning_summary,
            "doc_context": str(payload.get("doc_context", "")).strip(),
            "code_context": str(payload.get("code_context", "")).strip(),
            "blueprint_context": str(payload.get("blueprint_context", "")).strip(),
            "implementation_medium": implementation_medium,
            "implementation_medium_reason": implementation_reason,
            "investigation_doc": investigation_doc,
            "investigation_learning_summary": learning_summary,
            "investigation_learning_focus": learning_focus,
            "summary": f"{metadata.name} completed gameplay investigation round {round_index}.",
        }

    def assess_implementation_strategy(state: EngineerState) -> dict[str, Any]:
        scope_root = context.resolve_scope_root("host_project")
        allow_workspace_writes = bool(state.get("implementation_requested")) and state["implementation_medium"] == "cpp"
        workspace_source_file, workspace_test_file = _resolve_workspace_targets(
            task_id=str(state.get("task_id", state["task_prompt"])),
            scope_root=scope_root,
            source_hits=list(state.get("source_hits", [])),
            test_hits=list(state.get("test_hits", [])),
            source_roots=context.config.source_roots,
            test_roots=context.config.test_roots,
            allow_workspace_writes=allow_workspace_writes,
        )
        blueprint_fix_strategy = "manual-unreal-editor" if state["implementation_medium"] == "blueprint" else "not-applicable"
        if not state["implementation_requested"]:
            workspace_write_summary = "Read-only investigation requested; workspace writes remain disabled."
        elif state["implementation_medium"] == "blueprint":
            workspace_write_summary = "Manual Unreal Editor validation is required for Blueprint-only changes."
        elif allow_workspace_writes:
            workspace_write_summary = f"Workspace writes enabled for {workspace_source_file} and {workspace_test_file}."
        else:
            workspace_write_summary = "Workspace writes are unavailable because no writable gameplay source roots were found."
        return {
            "workspace_write_enabled": allow_workspace_writes,
            "workspace_write_summary": workspace_write_summary,
            "workspace_source_file": workspace_source_file,
            "workspace_test_file": workspace_test_file,
            "blueprint_fix_strategy": blueprint_fix_strategy,
            "blueprint_manual_action_required": state["implementation_medium"] == "blueprint",
            "summary": f"{metadata.name} assessed gameplay implementation strategy as `{state['implementation_medium']}`.",
        }

    def evaluate_investigation(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        updates, checks = _evaluate_investigation_quality(state)
        review_doc = str(updates["investigation_feedback"])
        (artifact_dir / f"investigation_review_round_{state['investigation_round']}.md").write_text(review_doc, encoding="utf-8")
        return {
            **updates,
            "summary": (
                f"{metadata.name} approved gameplay investigation round {state['investigation_round']}."
                if updates["investigation_approved"]
                else f"{metadata.name} requested another gameplay investigation round."
            ),
        }

    def build_design_doc(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        design_doc = _compose_design_doc(context, state)
        (artifact_dir / "architecture_plan.md").write_text(design_doc, encoding="utf-8")
        return {"design_doc": design_doc, "summary": f"{metadata.name} prepared gameplay design context."}

    def build_bug_context_doc(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        bug_context_doc = _compose_bug_context_doc(context, state)
        (artifact_dir / "bug_context.md").write_text(bug_context_doc, encoding="utf-8")
        return {"bug_context_doc": bug_context_doc, "summary": f"{metadata.name} prepared gameplay bug context."}

    def prepare_investigation_blocked_delivery(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        blocked_doc = "\n".join(
            [
                "# Investigation Blocked Delivery",
                "",
                f"- Task type: {state['task_type']}",
                f"- Investigation loop status: {state.get('investigation_loop_status', 'unknown')}",
                "",
                "## Blocking Issues",
                *_format_bullets(list(state.get("investigation_blocking_issues", []))).splitlines(),
                "",
                "## Next Actions",
                *_format_bullets(list(state.get("investigation_improvement_actions", []))).splitlines(),
            ]
        )
        (artifact_dir / "investigation_abort.md").write_text(blocked_doc, encoding="utf-8")
        final_report = {
            "status": "investigation-blocked",
            "implementation_requested": bool(state.get("implementation_requested", False)),
            "investigation_loop_status": str(state.get("investigation_loop_status", "unknown")),
            "review_loop_status": str(state.get("review_loop_status", "not-started")),
            "repair_loop_status": str(state.get("repair_loop_status", "not-started")),
            "compile_ok": False,
            "tests_ok": False,
            "implementation_medium": str(state.get("implementation_medium", "")),
            "blueprint_manual_action_required": bool(state.get("blueprint_manual_action_required", False)),
        }
        (artifact_dir / "final_report.md").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
        return {
            "final_report": final_report,
            "summary": f"{metadata.name} stopped because gameplay investigation never grounded a safe handoff.",
        }

    def prepare_investigation_delivery(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        delivery_doc = "\n".join(
            [
                "# Investigation Delivery",
                "",
                f"- Task type: {state['task_type']}",
                f"- Execution track: {state['execution_track']}",
                f"- Implementation requested: {state['implementation_requested']}",
                "",
                "## Ownership Summary",
                state.get("ownership_summary", "") or "No ownership summary was produced.",
                "",
                "## Current Runtime Paths",
                _format_bullets(list(state.get("current_runtime_paths", []))),
                "",
                "## Validation Path",
                _format_bullets(list(state.get("test_hits", [])), empty_message="No grounded test path found."),
            ]
        )
        (artifact_dir / "investigation_delivery.md").write_text(delivery_doc, encoding="utf-8")
        final_report = {
            "status": "investigation-completed",
            "implementation_requested": bool(state.get("implementation_requested", False)),
            "investigation_loop_status": str(state.get("investigation_loop_status", "passed")),
            "review_loop_status": str(state.get("review_loop_status", "not-started")),
            "repair_loop_status": str(state.get("repair_loop_status", "not-started")),
            "compile_ok": False,
            "tests_ok": False,
            "implementation_medium": str(state.get("implementation_medium", "")),
            "blueprint_manual_action_required": bool(state.get("blueprint_manual_action_required", False)),
        }
        (artifact_dir / "final_report.md").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
        return {
            "final_report": final_report,
            "summary": f"{metadata.name} delivered a gameplay investigation handoff without implementation.",
        }

    def plan_work(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        plan_doc = _compose_plan_doc(context, state, revise=False)
        (artifact_dir / "plan_doc.md").write_text(plan_doc, encoding="utf-8")
        return {"plan_doc": plan_doc, "summary": f"{metadata.name} drafted a gameplay implementation plan."}

    def request_review(state: EngineerState) -> dict[str, Any]:
        return {"summary": f"{metadata.name} requested gameplay plan review round {int(state.get('review_round', 0)) + 1}."}

    def enter_review_subgraph(state: EngineerState) -> dict[str, Any]:
        return {"summary": f"{metadata.name} entered the gameplay reviewer subgraph."}

    def capture_review_result(state: EngineerState) -> dict[str, Any]:
        return {
            "review_score": int(state.get("review_score", state.get("score", 0))),
            "review_feedback": str(state.get("review_feedback", state.get("feedback", ""))),
            "review_approved": bool(state.get("review_approved", state.get("approved", False))),
            "review_loop_status": str(state.get("review_loop_status", state.get("loop_status", "unknown"))),
            "review_loop_reason": str(state.get("review_loop_reason", state.get("loop_reason", ""))),
            "review_loop_should_continue": bool(state.get("review_loop_should_continue", state.get("loop_should_continue", False))),
            "review_loop_completed": bool(state.get("review_loop_completed", state.get("loop_completed", False))),
            "review_loop_stagnated_rounds": int(state.get("review_loop_stagnated_rounds", state.get("loop_stagnated_rounds", 0))),
            "review_blocking_issues": list(state.get("review_blocking_issues", state.get("blocking_issues", []))),
            "review_improvement_actions": list(
                state.get("review_improvement_actions", state.get("improvement_actions", []))
            ),
            "summary": (
                f"{metadata.name} review approved the gameplay plan."
                if state.get("review_approved", state.get("approved"))
                else f"{metadata.name} review requested another gameplay plan revision."
            ),
        }

    def revise_plan(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        plan_doc = _compose_plan_doc(context, state, revise=True)
        (artifact_dir / "plan_doc.md").write_text(plan_doc, encoding="utf-8")
        return {"plan_doc": plan_doc, "summary": f"{metadata.name} revised the gameplay implementation plan."}

    def implement_code(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        scope_root = context.resolve_scope_root("host_project")
        if state["implementation_medium"] == "blueprint":
            blueprint_target = ""
            if state.get("blueprint_hits"):
                blueprint_target = str(state["blueprint_hits"][0])
            elif state.get("blueprint_text_hits"):
                blueprint_target = str(state["blueprint_text_hits"][0])
            instructions_doc = _compose_blueprint_handoff(
                context,
                state,
                scope_root=scope_root,
                blueprint_target=blueprint_target,
            )
            manifest_doc = "\n".join(
                [
                    "# Blueprint Fix Manifest",
                    "",
                    f"- Blueprint manual action required: True",
                    f"- Blueprint fix strategy: {state['blueprint_fix_strategy']}",
                    f"- Target: {blueprint_target or 'None.'}",
                ]
            )
            (artifact_dir / "blueprint_fix_instructions.md").write_text(instructions_doc, encoding="utf-8")
            (artifact_dir / "blueprint_fix_manifest.md").write_text(manifest_doc, encoding="utf-8")
            if blueprint_target:
                target_path = scope_root / blueprint_target
                note_path = target_path.with_name(f"{target_path.stem}.agentswarm_fix.md")
                note_path.write_text(instructions_doc, encoding="utf-8")
            return {
                "implementation_notes": "Prepared a manual Blueprint handoff.",
                "implementation_status": "manual-validation-required",
                "summary": f"{metadata.name} prepared manual Blueprint gameplay fix instructions.",
            }

        code_bundle = _fallback_code_bundle(state, scope_root)
        codegen_llm = context.get_llm("codegen")
        if codegen_llm.is_enabled():
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
                source_path = scope_root / state["workspace_source_file"]
                test_path = scope_root / state["workspace_test_file"]
                current_source_code = _safe_read_text(source_path, limit=6000)
                current_test_code = _safe_read_text(test_path, limit=6000)
                generated = codegen_llm.generate_json(
                    instructions=(
                        "Generate a gameplay code bundle. Edit the provided workspace source and test files instead of rewriting blindly. "
                        "Preserve unrelated behavior, keep the change scoped to the grounded owner, and respond directly to any prior self-test failures."
                    ),
                    input_text=build_prompt_brief(
                        opening="Prepare the next gameplay code-and-tests bundle for this grounded owner path.",
                        sections=[
                            ("Task request", state["task_prompt"].strip()),
                            (
                                "Implementation frame",
                                "\n".join(
                                    [
                                        f"- Task type: {state['task_type']}",
                                        f"- Execution track: {state['execution_track']}",
                                        f"- Workspace source file: {state['workspace_source_file']}",
                                        f"- Workspace test file: {state['workspace_test_file']}",
                                    ]
                                ),
                            ),
                            (
                                "Grounded runtime owner",
                                _format_bullets(list(state.get("current_runtime_paths", []))),
                            ),
                            (
                                "Grounded validation paths",
                                _format_bullets(list(state.get("test_hits", []))),
                            ),
                            ("Investigation summary", state.get("investigation_doc", "").strip() or "None."),
                            ("Review feedback", state.get("review_feedback", "").strip() or "None."),
                            ("Plan document", state.get("plan_doc", "").strip() or "None."),
                            ("Design context", state.get("design_doc", "").strip() or "None."),
                            ("Bug context", state.get("bug_context_doc", "").strip() or "None."),
                            (
                                "Previous self-test output",
                                state.get("self_test_output", "").strip() or "None. This is the first code attempt.",
                            ),
                            (
                                "Current source file contents",
                                current_source_code or "File is empty or missing.",
                            ),
                            (
                                "Current test file contents",
                                current_test_code or "File is empty or missing.",
                            ),
                        ],
                        closing="Keep the change narrow, preserve adjacent gameplay behavior, and make the tests prove the intended player-visible outcome.",
                    ),
                    schema_name="gameplay_code_bundle",
                    schema=schema,
                )
                code_bundle = {
                    "source_code": str(generated.get("source_code", code_bundle["source_code"])),
                    "test_code": str(generated.get("test_code", code_bundle["test_code"])),
                    "implementation_notes": str(generated.get("implementation_notes", code_bundle["implementation_notes"])),
                }
            except Exception:
                code_bundle = _fallback_code_bundle(state, scope_root)

        source_path = scope_root / state["workspace_source_file"]
        test_path = scope_root / state["workspace_test_file"]
        source_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(code_bundle["source_code"], encoding="utf-8")
        test_path.write_text(code_bundle["test_code"], encoding="utf-8")
        manifest = "\n".join(
            [
                "# Workspace Write Manifest",
                "",
                f"- Source file: {state['workspace_source_file']}",
                f"- Test file: {state['workspace_test_file']}",
                f"- Code attempt: {int(state.get('code_attempt', 0)) + 1}",
                f"- Implementation medium: {state['implementation_medium']}",
                f"- Blueprint fix strategy: {state['blueprint_fix_strategy']}",
            ]
        )
        (artifact_dir / "workspace_write_manifest.md").write_text(manifest, encoding="utf-8")
        return {
            "code_attempt": int(state.get("code_attempt", 0)) + 1,
            "implementation_notes": code_bundle["implementation_notes"],
            "implementation_status": "code-written",
            "summary": f"{metadata.name} wrote gameplay code attempt {int(state.get('code_attempt', 0)) + 1}.",
        }

    def post_implementation_gate(state: EngineerState) -> dict[str, Any]:
        return {"summary": f"{metadata.name} evaluated whether the gameplay change needs automated validation."}

    def request_repair_validation(state: EngineerState) -> dict[str, Any]:
        return {"summary": f"{metadata.name} requested gameplay self-test validation."}

    def self_test(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        scope_root = context.resolve_scope_root("host_project")
        source_code = (scope_root / state["workspace_source_file"]).read_text(encoding="utf-8")
        test_code = (scope_root / state["workspace_test_file"]).read_text(encoding="utf-8")
        compile_ok, tests_ok, output = _run_compile_and_tests(source_code, test_code)
        (artifact_dir / "self_test.txt").write_text(output or "No output.", encoding="utf-8")
        return {
            "compile_ok": compile_ok,
            "tests_ok": tests_ok,
            "self_test_output": output,
            "summary": f"{metadata.name} completed gameplay self-test validation.",
        }

    def capture_repair_validation(state: EngineerState) -> dict[str, Any]:
        return {
            "repair_loop_status": "passed" if state["compile_ok"] and state["tests_ok"] else "retry",
            "repair_loop_reason": (
                "Gameplay validation passed."
                if state["compile_ok"] and state["tests_ok"]
                else "Gameplay validation still fails and needs another repair pass."
            ),
            "summary": (
                f"{metadata.name} validated the gameplay change successfully."
                if state["compile_ok"] and state["tests_ok"]
                else f"{metadata.name} needs another gameplay repair pass."
            ),
        }

    def repair_code(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        next_round = int(state.get("repair_round", 0)) + 1
        result = implement_code(state)
        repair_doc = "\n".join(
            [
                f"# Repair Round {next_round}",
                "",
                "## Previous Self Test Output",
                state.get("self_test_output", "") or "No previous output.",
                "",
                "## New Implementation Notes",
                result.get("implementation_notes", "") or "No new notes.",
            ]
        )
        (artifact_dir / f"repair_round_{next_round}.md").write_text(repair_doc, encoding="utf-8")
        return {
            **result,
            "repair_round": next_round,
            "summary": f"{metadata.name} executed gameplay repair round {next_round}.",
        }

    def prepare_delivery(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        final_status = "completed"
        if state["implementation_medium"] == "blueprint":
            final_status = "manual-validation-required"
        elif not state["implementation_requested"]:
            final_status = "investigation-completed"
        pull_request = "\n".join(
            [
                "# Gameplay Delivery Summary",
                "",
                f"Implementation medium: {state['implementation_medium']}",
                f"Blueprint fix strategy: {state['blueprint_fix_strategy']}",
                f"Blueprint manual action required: {state['blueprint_manual_action_required']}",
                f"Compile OK: {state.get('compile_ok', False)}",
                f"Tests OK: {state.get('tests_ok', False)}",
                "",
                "## Notes",
                state.get("implementation_notes", "") or "No additional implementation notes.",
            ]
        )
        (artifact_dir / "pull_request.md").write_text(pull_request, encoding="utf-8")
        final_report = {
            "status": final_status,
            "implementation_requested": bool(state.get("implementation_requested", False)),
            "implementation_medium": state["implementation_medium"],
            "blueprint_manual_action_required": bool(state.get("blueprint_manual_action_required", False)),
            "investigation_loop_status": str(state.get("investigation_loop_status", "passed")),
            "review_loop_status": str(state.get("review_loop_status", "passed")),
            "repair_loop_status": str(state.get("repair_loop_status", "passed")),
            "compile_ok": bool(state.get("compile_ok", False)),
            "tests_ok": bool(state.get("tests_ok", False)),
        }
        (artifact_dir / "final_report.md").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
        return {
            "final_report": final_report,
            "summary": f"{metadata.name} completed gameplay delivery with status `{final_status}`.",
        }

    def prepare_repair_blocked_delivery(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        repair_abort = "\n".join(
            [
                "# Repair Abort",
                "",
                f"- Repair rounds: {state.get('repair_round', 0)}",
                f"- Latest self test output: {state.get('self_test_output', '') or 'None.'}",
            ]
        )
        (artifact_dir / "repair_abort.md").write_text(repair_abort, encoding="utf-8")
        final_report = {
            "status": "repair-blocked",
            "implementation_requested": bool(state.get("implementation_requested", True)),
            "implementation_medium": state["implementation_medium"],
            "blueprint_manual_action_required": bool(state.get("blueprint_manual_action_required", False)),
            "investigation_loop_status": str(state.get("investigation_loop_status", "passed")),
            "review_loop_status": str(state.get("review_loop_status", "passed")),
            "repair_loop_status": "stagnated",
            "compile_ok": bool(state.get("compile_ok", False)),
            "tests_ok": bool(state.get("tests_ok", False)),
        }
        (artifact_dir / "final_report.md").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
        return {
            "repair_loop_status": "stagnated",
            "repair_loop_reason": "Gameplay repair loop exhausted the allowed rounds without a passing self-test.",
            "final_report": final_report,
            "summary": f"{metadata.name} stopped after the gameplay repair loop stagnated.",
        }

    def prepare_review_blocked_delivery(state: EngineerState) -> dict[str, Any]:
        artifact_dir = _artifact_dir(context, metadata, state)
        review_abort = "\n".join(
            [
                "# Review Abort",
                "",
                f"- Review rounds: {state.get('review_round', 0)}",
                f"- Review score: {state.get('review_score', 0)}/100",
                "",
                "## Blocking Issues",
                _format_bullets(list(state.get("review_blocking_issues", []))),
            ]
        )
        (artifact_dir / "review_abort.md").write_text(review_abort, encoding="utf-8")
        final_report = {
            "status": "review-blocked",
            "implementation_requested": bool(state.get("implementation_requested", True)),
            "implementation_medium": state["implementation_medium"],
            "blueprint_manual_action_required": bool(state.get("blueprint_manual_action_required", False)),
            "investigation_loop_status": str(state.get("investigation_loop_status", "passed")),
            "review_loop_status": "stagnated",
            "repair_loop_status": str(state.get("repair_loop_status", "not-started")),
            "compile_ok": False,
            "tests_ok": False,
        }
        (artifact_dir / "final_report.md").write_text(json.dumps(final_report, indent=2), encoding="utf-8")
        return {
            "final_report": final_report,
            "summary": f"{metadata.name} never reached gameplay plan approval.",
        }

    def evaluate_investigation_route(state: EngineerState) -> str:
        if state["task_type"] == "non_gameplay":
            return "prepare_investigation_blocked_delivery"
        if state["active_loop_should_continue"]:
            return "request_investigation"
        if not state["investigation_approved"]:
            return "prepare_investigation_blocked_delivery"
        return "build_design_doc" if state["task_type"] in {"feature", "maintenance"} else "build_bug_context_doc"

    def after_design_doc(state: EngineerState) -> str:
        return "plan_work" if state["implementation_requested"] else "prepare_investigation_delivery"

    def after_bug_context_doc(state: EngineerState) -> str:
        return "implement_code" if state["implementation_requested"] else "prepare_investigation_delivery"

    def after_review(state: EngineerState) -> str:
        if state["review_approved"]:
            return "implement_code"
        if state["review_loop_should_continue"]:
            return "revise_plan"
        return "prepare_review_blocked_delivery"

    def after_post_implementation_gate(state: EngineerState) -> str:
        if state["implementation_medium"] == "blueprint":
            return "prepare_delivery"
        return "request_repair_validation"

    def after_repair_validation(state: EngineerState) -> str:
        if state["compile_ok"] and state["tests_ok"]:
            return "prepare_delivery"
        if int(state.get("repair_round", 0)) >= MAX_REPAIR_ROUNDS:
            return "prepare_repair_blocked_delivery"
        return "repair_code"

    graph = StateGraph(EngineerState)
    graph.add_node("classify_request", trace_graph_node(graph_name=graph_name, node_name="classify_request", node_fn=classify_request))
    graph.add_node("request_investigation", trace_graph_node(graph_name=graph_name, node_name="request_investigation", node_fn=request_investigation))
    graph.add_node("prepare_investigation_strategy", trace_graph_node(graph_name=graph_name, node_name="prepare_investigation_strategy", node_fn=prepare_investigation_strategy))
    graph.add_node("simulate_engineer_investigation", trace_graph_node(graph_name=graph_name, node_name="simulate_engineer_investigation", node_fn=simulate_engineer_investigation))
    graph.add_node("assess_implementation_strategy", trace_graph_node(graph_name=graph_name, node_name="assess_implementation_strategy", node_fn=assess_implementation_strategy))
    graph.add_node("evaluate_investigation", trace_graph_node(graph_name=graph_name, node_name="evaluate_investigation", node_fn=evaluate_investigation))
    graph.add_node("build_design_doc", trace_graph_node(graph_name=graph_name, node_name="build_design_doc", node_fn=build_design_doc))
    graph.add_node("build_bug_context_doc", trace_graph_node(graph_name=graph_name, node_name="build_bug_context_doc", node_fn=build_bug_context_doc))
    graph.add_node("prepare_investigation_blocked_delivery", trace_graph_node(graph_name=graph_name, node_name="prepare_investigation_blocked_delivery", node_fn=prepare_investigation_blocked_delivery))
    graph.add_node("prepare_investigation_delivery", trace_graph_node(graph_name=graph_name, node_name="prepare_investigation_delivery", node_fn=prepare_investigation_delivery))
    graph.add_node("plan_work", trace_graph_node(graph_name=graph_name, node_name="plan_work", node_fn=plan_work))
    graph.add_node("request_review", trace_graph_node(graph_name=graph_name, node_name="request_review", node_fn=request_review))
    graph.add_node("enter_review_subgraph", trace_graph_node(graph_name=graph_name, node_name="enter_review_subgraph", node_fn=enter_review_subgraph))
    graph.add_node("gameplay-reviewer-workflow", reviewer_graph)
    graph.add_node("capture_review_result", trace_graph_node(graph_name=graph_name, node_name="capture_review_result", node_fn=capture_review_result))
    graph.add_node("revise_plan", trace_graph_node(graph_name=graph_name, node_name="revise_plan", node_fn=revise_plan))
    graph.add_node("implement_code", trace_graph_node(graph_name=graph_name, node_name="implement_code", node_fn=implement_code))
    graph.add_node("post_implementation_gate", trace_graph_node(graph_name=graph_name, node_name="post_implementation_gate", node_fn=post_implementation_gate))
    graph.add_node("request_repair_validation", trace_graph_node(graph_name=graph_name, node_name="request_repair_validation", node_fn=request_repair_validation))
    graph.add_node("self_test", trace_graph_node(graph_name=graph_name, node_name="self_test", node_fn=self_test))
    graph.add_node("capture_repair_validation", trace_graph_node(graph_name=graph_name, node_name="capture_repair_validation", node_fn=capture_repair_validation))
    graph.add_node("repair_code", trace_graph_node(graph_name=graph_name, node_name="repair_code", node_fn=repair_code))
    graph.add_node("prepare_delivery", trace_graph_node(graph_name=graph_name, node_name="prepare_delivery", node_fn=prepare_delivery))
    graph.add_node("prepare_repair_blocked_delivery", trace_graph_node(graph_name=graph_name, node_name="prepare_repair_blocked_delivery", node_fn=prepare_repair_blocked_delivery))
    graph.add_node("prepare_review_blocked_delivery", trace_graph_node(graph_name=graph_name, node_name="prepare_review_blocked_delivery", node_fn=prepare_review_blocked_delivery))
    graph.add_edge(START, "classify_request")
    graph.add_edge("classify_request", "request_investigation")
    graph.add_edge("request_investigation", "prepare_investigation_strategy")
    graph.add_edge("prepare_investigation_strategy", "simulate_engineer_investigation")
    graph.add_edge("simulate_engineer_investigation", "assess_implementation_strategy")
    graph.add_edge("assess_implementation_strategy", "evaluate_investigation")
    graph.add_conditional_edges(
        "evaluate_investigation",
        trace_route_decision(graph_name=graph_name, router_name="evaluate_investigation_route", route_fn=evaluate_investigation_route),
        {
            "request_investigation": "request_investigation",
            "build_design_doc": "build_design_doc",
            "build_bug_context_doc": "build_bug_context_doc",
            "prepare_investigation_blocked_delivery": "prepare_investigation_blocked_delivery",
        },
    )
    graph.add_conditional_edges("build_design_doc", after_design_doc, {"plan_work": "plan_work", "prepare_investigation_delivery": "prepare_investigation_delivery"})
    graph.add_conditional_edges("build_bug_context_doc", after_bug_context_doc, {"implement_code": "implement_code", "prepare_investigation_delivery": "prepare_investigation_delivery"})
    graph.add_edge("plan_work", "request_review")
    graph.add_edge("request_review", "enter_review_subgraph")
    graph.add_edge("enter_review_subgraph", "gameplay-reviewer-workflow")
    graph.add_edge("gameplay-reviewer-workflow", "capture_review_result")
    graph.add_conditional_edges("capture_review_result", after_review, {"implement_code": "implement_code", "revise_plan": "revise_plan", "prepare_review_blocked_delivery": "prepare_review_blocked_delivery"})
    graph.add_edge("revise_plan", "request_review")
    graph.add_edge("implement_code", "post_implementation_gate")
    graph.add_conditional_edges("post_implementation_gate", after_post_implementation_gate, {"prepare_delivery": "prepare_delivery", "request_repair_validation": "request_repair_validation"})
    graph.add_edge("request_repair_validation", "self_test")
    graph.add_edge("self_test", "capture_repair_validation")
    graph.add_conditional_edges("capture_repair_validation", after_repair_validation, {"prepare_delivery": "prepare_delivery", "repair_code": "repair_code", "prepare_repair_blocked_delivery": "prepare_repair_blocked_delivery"})
    graph.add_edge("repair_code", "request_repair_validation")
    graph.add_edge("prepare_delivery", END)
    graph.add_edge("prepare_investigation_blocked_delivery", END)
    graph.add_edge("prepare_investigation_delivery", END)
    graph.add_edge("prepare_repair_blocked_delivery", END)
    graph.add_edge("prepare_review_blocked_delivery", END)
    return graph.compile()
