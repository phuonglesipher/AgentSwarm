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
PLANNING_MODE_TO_TASK_TYPE = {
    "bugfix": "bugfix",
    "refactor": "maintenance",
    "improve_feature": "feature",
    "new_feature": "feature",
    "non_gameplay": "non_gameplay",
}
TASK_TYPE_DEFAULT_PLANNING_MODE = {
    "bugfix": "bugfix",
    "feature": "new_feature",
    "maintenance": "refactor",
    "non_gameplay": "non_gameplay",
}
PLANNING_MODE_LABELS = {
    "bugfix": "bugfix",
    "refactor": "refactor",
    "improve_feature": "feature improvement",
    "new_feature": "new feature",
    "non_gameplay": "non-gameplay",
}
BUGFIX_INTENT_MARKERS = (
    "fix",
    "bug",
    "regression",
    "broken",
    "failure",
    "fails",
    "crash",
    "error",
)
REFACTOR_INTENT_MARKERS = (
    "refactor",
    "cleanup",
    "clean up",
    "restructure",
    "reorganize",
    "simplify",
    "decouple",
    "extract",
    "stabilize",
    "stabilise",
    "harden",
    "maintainability",
    "tech debt",
)
IMPROVE_FEATURE_INTENT_MARKERS = (
    "improve",
    "enhance",
    "extend",
    "expand",
    "upgrade",
    "optimiz",
    "polish",
    "rebalance",
    "tune",
    "iterate on",
    "adjust",
)
EXISTING_FEATURE_HINTS = (
    "existing feature",
    "current feature",
    "existing gameplay",
    "current gameplay",
    "already ",
)
NEW_FEATURE_INTENT_MARKERS = (
    "new feature",
    "add ",
    "implement ",
    "create ",
    "introduce ",
    "build ",
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
    planning_mode: str
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


def _default_planning_mode(task_type: str) -> str:
    return TASK_TYPE_DEFAULT_PLANNING_MODE.get(task_type, "bugfix")


def _planning_mode_label(planning_mode: str) -> str:
    return PLANNING_MODE_LABELS.get(planning_mode, planning_mode.replace("_", " "))


def _normalize_planning_mode(task_type: str, planning_mode: str) -> str:
    normalized_mode = planning_mode.strip().lower()
    allowed_modes = {
        "bugfix": {"bugfix"},
        "feature": {"improve_feature", "new_feature"},
        "maintenance": {"refactor"},
        "non_gameplay": {"non_gameplay"},
    }
    if normalized_mode in allowed_modes.get(task_type, set()):
        return normalized_mode
    return _default_planning_mode(task_type)


def _infer_planning_mode(task_prompt: str, *, fallback_task_type: str) -> str:
    lowered = normalize_text(task_prompt)
    if fallback_task_type == "non_gameplay":
        return "non_gameplay"
    if any(marker in lowered for marker in BUGFIX_INTENT_MARKERS):
        return "bugfix"
    if any(marker in lowered for marker in REFACTOR_INTENT_MARKERS):
        return "refactor"
    mentions_existing_feature = any(marker in lowered for marker in EXISTING_FEATURE_HINTS)
    if any(marker in lowered for marker in IMPROVE_FEATURE_INTENT_MARKERS) or (
        fallback_task_type == "feature" and mentions_existing_feature
    ):
        return "improve_feature"
    if any(marker in lowered for marker in NEW_FEATURE_INTENT_MARKERS):
        return "new_feature"
    return _default_planning_mode(fallback_task_type)


def _planning_mode_profile(planning_mode: str) -> dict[str, Any]:
    if planning_mode == "refactor":
        return {
            "task_focus": "Planning focus: preserve player-visible behavior while improving structure, boundaries, and maintainability.",
            "design_overview": "- Treat this as refactor work: ground the current owner, the cleanup goal, and the invariants that cannot drift.",
            "design_behavior": "- Name the player-facing behavior that must remain unchanged and the hidden engineering pain the refactor is meant to remove.",
            "design_technical_note": "Capture the current seams, ordering constraints, migration boundaries, and rollback points before implementation.",
            "design_risk": "- Risk: behavior drift can slip in while reorganizing ownership, ordering, or shared helpers.",
            "design_focus": "Focus the brief on system boundaries, invariants, migration safety, and proof that behavior should stay stable.",
            "plan_overview": "- Preserve the current player-visible behavior unless the request explicitly calls for a visible improvement.",
            "plan_steps": [
                "Split the refactor into safe slices, keep public contracts stable, and avoid mixing cleanup with unrelated behavior changes.",
                "Protect invariants, ordering, and data flow explicitly so the new structure stays behaviorally equivalent.",
            ],
            "validation_focus": "Prove the refactor does not change player-visible behavior on the main path or its closest neighboring path.",
            "default_tests": [
                "Add or update regression coverage that proves the refactor preserved the current gameplay behavior.",
                "Add a targeted test for the most fragile neighboring path that could drift during the cleanup.",
            ],
            "risks": [
                "- Risk: hidden coupling or ordering assumptions may break once code is moved or split.",
                "- Mitigation: refactor in narrow slices, keep interfaces stable, and cover no-behavior-drift cases with tests.",
            ],
            "acceptance": [
                "- The targeted code path is cleaner, easier to own, or easier to extend without changing the intended gameplay outcome.",
                "- Existing player-visible behavior and adjacent regression checks stay unchanged.",
            ],
        }
    if planning_mode == "improve_feature":
        return {
            "task_focus": "Planning focus: evolve an existing gameplay feature, compare current behavior against the target improvement, and preserve compatibility.",
            "design_overview": "- Treat this as a feature improvement: ground the current behavior first, then describe the exact upgrade the player should feel.",
            "design_behavior": "- Compare the current player-facing behavior against the target improvement and call out which existing expectations must remain compatible.",
            "design_technical_note": "Capture the live extension point, compatibility constraints, and edge cases that the current feature already supports.",
            "design_risk": "- Risk: the improvement may help the main path but regress an existing contract, edge case, or balancing expectation.",
            "design_focus": "Focus the brief on current-vs-target behavior, compatibility boundaries, and the edge cases that existing players already rely on.",
            "plan_overview": "- Treat this as an evolution of an existing gameplay feature rather than a net-new system.",
            "plan_steps": [
                "Document the current behavior and hook the change into the existing feature owner instead of adding a parallel path.",
                "Preserve supported behavior, neighboring states, and existing caps or gating rules while applying the improvement.",
            ],
            "validation_focus": "Cover both the upgraded behavior and the existing supported behavior that must remain compatible.",
            "default_tests": [
                "Add or update tests that prove the requested improvement works on the existing feature path.",
                "Add a compatibility regression test for the old behavior or edge case that must remain intact.",
            ],
            "risks": [
                "- Risk: the improvement could regress a previously supported interaction, cap, or edge case.",
                "- Mitigation: compare current-vs-target behavior explicitly and keep compatibility coverage around the old path.",
            ],
            "acceptance": [
                "- Players experience the requested improvement on the existing feature path.",
                "- Existing supported behavior, neighboring states, and edge-case coverage remain intact.",
            ],
        }
    if planning_mode == "new_feature":
        return {
            "task_focus": "Planning focus: define the new player-facing capability, its owning integration point, and the neighboring systems that must stay stable.",
            "design_overview": "- Treat this as new feature work: define the player outcome, trigger conditions, and the gameplay boundaries around the new capability.",
            "design_behavior": "- Describe the new player-facing behavior, when it triggers, and which nearby states or systems must integrate cleanly.",
            "design_technical_note": "Capture the owning runtime hook, required gates or caps, and the integration surfaces the new path will touch.",
            "design_risk": "- Risk: the new feature can leak into adjacent systems if hooks, trigger conditions, or caps are left too broad.",
            "design_focus": "Focus the brief on player outcome, owning integration points, trigger rules, and the adjacent systems that need regression coverage.",
            "plan_overview": "- Introduce the requested gameplay behavior without destabilizing adjacent systems or generic shared hooks.",
            "plan_steps": [
                "Implement the new capability at the confirmed owner and keep its trigger conditions, caps, and state transitions explicit.",
                "Guard neighboring systems so the new behavior stays isolated to the intended gameplay path.",
            ],
            "validation_focus": "Cover the new feature path plus the closest non-triggering path that must remain unchanged.",
            "default_tests": [
                "Add a regression test proving the new feature activates only under the intended gameplay conditions.",
                "Add a regression test proving the closest neighboring path does not accidentally trigger the new behavior.",
            ],
            "risks": [
                "- Risk: the new capability may fire from the wrong state or bleed into nearby systems.",
                "- Mitigation: keep the hook narrow, gate the trigger conditions explicitly, and cover adjacent paths in tests.",
            ],
            "acceptance": [
                "- Players can trigger the new gameplay capability under the intended conditions.",
                "- Adjacent gameplay paths remain stable and automated coverage proves the new behavior stays gated correctly.",
            ],
        }
    if planning_mode == "non_gameplay":
        return {
            "task_focus": "Planning focus: this request does not belong to gameplay engineering.",
            "design_overview": "- This request appears outside gameplay ownership.",
            "design_behavior": "- Route the work to the owning non-gameplay discipline instead of producing a gameplay plan.",
            "design_technical_note": "No gameplay runtime owner should be assumed for this request.",
            "design_risk": "- Risk: a gameplay workflow could produce misleading implementation guidance for a task it does not own.",
            "design_focus": "Keep the document limited to ownership handoff and do not invent gameplay implementation work.",
            "plan_overview": "- Do not draft a gameplay implementation plan for non-gameplay-owned work.",
            "plan_steps": ["Route the task to the correct owner."],
            "validation_focus": "No gameplay validation should be proposed.",
            "default_tests": ["No gameplay tests should be proposed for non-gameplay-owned work."],
            "risks": ["- Risk: wrong-owner execution can waste time and create misleading artifacts."],
            "acceptance": ["- The task is routed away from gameplay engineering."],
        }
    return {
        "task_focus": "Planning focus: reproduce the issue, isolate the likely root cause, apply the narrowest safe fix, and prove the regression stays closed.",
        "design_overview": "- Treat this as bugfix work: capture the broken behavior, likely failing hook, and the smallest safe fix boundary.",
        "design_behavior": "- Describe the broken player-facing behavior, the intended behavior, and the neighboring path that must remain stable after the fix.",
        "design_technical_note": "Capture the failing owner, root-cause hypothesis, and validation path before implementation begins.",
        "design_risk": "- Risk: a broad fix could hide the symptom while shifting the regression into adjacent gameplay states.",
        "design_focus": "Focus the brief on repro, likely root cause, the smallest safe fix boundary, and the regression path that must be retested.",
        "plan_overview": "- Keep the fix as small as the grounded owner allows and avoid redesigning unrelated gameplay systems.",
        "plan_steps": [
            "Patch the smallest grounded hook or condition that explains the failure and keep adjacent state transitions unchanged.",
            "Add or update regression coverage for the failing path and the closest neighboring gameplay path that could drift.",
        ],
        "validation_focus": "Prove the original regression is closed and the closest neighboring path still behaves the same.",
        "default_tests": [
            "Add a regression test that reproduces the original gameplay failure and proves the fix closes it.",
            "Add a neighboring-path regression test so the narrow bugfix does not spill into adjacent gameplay states.",
        ],
        "risks": [
            "- Risk: the fix may mask the symptom while leaving the root cause or a neighboring regression behind.",
            "- Mitigation: anchor the fix on the grounded owner, keep the patch narrow, and cover the nearest adjacent path with tests.",
        ],
        "acceptance": [
            "- The original gameplay bug no longer reproduces on the grounded owner path.",
            "- The closest neighboring gameplay path remains stable and automated checks still pass.",
        ],
    }


def _planning_mode_owner_step(planning_mode: str, current_runtime_paths: list[str]) -> str:
    owner_path = current_runtime_paths[0] if current_runtime_paths else ""
    if planning_mode == "refactor":
        return (
            f"Anchor the refactor on {owner_path} and map the seams before moving code."
            if owner_path
            else "Confirm the live runtime owner, current seams, and invariants before refactoring."
        )
    if planning_mode == "improve_feature":
        return (
            f"Anchor the improvement on {owner_path} and compare the current behavior against the requested upgrade."
            if owner_path
            else "Confirm the live runtime owner and document the current feature behavior before improving it."
        )
    if planning_mode == "new_feature":
        return (
            f"Anchor the new capability on {owner_path} and identify the exact integration hook for the new behavior."
            if owner_path
            else "Confirm the owning gameplay runtime path and the exact integration hook before adding the new feature."
        )
    return (
        f"Reproduce the bug and anchor the fix on {owner_path}."
        if owner_path
        else "Confirm the owning gameplay runtime path and the failing hook before coding the fix."
    )


def _classify_task(context: WorkflowContext, task_prompt: str) -> tuple[str, str, str]:
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
    fallback_planning_mode = _infer_planning_mode(task_prompt, fallback_task_type=fallback_task_type)

    if non_gameplay_hits > gameplay_hits and gameplay_hits == 0:
        fallback_task_type = "non_gameplay"
        fallback_planning_mode = "non_gameplay"
        fallback_reason = "The prompt does not appear to be gameplay-owned work."
    else:
        fallback_reason = (
            f"The prompt is best handled as gameplay `{fallback_task_type}` work using the "
            f"`{_planning_mode_label(fallback_planning_mode)}` planning mode."
        )

    if context.llm.is_enabled():
        schema = {
            "type": "object",
            "properties": {
                "task_type": {"type": "string"},
                "planning_mode": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["task_type", "planning_mode", "reason"],
            "additionalProperties": False,
        }
        try:
            result = context.llm.generate_json(
                instructions=(
                    "Classify a gameplay-engineering request. Return a broad task_type and a detailed planning_mode. "
                    "task_type must be one of: bugfix, feature, maintenance, or non_gameplay. "
                    "planning_mode must be one of: bugfix, refactor, improve_feature, new_feature, or non_gameplay. "
                    "Map refactor work to task_type=maintenance. Map improve_feature and new_feature to task_type=feature. "
                    "Use non_gameplay only when the task is clearly outside gameplay ownership."
                ),
                input_text=build_prompt_brief(
                    opening="Decide how this gameplay-engineering request should be classified.",
                    sections=[("Request", task_prompt.strip())],
                    closing=(
                        "Choose the broad execution track and the more specific planning mode that best match the real gameplay ownership "
                        "and user intent."
                    ),
                ),
                schema_name="gameplay_task_classification",
                schema=schema,
            )
            task_type = str(result.get("task_type", fallback_task_type)).strip().lower()
            planning_mode = str(result.get("planning_mode", fallback_planning_mode)).strip().lower()
            if planning_mode in PLANNING_MODE_TO_TASK_TYPE and task_type not in {"bugfix", "feature", "maintenance", "non_gameplay"}:
                task_type = PLANNING_MODE_TO_TASK_TYPE[planning_mode]
            if task_type not in {"bugfix", "feature", "maintenance", "non_gameplay"}:
                task_type = fallback_task_type
            planning_mode = _normalize_planning_mode(task_type, planning_mode or fallback_planning_mode)
            reason = str(result.get("reason", fallback_reason)).strip() or fallback_reason
            return task_type, planning_mode, reason
        except Exception:
            return fallback_task_type, fallback_planning_mode, fallback_reason
    return fallback_task_type, fallback_planning_mode, fallback_reason


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
                            "\n".join(
                                [
                                    (
                                        f"This is investigation round {int(state.get('investigation_round', 0)) + 1} "
                                        f"for a {state.get('task_type', 'bugfix')} task."
                                    ),
                                    (
                                        f"Planning mode: {_planning_mode_label(state.get('planning_mode', _default_planning_mode(state.get('task_type', 'bugfix'))))}."
                                    ),
                                ]
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
            f"- Planning Mode: {state.get('planning_mode', _default_planning_mode(state['task_type']))}",
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
    planning_mode = _normalize_planning_mode(
        state["task_type"],
        str(state.get("planning_mode", _default_planning_mode(state["task_type"]))),
    )
    planning_profile = _planning_mode_profile(planning_mode)
    technical_notes = [
        *list(state.get("current_runtime_paths", [])),
        planning_profile["design_technical_note"],
    ]
    fallback = "\n".join(
        [
            "# Gameplay Design Context",
            "",
            "## Overview",
            f"- Request: {state['task_prompt']}",
            planning_profile["design_overview"],
            "",
            "## Existing References",
            _format_bullets(list(state.get("doc_hits", [])), empty_message="No strong design references were found."),
            "",
            "## Player-Facing Behavior",
            planning_profile["design_behavior"],
            "",
            "## Technical Notes",
            _format_bullets(technical_notes, empty_message="No runtime owner has been grounded yet."),
            "",
            "## Risks",
            planning_profile["design_risk"],
        ]
    )
    if context.llm.is_enabled():
        try:
            return context.llm.generate_text(
                instructions=(
                    "Write a concise markdown design context document with these exact sections: Overview, Existing References, "
                    "Player-Facing Behavior, Technical Notes, Risks. "
                    f"Treat the request as {_planning_mode_label(planning_mode)} work. {planning_profile['design_focus']}"
                ),
                input_text=build_prompt_brief(
                    opening="Prepare the gameplay design context that will ground planning and review.",
                    sections=[
                        ("Task request", state["task_prompt"].strip()),
                        (
                            "Planning frame",
                            "\n".join(
                                [
                                    f"- Task type: {state['task_type']}",
                                    f"- Planning mode: {planning_mode}",
                                ]
                            ),
                        ),
                        (
                            "Grounded references",
                            _format_bullets(list(state.get("doc_hits", []))),
                        ),
                        (
                            "Grounded runtime ownership",
                            _format_bullets(list(state.get("current_runtime_paths", []))),
                        ),
                    ],
                    closing=(
                        "Keep the design context concrete, scoped to gameplay ownership, explicit about nearby behavior that must remain stable, "
                        f"and aligned with this {_planning_mode_label(planning_mode)} planning mode."
                    ),
                ),
            )
        except Exception:
            return fallback
    return fallback


def _compose_plan_doc(context: WorkflowContext, state: EngineerState, *, revise: bool) -> str:
    planning_mode = _normalize_planning_mode(
        state["task_type"],
        str(state.get("planning_mode", _default_planning_mode(state["task_type"]))),
    )
    planning_profile = _planning_mode_profile(planning_mode)
    task_type_reason = state["classification_reason"] or f"This work is classified as {state['task_type']}."
    filtered_review_blocking_issues = _filter_plan_revision_items(list(state.get("review_blocking_issues", [])))
    filtered_review_improvement_actions = _filter_plan_revision_items(list(state.get("review_improvement_actions", [])))
    implementation_steps = [
        _planning_mode_owner_step(planning_mode, list(state.get("current_runtime_paths", []))),
        *list(planning_profile["plan_steps"]),
    ]
    unit_test_lines = list(state.get("test_hits", []))
    if unit_test_lines:
        unit_test_lines.append(planning_profile["validation_focus"])
    else:
        unit_test_lines = list(planning_profile["default_tests"])
    fallback = "\n".join(
        [
            "# Gameplay Implementation Plan",
            "",
            "## Overview",
            f"- {state['task_prompt']}",
            planning_profile["plan_overview"],
            "",
            "## Task Type",
            f"- {state['task_type']}",
            f"- Planning mode: {planning_mode}",
            f"- Classification reason: {task_type_reason}",
            planning_profile["task_focus"],
            "",
            "## Existing Docs",
            _format_bullets(list(state.get("doc_hits", [])), empty_message="No grounded docs found."),
            "",
            "## Implementation Steps",
            _format_bullets(implementation_steps),
            "",
            "## Unit Tests",
            _format_bullets(unit_test_lines),
            "",
            "## Risks",
            "\n".join(planning_profile["risks"]),
            "",
            "## Acceptance Criteria",
            "\n".join(planning_profile["acceptance"]),
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
                    "Unit Tests, Risks, Acceptance Criteria. "
                    f"Shape the plan for {_planning_mode_label(planning_mode)} work. {planning_profile['task_focus']}"
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
                                    f"- Planning mode: {planning_mode}",
                                    f"- Architecture review required: {state['requires_architecture_review']}",
                                ]
                            ),
                        ),
                        ("Mode-specific guidance", planning_profile["task_focus"]),
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
                        "do not add review-round bookkeeping, sign-off workflow, or artifact naming requirements. "
                        f"Stay aligned with this {_planning_mode_label(planning_mode)} planning mode."
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


def _evaluate_investigation_quality(
    context: WorkflowContext,
    state: EngineerState,
) -> tuple[dict[str, Any], list[InvestigationCheck]]:
    artifact_dir_value = str(state.get("artifact_dir", "")).strip()
    score_history_dir = Path(artifact_dir_value) if artifact_dir_value else None
    review_round = max(1, int(state.get("investigation_round", 1)))
    reviewer_llm = context.get_llm("reviewer")

    if not reviewer_llm.is_enabled():
        reason = "Reviewer LLM is unavailable, so gameplay investigation assessments cannot be generated."
        improvement_action = (
            "Enable the reviewer LLM and rerun gameplay investigation review so the scoring assessments come from LLM output."
        )
        feedback = _compose_loop_feedback(
            title="Investigation Confidence Review",
            round_index=review_round,
            score=0,
            threshold=INVESTIGATION_SCORE,
            approved=False,
            confidence_label="unmeasured",
            confidence_reason="MAD confidence is unavailable because no LLM-generated assessments were produced.",
            confidence=None,
            blocking_issues=[reason],
            improvement_actions=[improvement_action],
            sections=[],
            loop_reason=reason,
        )
        return (
            {
                "investigation_score": 0,
                "investigation_feedback": feedback,
                "investigation_missing_sections": [],
                "investigation_blocking_issues": [reason],
                "investigation_improvement_actions": [improvement_action],
                "investigation_approved": False,
                "investigation_loop_status": "llm-unavailable",
                "investigation_loop_reason": reason,
                "investigation_loop_stagnated_rounds": 0,
                "investigation_score_confidence": None,
                "investigation_score_confidence_label": "unmeasured",
                "investigation_score_confidence_reason": "MAD confidence is unavailable because no LLM-generated assessments were produced.",
                "active_loop_should_continue": False,
                "active_loop_completed": True,
                "active_loop_status": "llm-unavailable",
            },
            [],
        )

    schema = {
        "type": "object",
        "properties": {
            "feedback": {"type": "string"},
            "section_reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section": {"type": "string"},
                        "score": {"type": "integer"},
                        "status": {"type": "string"},
                        "rationale": {"type": "string"},
                        "action_items": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["section", "score", "status", "rationale", "action_items"],
                    "additionalProperties": False,
                },
            },
            "blocking_issues": {"type": "array", "items": {"type": "string"}},
            "improvement_actions": {"type": "array", "items": {"type": "string"}},
            "approved": {"type": "boolean"},
        },
        "required": ["feedback", "section_reviews", "blocking_issues", "improvement_actions", "approved"],
        "additionalProperties": False,
    }
    try:
        generated = reviewer_llm.generate_json(
            instructions=(
                "You are gameplay-engineer-workflow's strict investigation reviewer. Score the investigation only from the evidence provided. "
                "Return JSON with these exact section reviews: Supporting References, Runtime Owner Precision, Current vs Legacy Split, "
                "Ownership Summary, Root Cause Hypothesis, Investigation Summary, Implementation Medium, Validation Plan, Noise Control. "
                "Approval requires a score >= 85, no blocking issues, and a technically grounded handoff. "
                f"Minimum final-approval depth is {MIN_INVESTIGATION_ROUNDS} investigation rounds. "
                "Do not drift into process-only asks like sign-off, artifact naming, or bookkeeping. "
                "Keep the score numerically consistent with the section reviews."
            ),
            input_text=build_prompt_brief(
                opening="Review the current gameplay investigation as a strict senior engineer before planning or implementation can proceed.",
                sections=[
                    ("Task request", state["task_prompt"].strip()),
                    (
                        "Review context",
                        "\n".join(
                            [
                                f"- Investigation round: {review_round}",
                                f"- Task type: {state.get('task_type', 'bugfix')}",
                                (
                                    f"- Planning mode: {state.get('planning_mode', _default_planning_mode(state.get('task_type', 'bugfix')))}"
                                ),
                                f"- Execution track: {state.get('execution_track', state.get('task_type', 'bugfix'))}",
                                f"- Approval threshold: {INVESTIGATION_SCORE}/100",
                            ]
                        ),
                    ),
                    ("Investigation document", str(state.get("investigation_doc", "")).strip() or "None."),
                    ("Current runtime paths", _format_bullets(list(state.get("current_runtime_paths", [])))),
                    ("Legacy runtime paths", _format_bullets(list(state.get("legacy_runtime_paths", [])))),
                    ("Supporting docs", _format_bullets(list(state.get("doc_hits", [])))),
                    ("Supporting source files", _format_bullets(list(state.get("source_hits", [])))),
                    ("Supporting tests", _format_bullets(list(state.get("test_hits", [])))),
                    ("Supporting Blueprint assets", _format_bullets(list(state.get("blueprint_hits", [])))),
                    ("Supporting Blueprint text exports", _format_bullets(list(state.get("blueprint_text_hits", [])))),
                ],
                closing=(
                    "Score it hard, focus only on technical investigation quality, and require another independent verification pass "
                    "before final approval can stick."
                ),
            ),
            schema_name="gameplay_investigation_review",
            schema=schema,
        )
        raw_checks = [
            {
                "section": str(item["section"]).strip(),
                "score": max(0, int(item["score"])),
                "max_score": INVESTIGATION_SECTION_WEIGHT_MAP[str(item["section"]).strip()],
                "status": str(item["status"]).strip(),
                "rationale": str(item["rationale"]).strip(),
                "action_items": [str(action).strip() for action in item.get("action_items", []) if str(action).strip()],
            }
            for item in generated.get("section_reviews", [])
            if str(item.get("section", "")).strip() in INVESTIGATION_SECTION_WEIGHT_MAP
        ]
        expected_sections = {section for section, _ in INVESTIGATION_SECTION_WEIGHTS}
        received_sections = {item["section"] for item in raw_checks}
        if len(raw_checks) != len(INVESTIGATION_SECTION_WEIGHTS) or received_sections != expected_sections:
            raise ValueError("Reviewer LLM did not return the full gameplay investigation assessment set.")
        checks = [
            next(item for item in raw_checks if item["section"] == section)
            for section, _ in INVESTIGATION_SECTION_WEIGHTS
        ]
        blocking_issues = _filter_plan_revision_items(
            [str(item) for item in generated.get("blocking_issues", []) if str(item).strip()]
        )
        improvement_actions = _filter_plan_revision_items(
            [str(item) for item in generated.get("improvement_actions", []) if str(item).strip()]
        )
        explicit_approval = bool(generated.get("approved", False))
    except (LLMError, TypeError, ValueError) as exc:
        reason = f"Reviewer LLM failed to produce usable gameplay investigation assessments: {exc}"
        improvement_action = (
            "Fix reviewer LLM access or output formatting, then rerun gameplay investigation review so the scoring assessments come from LLM output."
        )
        feedback = _compose_loop_feedback(
            title="Investigation Confidence Review",
            round_index=review_round,
            score=0,
            threshold=INVESTIGATION_SCORE,
            approved=False,
            confidence_label="unmeasured",
            confidence_reason="MAD confidence is unavailable because no LLM-generated assessments were produced.",
            confidence=None,
            blocking_issues=[reason],
            improvement_actions=[improvement_action],
            sections=[],
            loop_reason=reason,
        )
        return (
            {
                "investigation_score": 0,
                "investigation_feedback": feedback,
                "investigation_missing_sections": [],
                "investigation_blocking_issues": [reason],
                "investigation_improvement_actions": [improvement_action],
                "investigation_approved": False,
                "investigation_loop_status": "llm-error",
                "investigation_loop_reason": reason,
                "investigation_loop_stagnated_rounds": 0,
                "investigation_score_confidence": None,
                "investigation_score_confidence_label": "unmeasured",
                "investigation_score_confidence_reason": "MAD confidence is unavailable because no LLM-generated assessments were produced.",
                "active_loop_should_continue": False,
                "active_loop_completed": True,
                "active_loop_status": "llm-error",
            },
            [],
        )

    if review_round < MIN_INVESTIGATION_ROUNDS:
        blocking_issues = _dedupe(
            [
                *blocking_issues,
                f"Minimum verification depth is {MIN_INVESTIGATION_ROUNDS} rounds, so this investigation still needs one more independent pass.",
            ]
        )
        improvement_actions = _dedupe([*improvement_actions, MANDATORY_INVESTIGATION_VERIFICATION_ACTION])
        explicit_approval = False

    score_decision = evaluate_score_decision(
        INVESTIGATION_SCORE_POLICY,
        round_index=review_round,
        assessments=_to_score_assessments(checks),
        explicit_approval=explicit_approval,
        blocking_issues=blocking_issues,
        improvement_actions=improvement_actions,
        artifact_dir=score_history_dir,
    )
    progress = evaluate_quality_loop(
        INVESTIGATION_LOOP_SPEC,
        round_index=review_round,
        score=score_decision.score,
        approved=score_decision.approved,
        blocking_issues=score_decision.blocking_issues,
        improvement_actions=score_decision.improvement_actions,
        previous_score=int(state.get("investigation_score", 0)) if review_round > 1 else None,
        prior_stagnated_rounds=int(state.get("investigation_loop_stagnated_rounds", 0)) if review_round > 1 else 0,
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
        loop_reason=str(generated.get("feedback", "")).strip() or progress.reason,
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
        task_type, planning_mode, reason = _classify_task(context, state["task_prompt"])
        gameplay_scope_verdict = "gameplay" if task_type != "non_gameplay" else "non_gameplay"
        implementation_requested = not _is_read_only_request(state["task_prompt"]) and task_type != "non_gameplay"
        execution_track = task_type if task_type in {"bugfix", "feature", "maintenance"} else "bugfix"
        return {
            "task_type": task_type,
            "planning_mode": planning_mode,
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
            "summary": (
                f"{metadata.name} classified the gameplay request as `{task_type}` with "
                f"`{planning_mode}` planning mode."
            ),
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
        updates, checks = _evaluate_investigation_quality(context, state)
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
                f"- Planning mode: {state.get('planning_mode', _default_planning_mode(state['task_type']))}",
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
            "task_type": str(state.get("task_type", "")),
            "planning_mode": str(state.get("planning_mode", "")),
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
                f"- Planning mode: {state.get('planning_mode', _default_planning_mode(state['task_type']))}",
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
            "task_type": str(state.get("task_type", "")),
            "planning_mode": str(state.get("planning_mode", "")),
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
                                        f"- Planning mode: {state.get('planning_mode', _default_planning_mode(state['task_type']))}",
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
                f"Task type: {state['task_type']}",
                f"Planning mode: {state.get('planning_mode', _default_planning_mode(state['task_type']))}",
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
            "task_type": str(state.get("task_type", "")),
            "planning_mode": str(state.get("planning_mode", "")),
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
            "task_type": str(state.get("task_type", "")),
            "planning_mode": str(state.get("planning_mode", "")),
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
            "task_type": str(state.get("task_type", "")),
            "planning_mode": str(state.get("planning_mode", "")),
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
