from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_logging import trace_graph_node, trace_route_decision
from core.llm import CodexCliLLMClient, LLMError
from core.executor import (
    ClaudeCodeExecutorClient,
    build_executor_system_prompt,
    build_executor_task_prompt,
)
from core.models import WorkflowContext, WorkflowMetadata
from core.text_utils import keyword_tokens, normalize_text, slugify, tokenize


APPROVAL_SCORE = 85
MIN_REVIEW_ROUNDS = 2
MAX_REVIEW_ROUNDS = 4
OPTIMIZATION_DOMAIN = "gamethread"
REVIEW_CRITERIA = (
    ("Problem Scoping", 15, "Task Framing"),
    ("Profiling Rigor", 25, "Profiling Evidence", "Hot Path Identification"),
    ("System-Specific Analysis", 20, "GAS & Ability Overhead", "AI System Load", "Physics Query Audit"),
    ("Optimization Quality", 20, "Batch & Throttle Opportunities", "Optimization Recommendations"),
    ("Risk & Regression", 10, "Regression Risk"),
    ("Verification Completeness", 10, "Verification Plan"),
)
INVESTIGATION_SECTIONS = (
    "Task Framing",
    "Profiling Evidence",
    "Hot Path Identification",
    "GAS & Ability Overhead",
    "AI System Load",
    "Physics Query Audit",
    "Batch & Throttle Opportunities",
    "Regression Risk",
    "Optimization Recommendations",
    "Verification Plan",
)
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".cpp",
    ".cs",
    ".h",
    ".hpp",
    ".ini",
    ".json",
    ".lua",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".usf",
    ".ush",
    ".yaml",
    ".yml",
}
_MAX_SUBDIRS_TO_SCAN = 10


class CriterionAssessment(TypedDict):
    criterion: str
    score: int
    max_score: int
    status: str
    rationale: str
    action_items: list[str]


class GameThreadInvestigationState(TypedDict):
    prompt: NotRequired[str]
    task_prompt: str
    task_id: NotRequired[str]
    run_dir: NotRequired[str]
    investigation_round: int
    review_round: int
    artifact_dir: str
    project_snapshot: str
    relevant_docs: list[str]
    relevant_source: list[str]
    relevant_tests: list[str]
    investigation_doc: str
    review_doc: str
    review_score: int
    review_feedback: str
    review_blocking_issues: list[str]
    review_improvement_actions: list[str]
    review_criterion_scores: list[CriterionAssessment]
    review_approved: bool
    loop_status: str
    loop_reason: str
    loop_should_continue: bool
    loop_completed: bool
    loop_stagnated_rounds: int
    final_report: dict[str, Any]
    summary: str
    review_criteria: NotRequired[list[tuple]]
    optimization_domain: NotRequired[str]


_MAX_PRIOR_CONTEXT_CHARS = 6000


def _truncate_prior_context(text: str, *, max_chars: int = _MAX_PRIOR_CONTEXT_CHARS) -> str:
    """Truncate long prior-round artifacts to keep the executor prompt manageable."""
    text = str(text).strip()
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated — full document available in artifacts ...]"


def _format_bullets(items: list[str], *, empty_message: str = "None.") -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return f"- {empty_message}"
    return "\n".join(f"- {item}" for item in cleaned)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in items:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _investigation_round_goal(
    *,
    investigation_round: int,
    review_feedback: str,
    previous_investigation: str,
) -> str:
    if investigation_round == 1:
        return "First pass. Identify the most expensive game thread consumers and build a profiling-backed hypothesis."
    if investigation_round < MIN_REVIEW_ROUNDS:
        return "Verification pass. Add fresh profiling evidence or call-frequency analysis before approval can stick."
    if review_feedback.strip():
        return "Verification pass. Explicitly answer the previous senior review with fresh performance evidence, not just a rewrite."
    if previous_investigation.strip():
        return "Verification pass. Independently re-check the current optimization hypothesis and tighten the cost proof."
    return "Verification pass. Re-validate the current hypothesis before final handoff."


def _investigation_pass_mandate(investigation_round: int) -> str:
    if investigation_round < MIN_REVIEW_ROUNDS:
        return (
            f"This workflow requires at least {MIN_REVIEW_ROUNDS} review rounds, so this pass must leave a clear path "
            "for an independent verification round instead of treating the first hypothesis as final. "
            "Include tick function analysis, GAS evaluation cost, and AI system load assessment."
        )
    return (
        "This pass must independently re-verify or falsify the previous hypothesis with at least one new piece of evidence: "
        "call frequency data, timer audit results, tick group analysis, or physics query counts."
    )


def _select_investigator_llm(context: WorkflowContext) -> tuple[Any, str]:
    executor = context.get_llm("executor")
    if isinstance(executor, ClaudeCodeExecutorClient) and executor.is_enabled():
        return executor, "claude-executor-tools"

    investigator_llm = context.get_llm("investigator")
    if isinstance(investigator_llm, CodexCliLLMClient):
        return investigator_llm.with_overrides(sandbox_mode="workspace-write"), "codex-agent-tools"
    return investigator_llm, "templated-llm"


def _short_slug(value: str, *, fallback: str, max_length: int = 18) -> str:
    slug = slugify(value, fallback=fallback)
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(normalize_text(value).encode("utf-8")).hexdigest()[:6]
    keep = max(1, max_length - len(digest) - 1)
    return f"{slug[:keep].rstrip('-') or fallback}-{digest}"


def _artifact_dir(context: WorkflowContext, metadata: WorkflowMetadata, state: GameThreadInvestigationState) -> Path:
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


def _narrow_roots(
    scope_root: Path,
    relative_roots: tuple[str, ...],
    exclude_roots: tuple[str, ...],
    query_tokens: set[str],
) -> list[Path]:
    """Score subdirectories by keyword overlap with the query and return only
    the top ``_MAX_SUBDIRS_TO_SCAN`` matches.  This avoids scanning thousands
    of files inside broad roots like ``Plugins/`` on large UE projects.

    Roots that are already narrow (fewer than ``_MAX_SUBDIRS_TO_SCAN`` direct
    children or no subdirectories at all) are returned as-is.
    """
    narrow: list[tuple[int, Path]] = []

    for rel_root in relative_roots:
        root = scope_root / rel_root
        if not root.exists():
            continue

        children: list[Path] = []
        try:
            children = [c for c in root.iterdir() if c.is_dir()]
        except OSError:
            pass

        if len(children) <= _MAX_SUBDIRS_TO_SCAN:
            narrow.append((0, root))
            continue

        scored_children: list[tuple[int, Path]] = []
        for child in children:
            if _should_skip(child, scope_root, exclude_roots):
                continue
            dir_tokens = tokenize(child.name)
            overlap = len(query_tokens & dir_tokens)
            scored_children.append((overlap, child))

            # Also check one level deeper (e.g. Plugins/Frameworks/SipherXxx)
            try:
                for grandchild in child.iterdir():
                    if not grandchild.is_dir():
                        continue
                    if _should_skip(grandchild, scope_root, exclude_roots):
                        continue
                    gc_tokens = tokenize(grandchild.name)
                    gc_overlap = len(query_tokens & gc_tokens)
                    if gc_overlap > overlap:
                        scored_children.append((gc_overlap, grandchild))
            except OSError:
                pass

        scored_children.sort(key=lambda item: -item[0])
        for score, child_path in scored_children[:_MAX_SUBDIRS_TO_SCAN]:
            narrow.append((score, child_path))

    if not narrow:
        return [scope_root]

    narrow.sort(key=lambda item: -item[0])
    return [path for _, path in narrow]


def _find_relevant_files(
    *,
    scope_root: Path,
    relative_roots: tuple[str, ...],
    exclude_roots: tuple[str, ...],
    query_text: str,
    max_hits: int = 5,
) -> list[str]:
    query_tokens = keyword_tokens(query_text) or tokenize(query_text)
    roots = _narrow_roots(scope_root, relative_roots, exclude_roots, query_tokens)
    scored: list[tuple[int, str]] = []

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if _should_skip(path, scope_root, exclude_roots):
                continue
            relative_path = path.relative_to(scope_root).as_posix()
            score = len(query_tokens & tokenize(f"{relative_path}\n{_safe_read_text(path)}"))
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


def _collect_project_context(
    context: WorkflowContext,
    task_prompt: str,
    review_feedback: str,
) -> dict[str, Any]:
    scope_root = context.resolve_scope_root("host_project")
    query_text = f"{task_prompt}\n{review_feedback}\ntick function game thread GAS ability AI physics query optimization".strip()
    source_roots = context.config.source_roots
    docs = _find_relevant_files(
        scope_root=scope_root,
        relative_roots=context.config.doc_roots,
        exclude_roots=context.config.exclude_roots,
        query_text=query_text,
    )
    source = _find_relevant_files(
        scope_root=scope_root,
        relative_roots=source_roots,
        exclude_roots=context.config.exclude_roots,
        query_text=query_text,
    )
    tests = _find_relevant_files(
        scope_root=scope_root,
        relative_roots=context.config.test_roots,
        exclude_roots=context.config.exclude_roots,
        query_text=query_text,
    )

    top_level: list[str] = []
    try:
        for child in sorted(scope_root.iterdir(), key=lambda item: item.name.lower()):
            if _should_skip(child, scope_root, context.config.exclude_roots):
                continue
            top_level.append(f"{child.name}{'/' if child.is_dir() else ''}")
            if len(top_level) >= 12:
                break
    except OSError:
        pass

    excerpts: list[str] = []
    for relative_path in [*docs[:2], *source[:2], *tests[:2]]:
        snippet = _safe_read_text(scope_root / relative_path, limit=400).strip()
        if snippet:
            excerpts.append(f"### {relative_path}\n{snippet}")

    snapshot = "\n".join(
        [
            "### Host Root Layout",
            _format_bullets(top_level, empty_message="No readable top-level entries found."),
            "",
            "### Candidate Docs",
            _format_bullets(docs, empty_message="No strong doc hits yet."),
            "",
            "### Candidate Source Files",
            _format_bullets(source, empty_message="No strong source hits yet."),
            "",
            "### Candidate Tests",
            _format_bullets(tests, empty_message="No strong test hits yet."),
            "",
            "### File Context",
            "\n\n".join(excerpts) or "No readable file excerpts were captured.",
        ]
    ).strip()
    return {"snapshot": snapshot, "docs": docs, "source": source, "tests": tests}


def _fallback_investigation_doc(
    *,
    task_prompt: str,
    investigation_round: int,
    project_snapshot: str,
    relevant_docs: list[str],
    relevant_source: list[str],
    relevant_tests: list[str],
    previous_investigation: str,
    review_feedback: str,
    improvement_actions: list[str],
) -> str:
    owners = relevant_source or relevant_docs or relevant_tests or ["No strong owner found yet; inspect tick functions first."]
    verification = relevant_tests or relevant_source or ["Add a focused performance regression check once hotspots are confirmed."]
    revision_goal = _investigation_round_goal(
        investigation_round=investigation_round,
        review_feedback=review_feedback,
        previous_investigation=previous_investigation,
    )
    lines = [
        "# Game Thread Optimization Investigation",
        "",
        "## Task Framing",
        f"- Round: {investigation_round}",
        f"- Request: {task_prompt}",
        f"- Revision goal: {revision_goal}",
        f"- Verification mandate: {_investigation_pass_mandate(investigation_round)}",
        "",
        "## Profiling Evidence",
        project_snapshot,
        "- Gather stat unit, stat game, and Unreal Insights data to identify game thread ms breakdown.",
        "- Look for TickComponent, TickActor, and timer callbacks that dominate the frame.",
        "",
        "## Hot Path Identification",
        *_format_bullets(owners).splitlines(),
        "- Identify the most expensive tick functions by searching for ::TickComponent, ::Tick, and SetActorTickEnabled patterns.",
        "- Count active actors per class to estimate per-frame cost.",
        "",
        "## GAS & Ability Overhead",
        "- Search for USipherAbilitySystemComponent tick and evaluation paths.",
        "- Check active GameplayEffect count and attribute replication frequency.",
        "- Look for AbilityLocalInputPressed/Confirmed patterns that may redundantly evaluate.",
        "",
        "## AI System Load",
        "- Analyze SipherAIScalableFramework for per-AI-agent tick cost.",
        "- Check Behavior Tree and State Tree evaluation frequency.",
        "- Look for perception system queries (AIPerception, EQS) running every frame.",
        "",
        "## Physics Query Audit",
        "- Search for SweepSingleByChannel, OverlapMultiByChannel, LineTraceSingleByChannel calls.",
        "- Count physics queries per frame and identify redundant or overly broad queries.",
        "- Check collision channel configuration for unnecessary complexity.",
        "",
        "## Batch & Throttle Opportunities",
        "- Identify candidates for TimerManager-based throttling instead of per-frame tick.",
        "- Look for significance-based tick rate opportunities (bAllowTickBeforeBeginPlay, tick intervals).",
        "- Check tick group distribution (TG_PrePhysics vs TG_DuringPhysics vs TG_PostPhysics).",
        "",
        "## Regression Risk",
        "- Assess what breaks if tick rates change or ability evaluation is deferred.",
        "- Identify frame-sensitive gameplay logic that depends on per-frame updates.",
        "- Check for animation-driven state machines that require consistent tick rates.",
        "",
        "## Optimization Recommendations",
        "- Rank changes by expected ms savings and implementation risk.",
        "- Provide concrete before/after expectations for each recommendation.",
        "",
        "## Verification Plan",
        *_format_bullets(verification, empty_message="Add a concrete perf test once hotspots are confirmed.").splitlines(),
        "- Use stat unit, stat game, and Insights markers to measure improvement.",
        "- Run automated performance baselines before and after changes.",
    ]
    if review_feedback.strip():
        lines.extend(["", "## Open Questions", "- Reviewer feedback still to satisfy:", *_format_bullets([review_feedback]).splitlines()])
    if improvement_actions:
        lines.extend(["", "## Open Questions" if "Open Questions" not in "\n".join(lines) else "", "- Reviewer checklist:", *_format_bullets(improvement_actions).splitlines()])
    return "\n".join(lines).strip()


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name
    reviewer_graph = context.get_workflow_graph("optimize-investigation-reviewer-workflow")

    def investigate(state: GameThreadInvestigationState) -> dict[str, Any]:
        investigation_round = int(state.get("investigation_round", 0)) + 1
        artifact_path = _artifact_dir(context, metadata, state)
        if investigation_round > 1 and str(state.get("project_snapshot", "")).strip():
            project_context = {
                "snapshot": state["project_snapshot"],
                "docs": list(state.get("relevant_docs", [])),
                "source": list(state.get("relevant_source", [])),
                "tests": list(state.get("relevant_tests", [])),
            }
        else:
            project_context = _collect_project_context(context, state["task_prompt"], str(state.get("review_feedback", "")))
        investigator_llm, investigation_mode = _select_investigator_llm(context)
        fallback_doc = _fallback_investigation_doc(
            task_prompt=state["task_prompt"],
            investigation_round=investigation_round,
            project_snapshot=project_context["snapshot"],
            relevant_docs=project_context["docs"],
            relevant_source=project_context["source"],
            relevant_tests=project_context["tests"],
            previous_investigation=str(state.get("investigation_doc", "")),
            review_feedback=str(state.get("review_feedback", "")),
            improvement_actions=list(state.get("review_improvement_actions", [])),
        )

        sections_str = ", ".join(INVESTIGATION_SECTIONS)
        investigation_doc = fallback_doc
        if investigation_mode == "claude-executor-tools" and isinstance(investigator_llm, ClaudeCodeExecutorClient):
            try:
                system_prompt = build_executor_system_prompt(
                    working_directory=str(context.host_root),
                    scope_constraints=[
                        "Investigate only — do not modify files, do not write patches.",
                        "Use Read, Grep, Glob, and Bash (read-only commands like stat, wc) to gather performance evidence.",
                        f"Write a markdown investigation brief with these sections: {sections_str}.",
                        "Focus on tick functions, GAS ability evaluation, AI system load, and physics query overhead.",
                        "Provide concrete numbers (actor counts, tick frequency, query counts) wherever possible.",
                        "CRITICAL: Your final output must be a CONCISE markdown brief under 4000 words. "
                        "Do NOT dump raw file contents or tool output. Summarize findings, cite file:line references, "
                        "and focus on actionable analysis. The brief will be reviewed by a separate LLM with limited context.",
                    ],
                )
                task_prompt = build_executor_task_prompt(
                    description=(
                        f"Investigate game thread performance like a senior performance engineer.\n\n"
                        f"Host project root: {context.host_root}\n"
                        f"Investigation round: {investigation_round}/{MAX_REVIEW_ROUNDS}\n\n"
                        f"Task: {state['task_prompt']}\n\n"
                        f"Round goal: {_investigation_round_goal(investigation_round=investigation_round, review_feedback=str(state.get('review_feedback', '')), previous_investigation=str(state.get('investigation_doc', '')))}\n\n"
                        f"Verification mandate: {_investigation_pass_mandate(investigation_round)}\n\n"
                        f"Game thread focus areas:\n"
                        f"- Tick functions: Search for ::TickComponent, ::Tick, SetActorTickEnabled, PrimaryActorTick\n"
                        f"- GAS: Search for USipherAbilitySystemComponent, GameplayEffect evaluation, ability activation paths\n"
                        f"- AI: Search for SipherAIScalableFramework tick, BehaviorTree evaluation, perception queries\n"
                        f"- Physics: Search for SweepSingle, OverlapMulti, LineTrace call sites and frequency\n\n"
                        f"Suggested docs: {_format_bullets(project_context['docs'], empty_message='None.')}\n"
                        f"Suggested source: {_format_bullets(project_context['source'], empty_message='None.')}\n"
                        f"Suggested tests: {_format_bullets(project_context['tests'], empty_message='None.')}\n\n"
                        f"Previous investigation:\n{_truncate_prior_context(state.get('investigation_doc', '') or 'None. First round.')}\n"
                    ),
                    prior_feedback=_truncate_prior_context(state.get("review_feedback", "")) or None,
                    context=project_context["snapshot"],
                )
                # Round 2+ has prior context, so fewer turns are needed. This prevents
                # the executor from timing out at 600s on large codebases.
                effective_max_turns = 15 if investigation_round > 1 else None
                result = investigator_llm.execute_task(
                    task_prompt=task_prompt,
                    system_prompt=system_prompt,
                    working_directory=str(context.host_root),
                    max_turns=effective_max_turns,
                )
                if result.success and result.result_text.strip():
                    investigation_doc = result.result_text
            except Exception:
                investigation_doc = fallback_doc
        elif investigator_llm.is_enabled():
            try:
                investigation_method = (
                    "Use the Codex agent tools available in this environment to inspect the project directly, read the most relevant "
                    "source files, and run targeted read-only commands that help prove the performance hypothesis. "
                    "Do not modify files, do not write patches, and do not invent command output. "
                    if investigation_mode == "codex-agent-tools"
                    else "Work only from the provided host-project context and previous review artifacts. Do not invent tool usage or command output. "
                )
                investigation_doc = investigator_llm.generate_text(
                    instructions=(
                        f"You are {metadata.name}. Investigate the host project's game thread performance like a senior "
                        f"performance engineer. Write a markdown investigation brief using this exact section order: {sections_str}. "
                        f"Stay concrete, evidence-driven, and strict about scope. {investigation_method}"
                        "Focus on tick functions, GAS evaluation, AI system load, and physics queries. "
                        "Provide concrete numbers (actor counts, call frequency, ms estimates) wherever possible. "
                        "If previous review feedback exists, address it explicitly. Do not use JSON."
                    ),
                    input_text=(
                        f"Host project root: {context.host_root}\n"
                        f"Investigation mode: {investigation_mode}\n"
                        f"Investigation round: {investigation_round}/{MAX_REVIEW_ROUNDS}\n\n"
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Round goal:\n{_investigation_round_goal(investigation_round=investigation_round, review_feedback=str(state.get('review_feedback', '')), previous_investigation=str(state.get('investigation_doc', '')))}\n\n"
                        f"Verification mandate:\n{_investigation_pass_mandate(investigation_round)}\n\n"
                        f"Minimum review rounds before final approval can stick: {MIN_REVIEW_ROUNDS}\n\n"
                        f"Suggested starting docs:\n{_format_bullets(project_context['docs'], empty_message='No strong doc hits yet.')}\n\n"
                        f"Suggested starting source files:\n{_format_bullets(project_context['source'], empty_message='No strong source hits yet.')}\n\n"
                        f"Suggested starting tests:\n{_format_bullets(project_context['tests'], empty_message='No strong test hits yet.')}\n\n"
                        f"Current project snapshot:\n{project_context['snapshot']}\n\n"
                        f"Previous investigation document:\n{_truncate_prior_context(state.get('investigation_doc', '') or 'None. This is the first round.')}\n\n"
                        f"Previous reviewer feedback:\n{_truncate_prior_context(state.get('review_feedback', '') or 'None. This is the first round.')}\n\n"
                        f"Previous reviewer checklist:\n{_format_bullets(list(state.get('review_improvement_actions', [])), empty_message='None.')}\n\n"
                        "Return only the next investigation document. The next document must add real evidence, not just rephrase the prior round."
                    ),
                )
            except LLMError:
                investigation_doc = fallback_doc

        (artifact_path / f"investigation_round_{investigation_round}.md").write_text(investigation_doc, encoding="utf-8")
        return {
            "investigation_round": investigation_round,
            "artifact_dir": str(artifact_path),
            "project_snapshot": project_context["snapshot"],
            "relevant_docs": project_context["docs"],
            "relevant_source": project_context["source"],
            "relevant_tests": project_context["tests"],
            "investigation_doc": investigation_doc,
            "review_criteria": [list(c) for c in REVIEW_CRITERIA],
            "optimization_domain": OPTIMIZATION_DOMAIN,
            "summary": f"{metadata.name} completed investigation round {investigation_round} and handed the brief to senior review.",
        }

    def request_review(state: GameThreadInvestigationState) -> dict[str, Any]:
        review_round = int(state.get("review_round", 0)) + 1
        return {
            "summary": (
                f"{metadata.name} handed investigation round {state['investigation_round']} to senior review "
                f"for review round {review_round}."
            )
        }

    def capture_review_result(state: GameThreadInvestigationState) -> dict[str, Any]:
        review_round = int(state.get("review_round", 0))
        artifact_path = _artifact_dir(context, metadata, state)
        final_status = "in_progress"
        if state["review_approved"]:
            final_status = "completed"
        elif state["loop_completed"]:
            final_status = "review-blocked"

        final_report = {
            "status": final_status,
            "investigation_rounds": int(state.get("investigation_round", 0)),
            "review_rounds": review_round,
            "review_score": int(state.get("review_score", 0)),
            "review_approved": bool(state.get("review_approved", False)),
            "loop_status": str(state.get("loop_status", "unknown")),
            "loop_reason": str(state.get("loop_reason", "")),
            "loop_stagnated_rounds": int(state.get("loop_stagnated_rounds", 0)),
            "artifact_dir": str(artifact_path),
            "relevant_docs": list(state.get("relevant_docs", [])),
            "relevant_source": list(state.get("relevant_source", [])),
            "relevant_tests": list(state.get("relevant_tests", [])),
            "blocking_issues": list(state.get("review_blocking_issues", [])),
            "improvement_actions": list(state.get("review_improvement_actions", [])),
        }

        summary = (
            f"{metadata.name} passed senior review in {review_round} round(s) with score {state['review_score']}/100."
            if state["review_approved"]
            else (
                f"{metadata.name} stopped after {review_round} round(s) with score {state['review_score']}/100. Loop status: {state['loop_status']}."
                if state["loop_completed"]
                else f"{metadata.name} scored {state['review_score']}/100 in review round {review_round} and will loop back into investigation."
            )
        )
        (artifact_path / "final_report.md").write_text(
            "\n".join(
                [
                    "# Game Thread Optimization Investigation Final Report",
                    "",
                    f"- Status: {final_status}",
                    f"- Review Score: {state['review_score']}/100",
                    f"- Review Approved: {state['review_approved']}",
                    f"- Loop Status: {state['loop_status']}",
                    f"- Loop Reason: {state['loop_reason']}",
                    "",
                    "## Blocking Issues",
                    *([f"- {item}" for item in state.get("review_blocking_issues", [])] or ["- None."]),
                    "",
                    "## Improvement Checklist",
                    *([f"- {item}" for item in state.get("review_improvement_actions", [])] or ["- None."]),
                    "",
                    "## Latest Review",
                    state.get("review_doc", ""),
                ]
            ),
            encoding="utf-8",
        )
        return {
            "artifact_dir": str(artifact_path),
            "final_report": final_report,
            "summary": summary,
        }

    def review_gate(state: GameThreadInvestigationState) -> str:
        return "investigate" if state["loop_should_continue"] else END

    graph = StateGraph(GameThreadInvestigationState)
    graph.add_node("investigate", trace_graph_node(graph_name=graph_name, node_name="investigate", node_fn=investigate))
    graph.add_node(
        "request_review",
        trace_graph_node(graph_name=graph_name, node_name="request_review", node_fn=request_review),
    )
    graph.add_node("optimize-investigation-reviewer-workflow", reviewer_graph)
    graph.add_node(
        "capture_review_result",
        trace_graph_node(graph_name=graph_name, node_name="capture_review_result", node_fn=capture_review_result),
    )
    graph.add_edge(START, "investigate")
    graph.add_edge("investigate", "request_review")
    graph.add_edge("request_review", "optimize-investigation-reviewer-workflow")
    graph.add_edge("optimize-investigation-reviewer-workflow", "capture_review_result")
    graph.add_conditional_edges(
        "capture_review_result",
        trace_route_decision(graph_name=graph_name, router_name="review_gate", route_fn=review_gate),
        {"investigate": "investigate", END: END},
    )
    return graph.compile()
