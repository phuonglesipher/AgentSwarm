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


APPROVAL_SCORE = 90
MIN_REVIEW_ROUNDS = 2
MAX_REVIEW_ROUNDS = 5
TEXT_SUFFIXES = {
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
    ".md",
    ".py",
    ".rb",
    ".rs",
    ".rst",
    ".sh",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
class CriterionAssessment(TypedDict):
    criterion: str
    score: int
    max_score: int
    status: str
    rationale: str
    action_items: list[str]


class InvestigationLoopState(TypedDict):
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
    review_missing_sections: list[str]
    review_criterion_scores: list[CriterionAssessment]
    review_actionable_feedback: str
    review_approved: bool
    loop_status: str
    loop_reason: str
    loop_should_continue: bool
    loop_completed: bool
    loop_stagnated_rounds: int
    final_report: dict[str, Any]
    summary: str


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
        return "First pass. Build the strongest initial hypothesis and identify exactly what still needs proof."
    if investigation_round < MIN_REVIEW_ROUNDS:
        return "Verification pass. Add fresh evidence or a read-only reproduction before approval can stick."
    if review_feedback.strip():
        return "Verification pass. Explicitly answer the previous senior review with fresh evidence, not just a rewrite."
    if previous_investigation.strip():
        return "Verification pass. Independently re-check the current hypothesis and tighten the causal proof."
    return "Verification pass. Re-validate the current hypothesis before final handoff."


def _investigation_pass_mandate(
    investigation_round: int,
    weak_criteria: list[str] | None = None,
) -> str:
    if investigation_round < MIN_REVIEW_ROUNDS:
        base = (
            f"This workflow requires at least {MIN_REVIEW_ROUNDS} review rounds, so this pass must leave a clear path "
            "for an independent verification round instead of treating the first hypothesis as final. "
            "Include a broad search for external consumers/callers of any identified APIs."
        )
    else:
        base = (
            "This pass must independently re-verify or falsify the previous hypothesis with at least one new piece of evidence, "
            "clearer sequencing proof, a read-only command/test observation, or proof of caller presence/absence for the identified APIs."
        )

    if not weak_criteria:
        return base

    emphasis: list[str] = []
    for criterion in weak_criteria:
        if "evidence" in criterion.lower() or "ownership" in criterion.lower():
            emphasis.append("Priority: search for external consumers/callers with grep evidence.")
        elif "verification" in criterion.lower():
            emphasis.append("Priority: propose concrete reproduction steps or test commands.")
        elif "focus" in criterion.lower():
            emphasis.append("Priority: tighten scope — remove tangential sections, sharpen hypothesis.")
        elif "architecture" in criterion.lower():
            emphasis.append("Priority: clarify the system boundary and ownership handoff.")
    if emphasis:
        return f"{base}\n\n" + " ".join(emphasis)
    return base


def _select_investigator_llm(context: WorkflowContext) -> tuple[Any, str]:
    # Prefer executor client for full tool access (Edit, Read, Bash, Grep).
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


_MAX_PRIOR_CONTEXT_CHARS = 8000
_PRIORITY_SECTIONS = ("Root Cause Hypothesis", "Consumer & Caller Analysis", "Verification Plan")


def _truncate_prior_context(text: str, *, max_chars: int = _MAX_PRIOR_CONTEXT_CHARS) -> str:
    """Truncate long prior-round artifacts, preserving high-value sections.

    Priority sections (hypothesis, caller evidence, verification plan) are
    kept in full before filling the remaining budget with other content.
    """
    text = str(text).strip()
    if not text or len(text) <= max_chars:
        return text

    # Extract high-value sections first
    kept_sections: list[str] = []
    priority_budget = 0
    for section in _PRIORITY_SECTIONS:
        block = "\n".join(_extract_heading_block(text, section)).strip()
        if block:
            section_text = f"## {section}\n{block}"
            kept_sections.append(section_text)
            priority_budget += len(section_text) + 2  # +2 for separator newlines

    remaining_budget = max_chars - priority_budget
    if remaining_budget > 500:
        # Fill rest of budget with beginning of document (contains Task Framing, etc.)
        prefix = text[:remaining_budget].rstrip()
        parts = [prefix, *kept_sections]
    else:
        parts = kept_sections

    return "\n\n".join(parts) + "\n\n[... truncated — full document in artifacts ...]"


def _artifact_dir(context: WorkflowContext, metadata: WorkflowMetadata, state: InvestigationLoopState) -> Path:
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


def _resolve_roots(scope_root: Path, relative_roots: tuple[str, ...]) -> list[Path]:
    candidates = [scope_root / relative_root for relative_root in relative_roots]
    existing = [path for path in candidates if path.exists()]
    return existing or [scope_root]


def _find_relevant_files(
    *,
    scope_root: Path,
    relative_roots: tuple[str, ...],
    exclude_roots: tuple[str, ...],
    query_text: str,
    max_hits: int = 5,
) -> list[str]:
    query_tokens = keyword_tokens(query_text) or tokenize(query_text)
    scored: list[tuple[int, str]] = []

    for root in _resolve_roots(scope_root, relative_roots):
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

    if hits:
        return hits

    fallback_hits: list[str] = []
    for root in _resolve_roots(scope_root, relative_roots):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if _should_skip(path, scope_root, exclude_roots):
                continue
            fallback_hits.append(path.relative_to(scope_root).as_posix())
            if len(fallback_hits) >= max_hits:
                return fallback_hits
    return fallback_hits


def _collect_project_context(
    context: WorkflowContext,
    task_prompt: str,
    review_feedback: str,
    improvement_actions: list[str] | None = None,
) -> dict[str, Any]:
    scope_root = context.resolve_scope_root("host_project")
    # Use improvement action keywords for more focused file discovery
    action_text = " ".join(improvement_actions or [])
    query_text = f"{task_prompt}\n{action_text}".strip() if action_text else f"{task_prompt}\n{review_feedback}".strip()
    docs = _find_relevant_files(
        scope_root=scope_root,
        relative_roots=context.config.doc_roots,
        exclude_roots=context.config.exclude_roots,
        query_text=query_text,
    )
    source = _find_relevant_files(
        scope_root=scope_root,
        relative_roots=context.config.source_roots,
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
    investigation_mode: str = "claude-executor-tools",
    project_snapshot: str,
    relevant_docs: list[str],
    relevant_source: list[str],
    relevant_tests: list[str],
    previous_investigation: str,
    weak_criteria: list[str] | None = None,
    review_feedback: str,
    improvement_actions: list[str],
) -> str:
    owners = relevant_source or relevant_docs or relevant_tests or ["No strong owner found yet; inspect the most likely entrypoint first."]
    verification = relevant_tests or relevant_source or ["Add a focused regression check once ownership is confirmed."]
    revision_goal = _investigation_round_goal(
        investigation_round=investigation_round,
        review_feedback=review_feedback,
        previous_investigation=previous_investigation,
    )
    lines = [
        "# Template Investigation",
        "",
        "## Task Framing",
        f"- Round: {investigation_round}",
        f"- Request: {task_prompt}",
        f"- Revision goal: {revision_goal}",
        f"- Verification mandate: {_investigation_pass_mandate(investigation_round, weak_criteria=weak_criteria)}",
        "",
        "## Project Root Findings",
        project_snapshot,
        "",
        "## Candidate Ownership",
        *_format_bullets(owners).splitlines(),
        "",
        "## Consumer & Caller Analysis",
        *(
            [
                "- Search broadly for external callers of the APIs/functions identified above.",
                "- Prove whether callers exist outside the owning module (grep for function names across the codebase).",
                "- If no external callers found, state that explicitly — absence of callers is high-value evidence.",
            ]
            if investigation_mode != "templated-llm"
            else [
                "- Based on the provided project context, identify likely callers of the candidate APIs.",
                "- State whether external callers are expected based on the module's public API surface.",
                "- If the context suggests no external callers, state that explicitly — it constrains the fix scope.",
            ]
        ),
        "",
        "## Root Cause Hypothesis",
        "- The most credible next step is to inspect the candidate owner first and validate the runtime handoff.",
        "- Treat the current findings as an evidence-backed hypothesis, not a final diagnosis.",
        "",
        "## Architecture Notes",
        "- Focus on the boundary between the likely owner, its caller, and the validating tests.",
        "- Prefer the narrowest subsystem that explains the symptom before touching adjacent architecture.",
        "",
        "## Clean Code Notes",
        "- Prefer the smallest safe change that keeps ownership obvious and avoids new coupling.",
        "- Keep validation and intent close to the affected runtime path.",
        "",
        "## Optimization Notes",
        "- Do not optimize blindly; first confirm the hot path or repeated work from the owning code path.",
        "- If no hotspot is evident, prioritize correctness and clear ownership over speculative tuning.",
        "",
        "## Verification Plan",
        *_format_bullets(verification, empty_message="Add a concrete regression test once ownership is clearer.").splitlines(),
        "- Reproduce the issue against the suspected owner before broadening the search.",
        "",
        "## Open Questions",
    ]
    if review_feedback.strip():
        lines.extend(["- Reviewer feedback still to satisfy:", *_format_bullets([review_feedback]).splitlines()])
    if improvement_actions:
        lines.extend(["- Reviewer checklist:", *_format_bullets(improvement_actions).splitlines()])
    if previous_investigation.strip():
        lines.append("- Compare this round with the previous investigation so the hypothesis narrows instead of drifting.")
    if not review_feedback.strip() and not previous_investigation.strip():
        lines.append("- Confirm whether the first suspected owner is truly responsible before widening to neighboring modules.")
    return "\n".join(lines).strip()


def _extract_heading_block(document: str, heading: str) -> list[str]:
    lines = document.splitlines()
    active = False
    collected: list[str] = []
    for line in lines:
        if line.strip() == f"## {heading}":
            active = True
            continue
        if active and line.startswith("## "):
            break
        if active and line.strip():
            collected.append(line.rstrip())
    return collected


def _build_actionable_feedback(state: dict[str, Any]) -> str:
    """Build a concise, actionable feedback string from structured review state.

    Strips out scoring metadata (confidence, loop status) and keeps only
    information the investigator needs to improve the next round.
    """
    parts: list[str] = []

    blocking_issues = list(state.get("review_blocking_issues", []))
    if blocking_issues:
        parts.append("## Blocking Issues")
        parts.extend(f"- {item}" for item in blocking_issues)

    improvement_actions = list(state.get("review_improvement_actions", []))
    if improvement_actions:
        parts.append("\n## Required Improvements")
        parts.extend(f"- {item}" for item in improvement_actions)

    missing_sections = list(state.get("review_missing_sections", []))
    if missing_sections:
        parts.append("\n## Missing Sections (must add)")
        parts.extend(f"- {section}" for section in missing_sections)

    criterion_scores: list[CriterionAssessment] = list(state.get("review_criterion_scores", []))
    weak = [c for c in criterion_scores if c["score"] < c["max_score"] * 0.7]
    if weak:
        parts.append("\n## Weak Criteria (focus here)")
        for c in weak:
            parts.append(f"- {c['criterion']}: {c['score']}/{c['max_score']} — {c['rationale']}")

    senior_notes = "\n".join(_extract_heading_block(
        state.get("review_doc", ""), "Senior Engineer Notes"
    )).strip()
    if senior_notes:
        parts.append(f"\n## Senior Engineer Notes\n{senior_notes}")

    return "\n".join(parts).strip() or "No actionable feedback."


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name
    reviewer_graph = context.get_workflow_graph("template-investigation-reviewer-workflow")

    def investigate(state: InvestigationLoopState) -> dict[str, Any]:
        investigation_round = int(state.get("investigation_round", 0)) + 1
        artifact_path = _artifact_dir(context, metadata, state)

        # Extract weak criteria names for mandate emphasis
        criterion_scores: list[CriterionAssessment] = list(state.get("review_criterion_scores", []))
        weak_criteria = [c["criterion"] for c in criterion_scores if c["score"] < c["max_score"] * 0.7]

        improvement_actions = list(state.get("review_improvement_actions", []))

        # Force context refresh when reviewer flagged specific areas to search
        should_refresh = investigation_round == 1
        if investigation_round > 1 and improvement_actions:
            actions_text = " ".join(improvement_actions).lower()
            if any(kw in actions_text for kw in ("search", "grep", "caller", "consumer", "module", "file")):
                should_refresh = True

        if not should_refresh and str(state.get("project_snapshot", "")).strip():
            project_context = {
                "snapshot": state["project_snapshot"],
                "docs": list(state.get("relevant_docs", [])),
                "source": list(state.get("relevant_source", [])),
                "tests": list(state.get("relevant_tests", [])),
            }
        else:
            project_context = _collect_project_context(
                context, state["task_prompt"],
                str(state.get("review_feedback", "")),
                improvement_actions=improvement_actions,
            )
        investigator_llm, investigation_mode = _select_investigator_llm(context)
        fallback_doc = _fallback_investigation_doc(
            task_prompt=state["task_prompt"],
            investigation_round=investigation_round,
            investigation_mode=investigation_mode,
            project_snapshot=project_context["snapshot"],
            relevant_docs=project_context["docs"],
            relevant_source=project_context["source"],
            relevant_tests=project_context["tests"],
            previous_investigation=str(state.get("investigation_doc", "")),
            review_feedback=str(state.get("review_actionable_feedback", "") or state.get("review_feedback", "")),
            improvement_actions=list(state.get("review_improvement_actions", [])),
            weak_criteria=weak_criteria,
        )

        investigation_doc = fallback_doc
        if investigation_mode == "claude-executor-tools" and isinstance(investigator_llm, ClaudeCodeExecutorClient):
            try:
                system_prompt = build_executor_system_prompt(
                    working_directory=str(context.host_root),
                    scope_constraints=[
                        "Investigate only — do not modify files, do not write patches.",
                        "Use Read, Grep, Glob, and Bash (read-only commands) to gather evidence.",
                        "Write a markdown investigation brief with these sections: Task Framing, Project Root Findings, "
                        "Candidate Ownership, Consumer & Caller Analysis, Root Cause Hypothesis, Architecture Notes, "
                        "Clean Code Notes, Optimization Notes, Verification Plan, Open Questions.",
                    ],
                )
                task_prompt = build_executor_task_prompt(
                    description=(
                        f"Investigate the host project like a senior engineer converging on root cause and owner.\n\n"
                        f"Host project root: {context.host_root}\n"
                        f"Investigation round: {investigation_round}/{MAX_REVIEW_ROUNDS}\n\n"
                        f"Task: {state['task_prompt']}\n\n"
                        f"Round goal: {_investigation_round_goal(investigation_round=investigation_round, review_feedback=str(state.get('review_feedback', '')), previous_investigation=str(state.get('investigation_doc', '')))}\n\n"
                        f"Verification mandate: {_investigation_pass_mandate(investigation_round, weak_criteria=weak_criteria)}\n\n"
                        f"Consumer analysis mandate: Search broadly (Grep, Glob) for external callers of any APIs or "
                        f"functions identified as candidates. Prove presence or absence of callers outside the owning "
                        f"module. If no external callers exist, state that explicitly — it changes optimization priority.\n\n"
                        f"Suggested docs: {_format_bullets(project_context['docs'], empty_message='None.')}\n"
                        f"Suggested source: {_format_bullets(project_context['source'], empty_message='None.')}\n"
                        f"Suggested tests: {_format_bullets(project_context['tests'], empty_message='None.')}\n\n"
                        f"Previous investigation:\n{_truncate_prior_context(state.get('investigation_doc', '') or 'None. First round.')}\n"
                    ),
                    prior_feedback=_truncate_prior_context(state.get("review_actionable_feedback", "") or state.get("review_feedback", "")) or None,
                    context=project_context["snapshot"],
                )
                result = investigator_llm.execute_task(
                    task_prompt=task_prompt,
                    system_prompt=system_prompt,
                    working_directory=str(context.host_root),
                )
                if result.success and result.result_text.strip():
                    investigation_doc = result.result_text
            except Exception:
                investigation_doc = fallback_doc
        elif investigator_llm.is_enabled():
            try:
                if investigation_mode == "codex-agent-tools":
                    investigation_method = (
                        "Use the Codex agent tools available in this environment to inspect the project directly, read the most relevant "
                        "source/docs/tests, and when it increases confidence, run targeted read-only commands or tests that help prove "
                        "the causal chain. Do not modify files, do not write patches, and do not invent command output. "
                    )
                    consumer_mandate = (
                        "Consumer analysis mandate: Search broadly (Grep, Glob) for external callers of any APIs or "
                        "functions identified as candidates. Prove presence or absence of callers outside the owning "
                        "module. If no external callers exist, state that explicitly — it changes optimization priority."
                    )
                else:
                    investigation_method = (
                        "Work only from the provided host-project context and previous review artifacts. "
                        "Do not invent tool usage or command output. "
                    )
                    consumer_mandate = (
                        "Consumer analysis mandate: Based on the provided project context, identify likely callers of the candidate APIs. "
                        "State whether external callers are expected based on the module's public API surface. "
                        "If the context suggests no external callers, state that explicitly — it constrains the fix scope."
                    )
                investigation_doc = investigator_llm.generate_text(
                    instructions=(
                        "You are template-investigation-workflow. Investigate the host project root like a senior engineer "
                        "trying to converge on the most credible root cause and owner. Write a markdown investigation brief using "
                        "this exact section order: Task Framing, Project Root Findings, Candidate Ownership, Consumer & Caller Analysis, "
                        "Root Cause Hypothesis, Architecture Notes, Clean Code Notes, Optimization Notes, Verification Plan, Open Questions. "
                        f"Stay concrete, evidence-driven, and strict about scope. {investigation_method}"
                        "If previous review feedback exists, address it explicitly. Do not use JSON."
                    ),
                    input_text=(
                        f"Host project root: {context.host_root}\n"
                        f"Investigation mode: {investigation_mode}\n"
                        f"Investigation round: {investigation_round}/{MAX_REVIEW_ROUNDS}\n\n"
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Round goal:\n{_investigation_round_goal(investigation_round=investigation_round, review_feedback=str(state.get('review_feedback', '')), previous_investigation=str(state.get('investigation_doc', '')))}\n\n"
                        f"Verification mandate:\n{_investigation_pass_mandate(investigation_round, weak_criteria=weak_criteria)}\n\n"
                        f"{consumer_mandate}\n\n"
                        f"Minimum review rounds before final approval can stick: {MIN_REVIEW_ROUNDS}\n\n"
                        f"Suggested starting docs:\n{_format_bullets(project_context['docs'], empty_message='No strong doc hits yet.')}\n\n"
                        f"Suggested starting source files:\n{_format_bullets(project_context['source'], empty_message='No strong source hits yet.')}\n\n"
                        f"Suggested starting tests:\n{_format_bullets(project_context['tests'], empty_message='No strong test hits yet.')}\n\n"
                        f"Current project snapshot:\n{project_context['snapshot']}\n\n"
                        f"Previous investigation document:\n{_truncate_prior_context(state.get('investigation_doc', '') or 'None. This is the first round.')}\n\n"
                        f"Previous reviewer feedback (actionable only):\n{_truncate_prior_context(state.get('review_actionable_feedback', '') or 'None. This is the first round.')}\n\n"
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
            "summary": f"{metadata.name} completed investigation round {investigation_round} and handed the brief to senior review.",
        }

    def request_review(state: InvestigationLoopState) -> dict[str, Any]:
        review_round = int(state.get("review_round", 0)) + 1
        return {
            "summary": (
                f"{metadata.name} handed investigation round {state['investigation_round']} to senior review "
                f"for review round {review_round}."
            )
        }

    def capture_review_result(state: InvestigationLoopState) -> dict[str, Any]:
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
            "missing_sections": list(state.get("review_missing_sections", [])),
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
                    "# Template Investigation Final Report",
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
        actionable_feedback = _build_actionable_feedback(state)
        return {
            "artifact_dir": str(artifact_path),
            "review_actionable_feedback": actionable_feedback,
            "final_report": final_report,
            "summary": summary,
        }

    def review_gate(state: InvestigationLoopState) -> str:
        return "investigate" if state["loop_should_continue"] else END

    graph = StateGraph(InvestigationLoopState)
    graph.add_node("investigate", trace_graph_node(graph_name=graph_name, node_name="investigate", node_fn=investigate))
    graph.add_node(
        "request_review",
        trace_graph_node(graph_name=graph_name, node_name="request_review", node_fn=request_review),
    )
    graph.add_node("template-investigation-reviewer-workflow", reviewer_graph)
    graph.add_node(
        "capture_review_result",
        trace_graph_node(graph_name=graph_name, node_name="capture_review_result", node_fn=capture_review_result),
    )
    graph.add_edge(START, "investigate")
    graph.add_edge("investigate", "request_review")
    graph.add_edge("request_review", "template-investigation-reviewer-workflow")
    graph.add_edge("template-investigation-reviewer-workflow", "capture_review_result")
    graph.add_conditional_edges(
        "capture_review_result",
        trace_route_decision(graph_name=graph_name, router_name="review_gate", route_fn=review_gate),
        {"investigate": "investigate", END: END},
    )
    return graph.compile()
