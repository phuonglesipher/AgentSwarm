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
from core.tool_engine import ToolEngine, ToolEngineConfig


APPROVAL_SCORE = 85
MIN_REVIEW_ROUNDS = 2
MAX_REVIEW_ROUNDS = 4
CRASH_DOMAIN = "crash-stability"
MEMORY_NAMESPACE = "crash_analysis"
REVIEW_CRITERIA = (
    ("Crash Identification", 10, "Crash Context"),
    ("Evidence Analysis", 25, "Call Stack Analysis", "Memory State Assessment"),
    ("Root Cause Rigor", 25, "Root Cause Hypothesis", "Reproduction Strategy"),
    ("Domain Classification", 10, "Domain Classification"),
    ("Fix Quality", 15, "Fix Recommendations"),
    ("Verification Completeness", 15, "Verification Plan"),
)
INVESTIGATION_SECTIONS = (
    "Crash Context",
    "Call Stack Analysis",
    "Memory State Assessment",
    "Root Cause Hypothesis",
    "Reproduction Strategy",
    "Domain Classification",
    "Fix Recommendations",
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


class CrashInvestigationState(TypedDict):
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
    crash_report: str


_MAX_PRIOR_CONTEXT_CHARS = 6000


def _truncate_prior_context(text: str, *, max_chars: int = _MAX_PRIOR_CONTEXT_CHARS) -> str:
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
        return (
            "First pass. Identify crash type, analyze relevant code paths, "
            "classify domain, and build evidence-backed root cause hypothesis."
        )
    if investigation_round < MIN_REVIEW_ROUNDS:
        return "Verification pass. Add independent evidence before approval can stick."
    if review_feedback.strip():
        return (
            "Verification pass. Explicitly answer the previous senior review "
            "with fresh crash evidence, not just a rewrite."
        )
    if previous_investigation.strip():
        return "Verification pass. Re-validate root cause hypothesis before final handoff."
    return "Verification pass. Re-validate the crash hypothesis before final handoff."


def _investigation_pass_mandate(investigation_round: int) -> str:
    if investigation_round < MIN_REVIEW_ROUNDS:
        return (
            f"This workflow requires at least {MIN_REVIEW_ROUNDS} review rounds. "
            "Include call stack analysis, memory state review, domain classification, "
            "and reproduction strategy."
        )
    return (
        "Re-verify or falsify the previous crash hypothesis with at least one "
        "new piece of evidence: code path trace, memory pattern, threading "
        "analysis, or reproduction result."
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


def _artifact_dir(context: WorkflowContext, metadata: WorkflowMetadata, state: CrashInvestigationState) -> Path:
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
    query_text = (
        f"{task_prompt}\n{review_feedback}\n"
        "crash callstack stack trace memory corruption access violation "
        "GC garbage collection TDR GPU device removal thread assert check ensure"
    ).strip()
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
    project_snapshot: str,
    relevant_docs: list[str],
    relevant_source: list[str],
    relevant_tests: list[str],
    previous_investigation: str,
    review_feedback: str,
    improvement_actions: list[str],
) -> str:
    owners = relevant_source or relevant_docs or relevant_tests or ["No strong owner found yet; inspect crash logs first."]
    verification = relevant_tests or relevant_source or ["Add a crash reproduction test once root cause is confirmed."]
    revision_goal = _investigation_round_goal(
        investigation_round=investigation_round,
        review_feedback=review_feedback,
        previous_investigation=previous_investigation,
    )
    lines = [
        "# Crash Investigation",
        "",
        "## Crash Context",
        f"- Round: {investigation_round}",
        f"- Request: {task_prompt}",
        f"- Revision goal: {revision_goal}",
        f"- Verification mandate: {_investigation_pass_mandate(investigation_round)}",
        "",
        project_snapshot,
        "",
        "## Call Stack Analysis",
        *_format_bullets(owners).splitlines(),
        "- Search for check(), ensure(), checkf(), verifyf() patterns.",
        "- Look for null pointer dereference patterns in crash vicinity.",
        "- Identify the crash module and function from stack frames.",
        "",
        "## Memory State Assessment",
        "- Search for LoadSynchronous() blocking calls.",
        "- Check for missing UPROPERTY() on UObject pointers (GC-unsafe).",
        "- Look for dangling TWeakObjectPtr without IsValid() checks.",
        "- Search for raw new/delete outside engine allocation patterns.",
        "",
        "## Root Cause Hypothesis",
        "- Must cite file:line evidence from source code.",
        "- State the crash type (null deref, GC, threading, GPU, etc.).",
        "",
        "## Reproduction Strategy",
        "- Steps to trigger the crash reliably.",
        "- Platform-specific reproduction if applicable.",
        "",
        "## Domain Classification",
        "- Analyze the crash module and code path to determine domain:",
        "  - **gameplay**: GAS, abilities, combo graph, character state, hit reaction",
        "  - **graphics**: GPU, shader, material, Niagara, rendering, RHI",
        "  - **engine**: threading, async loading, plugin, subsystem lifecycle",
        "  - **platform**: PS5/Xbox SDK, platform-specific API, certification",
        "  - **memory**: GC, allocation, leak, corruption",
        "- Recommend handoff workflow based on classification.",
        "",
        "## Fix Recommendations",
        "- Concrete code changes with before/after.",
        "- Rank by confidence level.",
        "",
        "## Verification Plan",
        *_format_bullets(verification, empty_message="Add a concrete crash reproduction test once root cause is confirmed.").splitlines(),
        "- Platform-specific testing needed.",
        "- Crash reproduction test after fix.",
    ]
    if review_feedback.strip():
        lines.extend(["", "## Open Questions", "- Reviewer feedback still to satisfy:", *_format_bullets([review_feedback]).splitlines()])
    if improvement_actions:
        lines.extend(["", "## Open Questions" if "Open Questions" not in "\n".join(lines) else "", "- Reviewer checklist:", *_format_bullets(improvement_actions).splitlines()])
    return "\n".join(lines).strip()


def _extract_section(doc: str, heading: str) -> str:
    lines = doc.splitlines()
    capture = False
    result: list[str] = []
    for line in lines:
        if line.strip().lower() == f"## {heading.lower()}":
            capture = True
            continue
        if capture and line.startswith("## "):
            break
        if capture:
            result.append(line)
    return "\n".join(result).strip() or "Not found."


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name
    reviewer_graph = context.get_workflow_graph("optimize-investigation-reviewer-workflow")

    def investigate(state: CrashInvestigationState) -> dict[str, Any]:
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
                        "Use Read, Grep, Glob, and Bash (read-only commands) to gather crash evidence.",
                        f"Write a markdown investigation brief with these sections: {sections_str}.",
                        "Check Saved/Crashes/ and Saved/Logs/ for crash dumps and logs.",
                        "Under Domain Classification, state which domain owns this crash: "
                        "gameplay (GAS, abilities, combat), graphics (GPU, shader, rendering), "
                        "engine (threading, build, plugin, subsystem), platform (PS5/Xbox SDK), "
                        "or memory (GC, allocation, leak). Recommend handoff workflow.",
                        "CRITICAL: Your final output must be a CONCISE markdown brief under 4000 words. "
                        "Do NOT dump raw file contents or tool output. Summarize findings, cite file:line references, "
                        "and focus on actionable analysis. The brief will be reviewed by a separate LLM with limited context.",
                    ],
                )
                task_prompt = build_executor_task_prompt(
                    description=(
                        f"Triage a crash/stability issue. Identify root cause AND classify which domain owns this crash.\n\n"
                        f"Host project root: {context.host_root}\n"
                        f"Investigation round: {investigation_round}/{MAX_REVIEW_ROUNDS}\n\n"
                        f"Task: {state['task_prompt']}\n\n"
                        f"Round goal: {_investigation_round_goal(investigation_round=investigation_round, review_feedback=str(state.get('review_feedback', '')), previous_investigation=str(state.get('investigation_doc', '')))}\n\n"
                        f"Verification mandate: {_investigation_pass_mandate(investigation_round)}\n\n"
                        f"Crash investigation focus areas:\n"
                        f"- Call stack: Search for check(), ensure(), checkf(), verifyf() patterns, null dereference\n"
                        f"- Memory: Search for LoadSynchronous(), missing UPROPERTY(), dangling pointers, raw new/delete\n"
                        f"- Threading: Search for async operations, GameThread assertions, race conditions, FRunnable\n"
                        f"- GPU: Search for RHI errors, shader compilation failures, device removal, TDR timeout\n"
                        f"- GC: Search for UObject pointers without UPROPERTY(), prevent GC marking issues\n"
                        f"- Logs: Check Saved/Crashes/ and Saved/Logs/ for crash reports\n\n"
                        f"Suggested docs: {_format_bullets(project_context['docs'], empty_message='None.')}\n"
                        f"Suggested source: {_format_bullets(project_context['source'], empty_message='None.')}\n"
                        f"Suggested tests: {_format_bullets(project_context['tests'], empty_message='None.')}\n\n"
                        f"Previous investigation:\n{_truncate_prior_context(state.get('investigation_doc', '') or 'None. First round.')}\n"
                    ),
                    prior_feedback=_truncate_prior_context(state.get("review_feedback", "")) or None,
                    context=project_context["snapshot"],
                )
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
                    "source files, and run targeted read-only commands that help prove the crash hypothesis. "
                    "Do not modify files, do not write patches, and do not invent command output. "
                    if investigation_mode == "codex-agent-tools"
                    else "Work only from the provided host-project context and previous review artifacts. Do not invent tool usage or command output. "
                )
                investigation_doc = investigator_llm.generate_text(
                    instructions=(
                        f"You are {metadata.name}. Triage a crash/stability issue like a senior engine programmer. "
                        f"Write a markdown investigation brief using this exact section order: {sections_str}. "
                        f"Stay concrete, evidence-driven, and strict about scope. {investigation_method}"
                        "Focus on call stacks, memory state, threading, GPU crashes, and GC issues. "
                        "Under Domain Classification, state which domain owns this crash and recommend a handoff workflow. "
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
            "optimization_domain": CRASH_DOMAIN,
            "summary": f"{metadata.name} completed investigation round {investigation_round} and handed the brief to senior review.",
        }

    def request_review(state: CrashInvestigationState) -> dict[str, Any]:
        review_round = int(state.get("review_round", 0)) + 1
        return {
            "summary": (
                f"{metadata.name} handed investigation round {state['investigation_round']} to senior review "
                f"for review round {review_round}."
            )
        }

    def capture_review_result(state: CrashInvestigationState) -> dict[str, Any]:
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
            "domain_classification": _extract_section(state.get("investigation_doc", ""), "Domain Classification"),
            "crash_report": state.get("crash_report", ""),
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
                    "# Crash Investigation Final Report",
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

    def generate_report(state: CrashInvestigationState) -> dict[str, Any]:
        artifact_path = _artifact_dir(context, metadata, state)

        tools: list[Any] = []
        for tool_name in metadata.tools:
            try:
                tools.append(context.get_tool(tool_name).tool)
            except KeyError:
                pass

        if not tools:
            report = _build_fallback_report(state)
        else:
            engine = ToolEngine(
                config=ToolEngineConfig(
                    system_id="crash-report-generation",
                    persona=(
                        "You are generating a structured crash analysis report from "
                        "investigation findings. Extract the key sections from the "
                        "investigation document and call the crash-analyze-report tool "
                        "to create a formatted handoff report."
                    ),
                    max_turns=2,
                ),
                tools=tools,
                llm=context.get_llm("investigator"),
            )
            result = engine.gather(
                task="Generate crash analysis report from investigation findings.",
                context=f"Investigation document:\n{_truncate_prior_context(state.get('investigation_doc', ''))}",
            )
            report = result.summary if result.success else _build_fallback_report(state)

        (artifact_path / "crash_report.md").write_text(report, encoding="utf-8")
        return {"crash_report": report}

    def save_crash_findings(state: CrashInvestigationState) -> dict[str, Any]:
        if not state.get("review_approved", False):
            return {}

        memory_path = context.memory_root / MEMORY_NAMESPACE
        memory_path.mkdir(parents=True, exist_ok=True)

        doc = state.get("investigation_doc", "")
        task_prompt = state.get("task_prompt", "")
        slug = _short_slug(task_prompt, fallback="crash")

        (memory_path / f"{slug}.md").write_text(
            f"# {task_prompt[:100]}\n\n"
            f"Score: {state.get('review_score', 0)}/100\n\n"
            f"## Root Cause\n{_extract_section(doc, 'Root Cause Hypothesis')}\n\n"
            f"## Domain\n{_extract_section(doc, 'Domain Classification')}\n\n"
            f"## Fix\n{_extract_section(doc, 'Fix Recommendations')}\n",
            encoding="utf-8",
        )
        return {}

    def review_gate(state: CrashInvestigationState) -> str:
        return "investigate" if state["loop_should_continue"] else "generate_report"

    graph = StateGraph(CrashInvestigationState)
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
    graph.add_node(
        "generate_report",
        trace_graph_node(graph_name=graph_name, node_name="generate_report", node_fn=generate_report),
    )
    graph.add_node(
        "save_crash_findings",
        trace_graph_node(graph_name=graph_name, node_name="save_crash_findings", node_fn=save_crash_findings),
    )
    graph.add_edge(START, "investigate")
    graph.add_edge("investigate", "request_review")
    graph.add_edge("request_review", "optimize-investigation-reviewer-workflow")
    graph.add_edge("optimize-investigation-reviewer-workflow", "capture_review_result")
    graph.add_conditional_edges(
        "capture_review_result",
        trace_route_decision(graph_name=graph_name, router_name="review_gate", route_fn=review_gate),
        {"investigate": "investigate", "generate_report": "generate_report"},
    )
    graph.add_edge("generate_report", "save_crash_findings")
    graph.add_edge("save_crash_findings", END)
    return graph.compile()


def _build_fallback_report(state: CrashInvestigationState) -> str:
    doc = state.get("investigation_doc", "")
    return (
        f"# Crash Analysis Report\n\n"
        f"## Crash Summary\n{state.get('task_prompt', '')[:200]}\n\n"
        f"## Root Cause\n{_extract_section(doc, 'Root Cause Hypothesis')}\n\n"
        f"## Domain Classification\n{_extract_section(doc, 'Domain Classification')}\n\n"
        f"## Reproduction Steps\n{_extract_section(doc, 'Reproduction Strategy')}\n\n"
        f"## Fix Guidance\n{_extract_section(doc, 'Fix Recommendations')}\n\n"
        f"## Verification Criteria\n{_extract_section(doc, 'Verification Plan')}\n"
    )
