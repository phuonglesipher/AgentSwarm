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
OPTIMIZATION_DOMAIN = "streaming"
REVIEW_CRITERIA = (
    ("Problem Scoping", 10, "Task Framing"),
    ("Partition & Streaming Architecture", 25, "World Partition Layout", "Level Streaming Audit"),
    ("Asset Pipeline Analysis", 20, "Asset Loading Pipeline", "Content Organization"),
    ("Memory & Texture Evidence", 20, "Texture Streaming Analysis", "Memory Pressure Points"),
    ("Hitch Root Cause", 15, "Hitch & Stall Sources"),
    ("Verification Completeness", 10, "Verification Plan"),
)
INVESTIGATION_SECTIONS = (
    "Task Framing",
    "World Partition Layout",
    "Level Streaming Audit",
    "Asset Loading Pipeline",
    "Texture Streaming Analysis",
    "Memory Pressure Points",
    "Hitch & Stall Sources",
    "Content Organization",
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


class StreamingInvestigationState(TypedDict):
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
    optick_analysis: NotRequired[str]
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
        return "First pass. Identify the most impactful streaming bottlenecks and build an evidence-backed hypothesis."
    if investigation_round < MIN_REVIEW_ROUNDS:
        return "Verification pass. Add fresh evidence on streaming costs, memory pressure, or hitch sources before approval can stick."
    if review_feedback.strip():
        return "Verification pass. Explicitly answer the previous senior review with fresh streaming evidence, not just a rewrite."
    if previous_investigation.strip():
        return "Verification pass. Independently re-check the current streaming hypothesis and tighten the evidence."
    return "Verification pass. Re-validate the current hypothesis before final handoff."


def _investigation_pass_mandate(investigation_round: int) -> str:
    if investigation_round < MIN_REVIEW_ROUNDS:
        return (
            f"This workflow requires at least {MIN_REVIEW_ROUNDS} review rounds, so this pass must leave a clear path "
            "for an independent verification round instead of treating the first hypothesis as final. "
            "Include world partition analysis, streaming volume audit, and async loading assessment."
        )
    return (
        "This pass must independently re-verify or falsify the previous hypothesis with at least one new piece of evidence: "
        "streaming pool stats, memory pressure data, hitch profiling, or asset reference chain analysis."
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


def _artifact_dir(context: WorkflowContext, metadata: WorkflowMetadata, state: StreamingInvestigationState) -> Path:
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
    the top ``_MAX_SUBDIRS_TO_SCAN`` matches.  Avoids scanning thousands of
    files inside broad roots like ``Plugins/`` on large UE projects.
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
    query_text = f"{task_prompt}\n{review_feedback}\nstreaming world partition level streaming texture async loading hitch memory".strip()
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
    optick_context: str | None = None,
) -> str:
    owners = relevant_source or relevant_docs or relevant_tests or ["No strong owner found yet; inspect streaming configuration first."]
    verification = relevant_tests or relevant_source or ["Add a focused streaming regression check once bottlenecks are confirmed."]
    revision_goal = _investigation_round_goal(
        investigation_round=investigation_round,
        review_feedback=review_feedback,
        previous_investigation=previous_investigation,
    )
    lines = [
        "# Streaming Optimization Investigation",
        "",
        "## Task Framing",
        f"- Round: {investigation_round}",
        f"- Request: {task_prompt}",
        f"- Revision goal: {revision_goal}",
        f"- Verification mandate: {_investigation_pass_mandate(investigation_round)}",
        "",
        "## World Partition Layout",
        project_snapshot,
        "- Analyze grid cell sizes, data layer configuration, and HLOD setup.",
        "- Check streaming distances and whether they match gameplay requirements.",
        "",
        "## Level Streaming Audit",
        "- Search for streaming volume configurations and load/unload triggers.",
        "- Identify always-loaded vs streamed content split and whether it is balanced.",
        "",
        "## Asset Loading Pipeline",
        "- Search for async loading patterns (StreamableManager, LoadPackageAsync).",
        "- Check priority queue configuration and bundle structure.",
        "- Identify synchronous loads that may cause hitches.",
        "",
        "## Texture Streaming Analysis",
        "- Check texture streaming pool size in DefaultEngine.ini.",
        "- Identify texture group budgets and over-budget textures.",
        "- Look for mip bias configuration and forced mip levels.",
        "",
        "## Memory Pressure Points",
        "- Analyze peak resident memory during streaming transitions.",
        "- Check streaming pool utilization and GC pressure from streaming.",
        "- Identify memory spikes during level transitions.",
        "",
        "## Hitch & Stall Sources",
        *([f"\n### Optick Capture Data\n{optick_context}"] if optick_context else []),
        "- Search for FlushAsyncLoading, LoadObject, and other sync load patterns on game thread.",
        "- Identify blocking loads during gameplay (not just level transitions).",
        "- Look for flush points that force synchronous completion.",
        "",
        "## Content Organization",
        *_format_bullets(owners).splitlines(),
        "- Check asset redundancy across world partition cells.",
        "- Analyze soft/hard reference chains that may pull in excessive dependencies.",
        "",
        "## Optimization Recommendations",
        "- Rank changes by expected improvement (reduced hitches, lower memory, faster loads).",
        "- Provide concrete before/after expectations for each recommendation.",
        "",
        "## Verification Plan",
        *_format_bullets(verification, empty_message="Add a concrete streaming test once bottlenecks are confirmed.").splitlines(),
        "- Use stat streaming, memreport, and Insights async loading markers to measure improvement.",
        "- Test streaming transitions under worst-case player movement patterns.",
    ]
    if review_feedback.strip():
        lines.extend(["", "## Open Questions", "- Reviewer feedback still to satisfy:", *_format_bullets([review_feedback]).splitlines()])
    if improvement_actions:
        lines.extend(["", "## Open Questions" if "Open Questions" not in "\n".join(lines) else "", "- Reviewer checklist:", *_format_bullets(improvement_actions).splitlines()])
    return "\n".join(lines).strip()


OPTICK_THREAD_FILTER = "AsyncLoadingThread,StreamingThread"
OPTICK_SCOPE_FILTER = "AsyncLoad,Stream,FlushAsync,LoadObject,WorldPartition,LevelStreaming,StreamableManager,CreateActorsForLevel"

OPTICK_DOMAIN_FOCUS = (
    "Analyze the capture focusing on streaming hitches and loading stalls. "
    "Pay special attention to frame_times_ms for sudden spikes indicating hitches. "
    "Highlight scopes related to FlushAsyncLoading, LoadObject, streaming manager, and world partition loading. "
    "Report frames above 16.67ms and 33.33ms thresholds as hitch frequency indicators."
)


def _gather_optick_context(
    context: WorkflowContext,
    metadata: WorkflowMetadata,
    state: StreamingInvestigationState,
) -> str | None:
    # Already attempted in a prior round — return cached result (even if empty).
    if "optick_analysis" in state:
        cached = str(state["optick_analysis"]).strip()
        return cached or None

    tools: list[Any] = []
    for tool_name in metadata.tools:
        try:
            tools.append(context.get_tool(tool_name).tool)
        except KeyError:
            pass
    if not tools:
        return None

    llm = context.get_llm("investigator")
    if not llm or not llm.is_enabled():
        return None

    try:
        engine = ToolEngine(
            config=ToolEngineConfig(
                system_id=f"{metadata.name}-optick-gather",
                persona=f"You are a performance data analyst. {OPTICK_DOMAIN_FOCUS}",
                max_turns=2,
                require_tool_use=False,
            ),
            tools=tools,
            llm=llm,
        )
        result = engine.gather(
            task=(
                "Extract any .opt file path from the task prompt below and analyze it using the optick-analyze tool. "
                "If no .opt file is mentioned, set done=true immediately.\n\n"
                f"When calling optick-analyze, use these filters to focus on streaming data:\n"
                f"  thread_names: \"{OPTICK_THREAD_FILTER}\"\n"
                f"  scope_keywords: \"{OPTICK_SCOPE_FILTER}\"\n"
                f"  spike_threshold_ms: 16.67\n\n"
                f"Task prompt:\n{state['task_prompt']}"
            ),
        )
        if result.success and result.tool_results_text().strip():
            return result.tool_results_text()
    except Exception:
        pass
    return None


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name
    reviewer_graph = context.get_workflow_graph("optimize-investigation-reviewer-workflow")

    def investigate(state: StreamingInvestigationState) -> dict[str, Any]:
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
        optick_context = _gather_optick_context(context, metadata, state)
        optick_section = f"Optick capture analysis:\n{optick_context}" if optick_context else "No Optick capture data available."
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
            optick_context=optick_context,
        )

        sections_str = ", ".join(INVESTIGATION_SECTIONS)
        investigation_doc = fallback_doc
        if investigation_mode == "claude-executor-tools" and isinstance(investigator_llm, ClaudeCodeExecutorClient):
            try:
                system_prompt = build_executor_system_prompt(
                    working_directory=str(context.host_root),
                    scope_constraints=[
                        "Investigate only — do not modify files, do not write patches.",
                        "Use Read, Grep, Glob, and Bash (read-only commands) to gather streaming evidence.",
                        f"Write a markdown investigation brief with these sections: {sections_str}.",
                        "Focus on world partition config, streaming volumes, async loading, texture streaming, and hitch sources.",
                        "Provide concrete numbers (cell sizes, pool sizes, load times, memory usage) wherever possible.",
                        "CRITICAL: Your final output must be a CONCISE markdown brief under 4000 words. "
                        "Do NOT dump raw file contents or tool output. Summarize findings, cite file:line references, "
                        "and focus on actionable analysis. The brief will be reviewed by a separate LLM with limited context.",
                    ],
                )
                task_prompt = build_executor_task_prompt(
                    description=(
                        f"Investigate streaming and asset loading performance like a senior performance engineer.\n\n"
                        f"Host project root: {context.host_root}\n"
                        f"Investigation round: {investigation_round}/{MAX_REVIEW_ROUNDS}\n\n"
                        f"Task: {state['task_prompt']}\n\n"
                        f"Round goal: {_investigation_round_goal(investigation_round=investigation_round, review_feedback=str(state.get('review_feedback', '')), previous_investigation=str(state.get('investigation_doc', '')))}\n\n"
                        f"Verification mandate: {_investigation_pass_mandate(investigation_round)}\n\n"
                        f"Streaming focus areas:\n"
                        f"- World Partition: Search for WorldPartition config, grid cell sizes, data layers, HLOD settings\n"
                        f"- Level Streaming: Search for LevelStreamingDynamic, streaming volumes, load triggers\n"
                        f"- Async Loading: Search for LoadPackageAsync, StreamableManager, FlushAsyncLoading\n"
                        f"- Texture Streaming: Check DefaultEngine.ini for r.Streaming.PoolSize, texture group budgets\n"
                        f"- Hitches: Search for LoadObject, FlushAsyncLoading on game thread\n\n"
                        f"Suggested docs: {_format_bullets(project_context['docs'], empty_message='None.')}\n"
                        f"Suggested source: {_format_bullets(project_context['source'], empty_message='None.')}\n"
                        f"Suggested tests: {_format_bullets(project_context['tests'], empty_message='None.')}\n\n"
                        f"{optick_section}\n\n"
                        f"Previous investigation:\n{_truncate_prior_context(state.get('investigation_doc', '') or 'None. First round.')}\n"
                    ),
                    prior_feedback=_truncate_prior_context(state.get("review_feedback", "")) or None,
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
                investigation_method = (
                    "Use the Codex agent tools available in this environment to inspect the project directly, read the most relevant "
                    "source files, and run targeted read-only commands that help prove the streaming hypothesis. "
                    "Do not modify files, do not write patches, and do not invent command output. "
                    if investigation_mode == "codex-agent-tools"
                    else "Work only from the provided host-project context and previous review artifacts. Do not invent tool usage or command output. "
                )
                investigation_doc = investigator_llm.generate_text(
                    instructions=(
                        f"You are {metadata.name}. Investigate the host project's streaming and asset loading performance like a senior "
                        f"performance engineer. Write a markdown investigation brief using this exact section order: {sections_str}. "
                        f"Stay concrete, evidence-driven, and strict about scope. {investigation_method}"
                        "Focus on world partition layout, streaming volumes, async loading patterns, texture streaming, and hitch sources. "
                        "Provide concrete numbers (cell sizes, pool sizes, memory, load times) wherever possible. "
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
                        f"{optick_section}\n\n"
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
            "optick_analysis": optick_context or "",
            "review_criteria": [list(c) for c in REVIEW_CRITERIA],
            "optimization_domain": OPTIMIZATION_DOMAIN,
            "summary": f"{metadata.name} completed investigation round {investigation_round} and handed the brief to senior review.",
        }

    def request_review(state: StreamingInvestigationState) -> dict[str, Any]:
        review_round = int(state.get("review_round", 0)) + 1
        return {
            "summary": (
                f"{metadata.name} handed investigation round {state['investigation_round']} to senior review "
                f"for review round {review_round}."
            )
        }

    def capture_review_result(state: StreamingInvestigationState) -> dict[str, Any]:
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
                    "# Streaming Optimization Investigation Final Report",
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

    def review_gate(state: StreamingInvestigationState) -> str:
        return "investigate" if state["loop_should_continue"] else END

    graph = StateGraph(StreamingInvestigationState)
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
