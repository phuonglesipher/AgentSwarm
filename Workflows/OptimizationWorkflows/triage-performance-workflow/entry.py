from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.decision.engine import DecisionEngine
from core.decision.profile import Branch, DecisionProfile
from core.models import WorkflowContext, WorkflowMetadata
from core.tool_engine import ToolEngine, ToolEngineConfig

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Child workflow mapping
# ------------------------------------------------------------------ #

CHILD_WORKFLOWS = {
    "gamethread": "optimize-gamethread-workflow",
    "rendering": "optimize-rendering-workflow",
    "streaming": "optimize-streaming-workflow",
}


# ------------------------------------------------------------------ #
#  State
# ------------------------------------------------------------------ #


class TriageState(TypedDict):
    prompt: NotRequired[str]
    task_prompt: str
    task_id: NotRequired[str]
    run_dir: NotRequired[str]
    profiling_analysis: NotRequired[str]
    profiling_data: NotRequired[dict[str, Any]]
    classified_domains: NotRequired[list[str]]
    domain_results: NotRequired[list[dict[str, Any]]]
    final_report: NotRequired[dict[str, Any]]
    summary: NotRequired[str]


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _artifact_dir(context: WorkflowContext, metadata: WorkflowMetadata, state: TriageState) -> Path:
    run_dir = state.get("run_dir", "")
    if run_dir:
        d = Path(run_dir) / metadata.name
    else:
        d = context.artifact_root / metadata.name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _summarize_profiling_state(state: dict[str, Any]) -> str:
    """Extract key profiling metrics for DecisionEngine LLM prompt."""
    data = state.get("profiling_data", {})
    if not data:
        return f"No profiling data available.\nTask prompt: {state.get('task_prompt', '')[:500]}"

    lines: list[str] = []

    # Bound analysis (from optick or utrace)
    ba = data.get("bound_analysis", {})
    if ba:
        lines.append(f"Primary bound: {ba.get('primary_bound', 'unknown')}")
        lines.append(f"Bound by: {ba.get('bound_by', 'unknown')}")
        lines.append(f"Explanation: {ba.get('explanation', '')}")
        budget = ba.get("frame_budget", {})
        if budget:
            lines.append(f"Frame budget: {json.dumps(budget)}")

    # Frame summary
    fs = data.get("frame_summary", {})
    if fs:
        lines.append(f"Frames: {fs.get('total_frames', 0)}, "
                      f"avg={fs.get('avg_ms', 0)}ms, "
                      f"p99={fs.get('p99_ms', 0)}ms, "
                      f"miss_60fps={fs.get('frames_above_16ms', 0)}, "
                      f"miss_30fps={fs.get('frames_above_33ms', 0)}")

    # GPU pipeline (utrace only)
    gpu_queues = data.get("gpu_queues", [])
    if gpu_queues:
        for q in gpu_queues:
            frames = q.get("frames", 1) or 1
            work = q.get("work_ms", 0) / frames
            wait = q.get("wait_ms", 0) / frames
            if work > 0 or wait > 0:
                lines.append(f"GPU {q.get('name', '?')}: work={work:.1f}ms/f wait={wait:.1f}ms/f")

    # Thread breakdown (top 5)
    tb = data.get("thread_breakdown", [])
    if tb:
        lines.append("Thread breakdown (top 5):")
        for t in tb[:5]:
            lines.append(f"  {t.get('name', '?')}: {t.get('total_ms', 0):.0f}ms")

    # Bookmarks — streaming events
    bookmarks = data.get("bookmarks", [])
    streaming_bms = [b for b in bookmarks if "RequestLevel" in b.get("text", "")]
    if streaming_bms:
        lines.append(f"Level streaming events: {len(streaming_bms)}")
    pso_bms = [b for b in bookmarks if "PSO" in b.get("text", "")]
    if pso_bms:
        lines.append(f"PSO compilation events: {len(pso_bms)}")

    # Bound distribution (optick)
    bd = ba.get("bound_distribution", [])
    if bd:
        lines.append("Bound distribution: " + ", ".join(
            f"{d['thread']} {d['pct']}%" for d in bd[:4]
        ))

    # Always include task prompt so LLM can judge intent (analyze-only vs optimize)
    lines.append(f"\nUser task prompt: {state.get('task_prompt', '')[:500]}")

    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Graph builder
# ------------------------------------------------------------------ #


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name

    # -- Decision profile for domain routing ---------------------------

    triage_decision_profile = DecisionProfile(
        system_id="perf-triage-domain-router",
        branches=(
            Branch(
                name="gamethread",
                description=(
                    "Game thread is the primary CPU bottleneck. GameThread has the highest "
                    "per-frame cost. Focus on tick functions, GAS abilities, AI, physics queries, "
                    "Blueprint VM overhead. Bound analysis says CPU-bound on GameThread."
                ),
                target="dispatch_gamethread",
            ),
            Branch(
                name="rendering",
                description=(
                    "Rendering pipeline is the bottleneck. GPU-bound (Graphics queue work exceeds "
                    "CPU frame time), RenderThread-bound, or RHIThread-bound. Focus on draw calls, "
                    "materials, shadows, Nanite, Lumen, TSR, post-processing. GPU pipeline shows "
                    "high work_ms or cross-queue sync issues."
                ),
                is_default=True,
                target="dispatch_rendering",
            ),
            Branch(
                name="streaming",
                description=(
                    "Level streaming or asset loading causes hitches. Bookmarks show RequestLevel "
                    "events, PSO compilation stalls, or AsyncLoadingThread is hot. Frame spikes "
                    "correlate with streaming events."
                ),
                target="dispatch_streaming",
            ),
            Branch(
                name="multi_domain",
                description=(
                    "Multiple domains are problematic — e.g., CPU and GPU both over budget, "
                    "or streaming hitches on top of a rendering bottleneck. Dispatch to all "
                    "relevant workflows for comprehensive analysis."
                ),
                target="dispatch_all",
            ),
            Branch(
                name="analysis_only",
                description=(
                    "The user only wants profiling analysis — they asked to 'analyze', "
                    "'check performance', 'profile', or 'look at the trace' without requesting "
                    "optimization suggestions or fixes. Return the profiling results directly "
                    "without dispatching to any optimization workflow."
                ),
                target="report_analysis",
            ),
        ),
        context_instruction=(
            "You are a performance triage expert for Unreal Engine 5. "
            "Given profiling data (bound analysis, thread breakdown, GPU pipeline, "
            "bookmarks) AND the user's task prompt, decide the next step.\n"
            "If the user only wants analysis/profiling results (e.g., 'analyze this trace', "
            "'check FPS', 'profile this capture') → choose analysis_only.\n"
            "If the user wants optimization/fixes (e.g., 'optimize performance', "
            "'fix FPS drops', 'improve frame time') → choose the appropriate domain.\n"
            "Choose multi_domain only when multiple areas are clearly over budget."
        ),
        state_summarizer=_summarize_profiling_state,
        llm_profile="investigator",
    )

    # -- Node: triage_analyze ------------------------------------------

    def triage_analyze(state: TriageState) -> dict[str, Any]:
        """Run profiling tool (optick-analyze or utrace-analyze) for full breakdown."""
        tools: list[Any] = []
        for tool_name in metadata.tools:
            try:
                tools.append(context.get_tool(tool_name).tool)
            except KeyError:
                pass

        if not tools:
            logger.warning("[%s] No profiling tools available", graph_name)
            return {"profiling_analysis": "", "profiling_data": {}}

        llm = context.get_llm("investigator")
        if not llm or not llm.is_enabled():
            logger.warning("[%s] LLM unavailable for triage analysis", graph_name)
            return {"profiling_analysis": "", "profiling_data": {}}

        try:
            engine = ToolEngine(
                config=ToolEngineConfig(
                    system_id=f"{metadata.name}-triage-analyze",
                    persona=(
                        "You are a performance triage analyst. Your job is to analyze "
                        "profiling captures and identify which subsystems are bottlenecked."
                    ),
                    max_turns=2,
                    require_tool_use=False,
                ),
                tools=tools,
                llm=llm,
            )
            result = engine.gather(
                task=(
                    "Extract any profiling capture file path from the task prompt below.\n"
                    "- If it's a .opt file, use optick-analyze with per_thread_top_n=10 and spike_threshold_ms=16.67\n"
                    "- If it's a .utrace file, use utrace-analyze with analysis_mode=all and spike_threshold_ms=16.67\n"
                    "Do NOT filter by thread_names or scope_keywords — we need the full picture.\n"
                    "If no profiling file is mentioned, set done=true immediately.\n\n"
                    f"Task prompt:\n{state['task_prompt']}"
                ),
            )
            if result.success and result.tool_results_text().strip():
                artifact = (
                    result.first_artifact("utrace-analyze")
                    or result.first_artifact("optick-analyze")
                    or {}
                )
                return {
                    "profiling_analysis": result.tool_results_text(),
                    "profiling_data": artifact,
                }
        except Exception:
            logger.exception("[%s] Error during triage analysis", graph_name)

        return {"profiling_analysis": "", "profiling_data": {}}

    # -- Node: triage_classify -----------------------------------------

    def triage_classify(state: TriageState) -> dict[str, Any]:
        """Classify bottleneck domains using DecisionEngine."""
        decision_engine = DecisionEngine(
            triage_decision_profile, context, metadata,
        )
        choice = decision_engine.decide(state)

        if choice == "report_analysis":
            # Analysis-only: skip optimization workflows
            domains = ["__analysis_only__"]
        elif choice == "dispatch_all":
            domains = list(CHILD_WORKFLOWS.keys())
        elif choice in ("dispatch_gamethread", "dispatch_rendering", "dispatch_streaming"):
            domain = choice.replace("dispatch_", "")
            domains = [domain]
        else:
            # Fallback: keyword classification from prompt text
            domains = _keyword_classify(state["task_prompt"])
            if not domains:
                domains = list(CHILD_WORKFLOWS.keys())

        logger.info("[%s] DecisionEngine classified domains: %s (choice=%s)", graph_name, domains, choice)
        return {"classified_domains": domains}

    # -- Node: dispatch_domains ----------------------------------------

    def dispatch_domains(state: TriageState) -> dict[str, Any]:
        """Invoke child optimization workflow(s) for each classified domain."""
        domains = state.get("classified_domains", [])
        profiling_context = str(state.get("profiling_analysis", "")).strip()
        results: list[dict[str, Any]] = []

        triage_preamble = ""
        if profiling_context:
            triage_preamble = (
                "## Triage Analysis (pre-parsed profiling data)\n\n"
                f"{profiling_context}\n\n---\n\n"
            )

        for domain in domains:
            workflow_name = CHILD_WORKFLOWS.get(domain)
            if not workflow_name:
                logger.warning("[%s] No workflow for domain: %s", graph_name, domain)
                results.append({"domain": domain, "error": f"No workflow for domain {domain}"})
                continue

            enriched_prompt = f"{triage_preamble}{state['task_prompt']}"
            payload: dict[str, Any] = {
                "task_prompt": enriched_prompt,
                "task_id": state.get("task_id", ""),
                "run_dir": state.get("run_dir", ""),
            }

            logger.info("[%s] Dispatching to %s for domain '%s'", graph_name, workflow_name, domain)
            try:
                child_result = context.invoke_workflow(workflow_name, payload)
                results.append({
                    "domain": domain,
                    "workflow": workflow_name,
                    "summary": child_result.get("summary", ""),
                    "final_report": child_result.get("final_report", {}),
                })
            except Exception as exc:
                logger.exception("[%s] Error invoking %s", graph_name, workflow_name)
                results.append({"domain": domain, "workflow": workflow_name, "error": str(exc)})

        return {"domain_results": results}

    # -- Node: collect_results -----------------------------------------

    def collect_results(state: TriageState) -> dict[str, Any]:
        """Merge child workflow results into a unified report."""
        results = state.get("domain_results", [])
        artifact_path = _artifact_dir(context, metadata, state)

        sections: list[str] = ["# Performance Triage Report\n"]
        sections.append(f"**Classified domains:** {', '.join(state.get('classified_domains', []))}\n")

        # Include bound analysis summary if available
        data = state.get("profiling_data", {})
        ba = data.get("bound_analysis", {})
        if ba:
            sections.append(f"**Bound:** {ba.get('primary_bound', '?')} — {ba.get('bound_by', '?')}")
            sections.append(f"{ba.get('explanation', '')}\n")

        for r in results:
            domain = r.get("domain", "unknown")
            sections.append(f"\n## {domain.title()} Domain\n")
            if r.get("error"):
                sections.append(f"**Error:** {r['error']}\n")
            else:
                sections.append(r.get("summary", "No summary available."))
                sections.append("")

        report_text = "\n".join(sections)

        report_file = artifact_path / "triage_report.md"
        report_file.write_text(report_text, encoding="utf-8")

        summary_parts = []
        for r in results:
            domain = r.get("domain", "?")
            if r.get("error"):
                summary_parts.append(f"{domain}: error — {r['error']}")
            else:
                summary_parts.append(f"{domain}: {r.get('summary', 'completed')}")

        summary = (
            f"Performance triage completed. "
            f"Domains investigated: {', '.join(state.get('classified_domains', []))}. "
            + " | ".join(summary_parts)
        )

        return {
            "final_report": {
                "classified_domains": state.get("classified_domains", []),
                "domain_results": results,
                "report_path": str(report_file),
            },
            "summary": summary,
        }

    # -- Node: report_analysis -----------------------------------------

    def report_analysis(state: TriageState) -> dict[str, Any]:
        """Return profiling analysis directly without dispatching optimization workflows."""
        artifact_path = _artifact_dir(context, metadata, state)
        profiling_text = str(state.get("profiling_analysis", "")).strip()
        data = state.get("profiling_data", {})

        sections: list[str] = ["# Performance Analysis Report\n"]

        ba = data.get("bound_analysis", {})
        if ba:
            sections.append(f"**Bound:** {ba.get('primary_bound', '?')} — {ba.get('bound_by', '?')}")
            sections.append(f"{ba.get('explanation', '')}\n")

        if profiling_text:
            sections.append("## Profiling Data\n")
            sections.append(profiling_text)

        report_text = "\n".join(sections)
        report_file = artifact_path / "analysis_report.md"
        report_file.write_text(report_text, encoding="utf-8")

        bound_summary = ""
        if ba:
            bound_summary = f" {ba.get('primary_bound', '?')}-bound ({ba.get('bound_by', '?')}). "

        summary = (
            f"Performance analysis completed.{bound_summary}"
            f"See {report_file.name} for full profiling data."
        )

        return {
            "final_report": {
                "classified_domains": ["analysis_only"],
                "report_path": str(report_file),
            },
            "summary": summary,
        }

    # -- Routing function for classify → dispatch or report ------------

    def _route_after_classify(state: TriageState) -> str:
        domains = state.get("classified_domains", [])
        if domains == ["__analysis_only__"]:
            return "report_analysis"
        return "dispatch_domains"

    # -- Assemble graph ------------------------------------------------

    g = StateGraph(TriageState)

    g.add_node("triage_analyze", triage_analyze)
    g.add_node("triage_classify", triage_classify)
    g.add_node("dispatch_domains", dispatch_domains)
    g.add_node("collect_results", collect_results)
    g.add_node("report_analysis", report_analysis)

    g.add_edge(START, "triage_analyze")
    g.add_edge("triage_analyze", "triage_classify")
    g.add_conditional_edges("triage_classify", _route_after_classify, {
        "dispatch_domains": "dispatch_domains",
        "report_analysis": "report_analysis",
    })
    g.add_edge("dispatch_domains", "collect_results")
    g.add_edge("collect_results", END)
    g.add_edge("report_analysis", END)

    return g


# ------------------------------------------------------------------ #
#  Keyword fallback (when LLM unavailable)
# ------------------------------------------------------------------ #


def _keyword_classify(task_prompt: str) -> list[str]:
    """Keyword-based fallback when DecisionEngine LLM is unavailable."""
    text = task_prompt.lower()
    domains: list[str] = []

    gt_keywords = {"gamethread", "game thread", "tick", "gas", "ability", "ai", "physics", "behavior tree"}
    rt_keywords = {"render", "draw call", "gpu", "shader", "material", "nanite", "lumen", "shadow", "lod"}
    st_keywords = {"stream", "loading", "hitch", "world partition", "level streaming", "async", "texture streaming"}

    if any(kw in text for kw in gt_keywords):
        domains.append("gamethread")
    if any(kw in text for kw in rt_keywords):
        domains.append("rendering")
    if any(kw in text for kw in st_keywords):
        domains.append("streaming")
    return domains
