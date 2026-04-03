from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.models import WorkflowContext, WorkflowMetadata
from core.tool_engine import ToolEngine, ToolEngineConfig

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Thread → domain mapping
# ------------------------------------------------------------------ #

THREAD_DOMAIN_MAP: dict[str, str] = {
    "gamethread": "gamethread",
    "renderthread": "rendering",
    "rhithread": "rendering",
    "gputhread": "rendering",
    "asyncloadingthread": "streaming",
    "streamingthread": "streaming",
    "pakprecachethread": "streaming",
}

# A thread is considered a bottleneck when its total_ms exceeds this
# fraction of the heaviest thread's total_ms.
BOTTLENECK_RATIO = 0.30

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
    optick_analysis: NotRequired[str]
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


def _classify_from_thread_breakdown(raw_analysis: str) -> list[str]:
    """Parse optick tool output and classify bottleneck domains.

    The optick-analyze tool returns text containing a JSON artifact.
    We look for per_thread_scopes or thread_breakdown data to decide
    which domains are hot.
    """
    # Try to extract structured data from the tool output.
    thread_totals: dict[str, float] = {}

    # The tool output contains markdown blocks with JSON.  Try to find
    # the artifact dict that has per_thread_scopes.
    for line in raw_analysis.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Try to parse any JSON object we find.
        if stripped.startswith("{"):
            try:
                data = json.loads(stripped)
                if "per_thread_scopes" in data:
                    for thread_name, scopes in data["per_thread_scopes"].items():
                        total = sum(s.get("total_ms", 0) for s in scopes)
                        thread_totals[thread_name.lower().replace(" ", "")] = total
                break
            except (json.JSONDecodeError, TypeError):
                pass

    # Also try to find thread_breakdown in the raw text (key: value lines).
    if not thread_totals:
        for line in raw_analysis.splitlines():
            stripped = line.strip().lower()
            for thread_key in THREAD_DOMAIN_MAP:
                if thread_key in stripped:
                    # Try to extract a numeric value from the line.
                    for token in stripped.split():
                        token = token.rstrip("ms,")
                        try:
                            val = float(token)
                            if val > 0:
                                thread_totals[thread_key] = max(
                                    thread_totals.get(thread_key, 0), val
                                )
                        except ValueError:
                            continue

    if not thread_totals:
        return []

    # Map threads to domains and aggregate.
    domain_totals: dict[str, float] = {}
    for thread_key, total_ms in thread_totals.items():
        domain = THREAD_DOMAIN_MAP.get(thread_key)
        if domain:
            domain_totals[domain] = domain_totals.get(domain, 0) + total_ms

    if not domain_totals:
        return []

    max_total = max(domain_totals.values())
    if max_total <= 0:
        return []

    # Any domain exceeding the threshold ratio is a bottleneck.
    bottlenecks = [
        domain
        for domain, total in sorted(domain_totals.items(), key=lambda x: -x[1])
        if total >= max_total * BOTTLENECK_RATIO
    ]
    return bottlenecks


def _llm_classify_prompt(task_prompt: str) -> list[str]:
    """Keyword-based fallback when no profiling data is available."""
    text = task_prompt.lower()
    domains: list[str] = []

    gamethread_keywords = {"gamethread", "game thread", "tick", "gas", "ability", "ai", "physics", "behavior tree"}
    rendering_keywords = {"render", "draw call", "gpu", "shader", "material", "nanite", "lumen", "shadow", "lod"}
    streaming_keywords = {"stream", "loading", "hitch", "world partition", "level streaming", "async", "texture streaming"}

    if any(kw in text for kw in gamethread_keywords):
        domains.append("gamethread")
    if any(kw in text for kw in rendering_keywords):
        domains.append("rendering")
    if any(kw in text for kw in streaming_keywords):
        domains.append("streaming")
    return domains


# ------------------------------------------------------------------ #
#  Graph builder
# ------------------------------------------------------------------ #


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    graph_name = metadata.name

    # -- Node: triage_analyze -------------------------------------------

    def triage_analyze(state: TriageState) -> dict[str, Any]:
        """Run optick-analyze with no thread filter to get full breakdown."""
        tools: list[Any] = []
        for tool_name in metadata.tools:
            try:
                tools.append(context.get_tool(tool_name).tool)
            except KeyError:
                pass

        if not tools:
            logger.warning("[%s] No tools available for triage analysis", graph_name)
            return {"optick_analysis": ""}

        llm = context.get_llm("investigator")
        if not llm or not llm.is_enabled():
            logger.warning("[%s] LLM unavailable for triage analysis", graph_name)
            return {"optick_analysis": ""}

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
                    "Extract any .opt file path from the task prompt below and analyze it "
                    "using the optick-analyze tool. Use broad analysis parameters to see all threads:\n"
                    "  per_thread_top_n: 10\n"
                    "  spike_threshold_ms: 16.67\n"
                    "Do NOT filter by thread_names or scope_keywords — we need the full picture.\n"
                    "If no .opt file is mentioned, set done=true immediately.\n\n"
                    f"Task prompt:\n{state['task_prompt']}"
                ),
            )
            if result.success and result.tool_results_text().strip():
                return {"optick_analysis": result.tool_results_text()}
        except Exception:
            logger.exception("[%s] Error during triage analysis", graph_name)

        return {"optick_analysis": ""}

    # -- Node: triage_classify -----------------------------------------

    def triage_classify(state: TriageState) -> dict[str, Any]:
        """Classify bottleneck domains from profiling data or prompt text."""
        raw = str(state.get("optick_analysis", "")).strip()
        domains: list[str] = []

        if raw:
            domains = _classify_from_thread_breakdown(raw)

        # Fallback: keyword classification from prompt text.
        if not domains:
            domains = _llm_classify_prompt(state["task_prompt"])

        # Last resort: dispatch all domains.
        if not domains:
            logger.info("[%s] Cannot determine domain — dispatching all", graph_name)
            domains = list(CHILD_WORKFLOWS.keys())

        logger.info("[%s] Classified bottleneck domains: %s", graph_name, domains)
        return {"classified_domains": domains}

    # -- Node: dispatch_domains ----------------------------------------

    def dispatch_domains(state: TriageState) -> dict[str, Any]:
        """Invoke child optimization workflow(s) for each classified domain."""
        domains = state.get("classified_domains", [])
        optick_context = str(state.get("optick_analysis", "")).strip()
        results: list[dict[str, Any]] = []

        triage_preamble = ""
        if optick_context:
            triage_preamble = (
                "## Triage Analysis (pre-parsed profiling data)\n\n"
                f"{optick_context}\n\n---\n\n"
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

        for r in results:
            domain = r.get("domain", "unknown")
            sections.append(f"\n## {domain.title()} Domain\n")
            if r.get("error"):
                sections.append(f"**Error:** {r['error']}\n")
            else:
                sections.append(r.get("summary", "No summary available."))
                sections.append("")

        report_text = "\n".join(sections)

        # Write artifact.
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

    # -- Assemble graph ------------------------------------------------

    g = StateGraph(TriageState)

    g.add_node("triage_analyze", triage_analyze)
    g.add_node("triage_classify", triage_classify)
    g.add_node("dispatch_domains", dispatch_domains)
    g.add_node("collect_results", collect_results)

    g.add_edge(START, "triage_analyze")
    g.add_edge("triage_analyze", "triage_classify")
    g.add_edge("triage_classify", "dispatch_domains")
    g.add_edge("dispatch_domains", "collect_results")
    g.add_edge("collect_results", END)

    return g
