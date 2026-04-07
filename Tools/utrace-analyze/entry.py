from __future__ import annotations

import sys
from pathlib import Path

# Allow importing sibling modules from the same directory
_tool_dir = str(Path(__file__).parent)
if _tool_dir not in sys.path:
    sys.path.insert(0, _tool_dir)

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata
from utrace_analyzer import analyze_utrace
from utrace_parser import parse_utrace_file


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def utrace_analyze(
        file_path: str,
        analysis_mode: str = "summary",
        top_n: int = 20,
        thread_names: str = "",
        spike_threshold_ms: float = 0.0,
        max_memory_events: int = 500_000,
    ) -> tuple[str, dict]:
        """Parse an Unreal Insights .utrace file and return performance analysis.

        Reads binary .utrace files produced by Unreal Engine's trace system
        and extracts CPU profiler scopes, GPU pass timing, memory allocations,
        counters, frame timing, and session diagnostics.

        Args:
            file_path: Absolute or host-project-relative path to the .utrace file.
                If a directory is given, picks the most recent .utrace file in it.
            analysis_mode: What to analyze — "summary" (brief overview),
                "cpu", "gpu", "memory", "counters", or "all" (full detail).
            top_n: Number of top items per category (default 20).
            thread_names: Comma-separated thread names to filter CPU analysis
                (e.g. "GameThread,RenderThread"). Empty means all threads.
            spike_threshold_ms: When >0, report frame spikes exceeding this
                duration (e.g. 16.67 for 60fps target).
            max_memory_events: Cap on stored memory events to prevent OOM
                on large traces (default 500K).
        """
        # Resolve path
        p = Path(file_path)
        if not p.is_absolute():
            host_root = context.resolve_scope_root("host_project")
            p = host_root / file_path

        # Directory support — pick most recent .utrace
        if p.is_dir():
            utrace_files = sorted(p.glob("*.utrace"), key=lambda f: f.stat().st_mtime, reverse=True)
            if not utrace_files:
                return (
                    f"No .utrace files found in: {file_path}",
                    {"error": "no_utrace_files", "path": str(p)},
                )
            p = utrace_files[0]

        if not p.exists():
            return (
                f"File not found: {file_path}",
                {"error": "file_not_found", "path": str(p)},
            )

        if p.suffix.lower() != ".utrace":
            return (
                f"Not a utrace file (expected .utrace): {p.name}",
                {"error": "wrong_extension", "path": str(p)},
            )

        # Validate analysis mode
        valid_modes = {"summary", "cpu", "gpu", "memory", "counters", "all"}
        if analysis_mode not in valid_modes:
            return (
                f"Invalid analysis_mode '{analysis_mode}'. Valid: {', '.join(sorted(valid_modes))}",
                {"error": "invalid_mode"},
            )

        # Parse
        try:
            capture = parse_utrace_file(p, max_memory_events=max_memory_events)
        except ValueError as e:
            return (
                f"Failed to parse utrace file: {e}",
                {"error": "parse_error", "detail": str(e)},
            )
        except Exception as e:
            return (
                f"Unexpected error parsing {p.name}: {e}",
                {"error": "unexpected_error", "detail": str(e)},
            )

        # Parse filter strings
        thread_list = (
            [t.strip() for t in thread_names.split(",") if t.strip()]
            if thread_names else None
        )

        # Analyze
        analysis = analyze_utrace(
            capture,
            mode=analysis_mode,
            top_n=top_n,
            thread_names=thread_list,
            spike_threshold_ms=spike_threshold_ms,
        )

        # Build human-readable summary
        lines = [f"Parsed: {p.name} ({p.stat().st_size / (1024*1024):.1f} MB)"]

        # Stream info
        si = analysis.get("stream_info", {})
        lines.append(
            f"Protocol v{si.get('protocol_version', '?')}, "
            f"Transport v{si.get('transport_version', '?')}, "
            f"{si.get('total_packets', 0)} packets, "
            f"{si.get('total_events', 0)} events, "
            f"{si.get('event_types_count', 0)} event types"
        )

        if si.get("parse_errors", 0) > 0:
            lines.append(f"Parse errors: {si['parse_errors']}")

        if si.get("unknown_events", 0) > 0:
            lines.append(f"Unknown/unhandled events: {si['unknown_events']}")

        # Filters
        filter_parts = []
        if thread_list:
            filter_parts.append(f"threads={','.join(thread_list)}")
        if spike_threshold_ms > 0:
            filter_parts.append(f"spike_threshold={spike_threshold_ms}ms")
        if filter_parts:
            lines.append(f"Filters: {', '.join(filter_parts)}")

        # Session info
        session = analysis.get("session_info", {})
        if session:
            interesting = {k: v for k, v in session.items()
                          if k.lower() in ("platform", "appname", "commandline", "buildversion", "configurationname")}
            if interesting:
                lines.append("Session: " + ", ".join(f"{k}={v}" for k, v in interesting.items()))

        # Frame summary
        fs = analysis.get("frame_summary")
        if fs:
            lines.append(
                f"Frames: {fs['total_frames']}, "
                f"Avg: {fs['avg_ticks']} ticks, "
                f"P99: {fs['p99_ticks']} ticks, "
                f"Max: {fs['max_ticks']} ticks"
            )

        # CPU hottest scopes
        cpu_scopes = analysis.get("cpu_hottest_scopes", [])
        if cpu_scopes:
            lines.append(f"CPU — {analysis.get('cpu_total_scopes', 0)} unique scopes, "
                         f"{analysis.get('cpu_scope_event_count', 0)} scope events:")
            for i, scope in enumerate(cpu_scopes[:5], 1):
                lines.append(
                    f"  {i}. {scope['name']} — "
                    f"{scope['total_ticks']} ticks total, "
                    f"{scope['count']} calls"
                )
            if len(cpu_scopes) > 5:
                lines.append(f"  ... and {len(cpu_scopes) - 5} more in artifact")

        # GPU
        gpu_passes = analysis.get("gpu_pass_timing", [])
        if gpu_passes:
            lines.append(f"GPU — {analysis.get('gpu_event_count', 0)} events:")
            for i, p_stat in enumerate(gpu_passes[:5], 1):
                lines.append(
                    f"  {i}. {p_stat['name']} — "
                    f"{p_stat['total_ticks']} ticks, "
                    f"{p_stat['count']} calls"
                )
            if len(gpu_passes) > 5:
                lines.append(f"  ... and {len(gpu_passes) - 5} more in artifact")

        # Memory
        if "memory_peak_bytes" in analysis:
            lines.append(
                f"Memory — peak: {analysis['memory_peak_mb']} MB, "
                f"current: {analysis['memory_current_mb']} MB, "
                f"allocs: {analysis['memory_alloc_count']}, "
                f"frees: {analysis['memory_free_count']}, "
                f"live: {analysis['memory_live_allocations']}"
            )
            if analysis.get("memory_events_truncated"):
                lines.append(f"  (memory events capped at {max_memory_events})")

        # Counters
        counters = analysis.get("counters", [])
        if counters:
            lines.append(f"Counters — {analysis.get('counter_spec_count', 0)} defined, "
                         f"{analysis.get('counter_sample_count', 0)} samples:")
            for c in counters[:5]:
                lines.append(f"  {c['name']}: last={c['last']}, avg={c['avg']}, "
                             f"min={c['min']}, max={c['max']}")
            if len(counters) > 5:
                lines.append(f"  ... and {len(counters) - 5} more in artifact")

        # Spikes
        spike_count = analysis.get("frame_spike_count")
        if spike_count is not None:
            lines.append(
                f"Frame spikes (>{spike_threshold_ms}ms): {spike_count} "
                f"({analysis.get('frame_spike_pct', 0)}% of frames)"
            )

        # Event type catalog
        event_types = analysis.get("event_types", [])
        if event_types:
            lines.append(f"Event types discovered ({len(event_types)}):")
            for et in event_types[:10]:
                lines.append(f"  UID {et['uid']:4d}: {et['name']} ({et['fields']} fields)")
            if len(event_types) > 10:
                lines.append(f"  ... and {len(event_types) - 10} more in artifact")

        summary = "\n".join(lines) if lines else "Capture parsed but no data found."
        return (summary, analysis)

    return utrace_analyze
