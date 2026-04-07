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


def _analyze_single_file(
    p: Path,
    analysis_mode: str,
    top_n: int,
    thread_list: list[str] | None,
    keyword_list: list[str] | None,
    per_thread_top_n: int,
    spike_threshold_ms: float,
    max_memory_events: int,
) -> tuple[dict | None, str | None]:
    """Parse and analyze one .utrace file.

    Returns (analysis_dict, error_string).  Exactly one is None.
    """
    try:
        capture = parse_utrace_file(p, max_memory_events=max_memory_events)
    except ValueError as e:
        return None, f"Failed to parse {p.name}: {e}"
    except Exception as e:
        return None, f"Unexpected error parsing {p.name}: {e}"

    analysis = analyze_utrace(
        capture,
        mode=analysis_mode,
        top_n=top_n,
        thread_names=thread_list,
        scope_keywords=keyword_list,
        per_thread_top_n=per_thread_top_n,
        spike_threshold_ms=spike_threshold_ms,
    )
    del capture
    analysis["filename"] = p.name
    return analysis, None


def _format_single_analysis(
    analysis: dict,
    thread_list: list[str] | None,
    keyword_list: list[str] | None,
    per_thread_top_n: int,
    spike_threshold_ms: float,
) -> str:
    """Build a human-readable summary from one analysis dict."""
    lines: list[str] = []

    # File info
    filename = analysis.get("filename", "")
    si = analysis.get("stream_info", {})
    freq = analysis.get("cpu_cycle_frequency", 0)
    if filename:
        lines.append(f"Parsed: {filename}")
    if freq > 0:
        lines.append(f"CPU frequency: {freq / 1e6:.1f} MHz")

    lines.append(
        f"Protocol v{si.get('protocol_version', '?')}, "
        f"Transport v{si.get('transport_version', '?')}, "
        f"{si.get('total_events', 0):,} events, "
        f"{si.get('event_types_count', 0)} event types"
    )
    if si.get("parse_errors", 0) > 0:
        lines.append(f"Parse errors: {si['parse_errors']}")

    # Session info
    session = analysis.get("session_info", {})
    if session:
        interesting = {k: v for k, v in session.items()
                       if k.lower() in ("platform", "appname", "commandline",
                                         "buildversion", "configurationname",
                                         "projectname")}
        if interesting:
            lines.append("Session: " + ", ".join(f"{k}={v}" for k, v in interesting.items()))

    # Active filters
    filter_parts = []
    if thread_list:
        filter_parts.append(f"threads={','.join(thread_list)}")
    if keyword_list:
        filter_parts.append(f"keywords={','.join(keyword_list)}")
    if per_thread_top_n > 0:
        filter_parts.append(f"per_thread_top_n={per_thread_top_n}")
    if spike_threshold_ms > 0:
        filter_parts.append(f"spike_threshold={spike_threshold_ms}ms")
    if filter_parts:
        lines.append(f"Filters: {', '.join(filter_parts)}")

    # Frame summary
    if "frame_summary" in analysis:
        fs = analysis["frame_summary"]
        lines.append(
            f"Frames: {fs['total_frames']}, "
            f"Avg: {fs['avg_ms']}ms, "
            f"Median: {fs['median_ms']}ms, "
            f"P95: {fs['p95_ms']}ms, "
            f"P99: {fs['p99_ms']}ms, "
            f"Max: {fs['max_ms']}ms"
        )
        lines.append(f"Frames >16.67ms (miss 60fps): {fs['frames_above_16ms']}")
        lines.append(f"Frames >33.33ms (miss 30fps): {fs['frames_above_33ms']}")

    # Hottest scopes
    if "per_thread_scopes" in analysis:
        for tname, scopes in analysis["per_thread_scopes"].items():
            lines.append(f"{tname} — top {len(scopes)} scopes:")
            for i, scope in enumerate(scopes[:5], 1):
                lines.append(
                    f"  {i}. {scope['name']} — {scope['total_ms']}ms total, "
                    f"{scope['avg_ms']}ms avg, {scope['max_ms']}ms max, "
                    f"{scope['calls']} calls"
                )
            if len(scopes) > 5:
                lines.append(f"  ... and {len(scopes) - 5} more in artifact")
    elif "hottest_scopes" in analysis:
        lines.append(f"Top {len(analysis['hottest_scopes'])} hottest scopes by total time:")
        for i, scope in enumerate(analysis["hottest_scopes"][:5], 1):
            lines.append(
                f"  {i}. {scope['name']} — {scope['total_ms']}ms total, "
                f"{scope['avg_ms']}ms avg, {scope['max_ms']}ms max, "
                f"{scope['calls']} calls"
            )
        if len(analysis["hottest_scopes"]) > 5:
            lines.append(f"  ... and {len(analysis['hottest_scopes']) - 5} more in artifact")

    # Thread breakdown
    if "thread_breakdown" in analysis:
        tb = analysis["thread_breakdown"]
        lines.append(f"Thread breakdown ({len(tb)} threads):")
        for t in tb[:5]:
            lines.append(f"  {t['name']}: {t['total_ms']}ms")
        if len(tb) > 5:
            lines.append(f"  ... and {len(tb) - 5} more in artifact")

    # Spikes
    if "spike_count" in analysis:
        lines.append(
            f"Frame spikes (>{spike_threshold_ms}ms): {analysis['spike_count']} "
            f"({analysis['spike_pct']}% of frames)"
        )

    # GPU
    gpu_passes = analysis.get("gpu_pass_timing", [])
    if gpu_passes:
        lines.append(f"GPU — {analysis.get('gpu_event_count', 0)} events:")
        for i, p_stat in enumerate(gpu_passes[:5], 1):
            lines.append(
                f"  {i}. {p_stat['name']} — {p_stat['total_ms']}ms total, "
                f"{p_stat['count']} calls"
            )

    # Memory
    if "memory_peak_mb" in analysis:
        lines.append(
            f"Memory — peak: {analysis['memory_peak_mb']} MB, "
            f"current: {analysis['memory_current_mb']} MB, "
            f"allocs: {analysis['memory_alloc_count']}, "
            f"frees: {analysis['memory_free_count']}"
        )

    # Bookmarks
    bookmark_counts = analysis.get("bookmark_counts_by_name", [])
    if bookmark_counts:
        lines.append(f"Bookmarks — {analysis.get('bookmark_count', 0)} total:")
        for i, bc in enumerate(bookmark_counts[:5], 1):
            lines.append(f"  {i}. {bc['name']} — {bc['count']} occurrences")
        if len(bookmark_counts) > 5:
            lines.append(f"  ... and {len(bookmark_counts) - 5} more in artifact")
    elif analysis.get("bookmark_count", 0) > 0:
        lines.append(f"Bookmarks: {analysis['bookmark_count']} (specs not resolved)")

    return "\n".join(lines) if lines else "Capture parsed but no data found."


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def utrace_analyze(
        file_path: str,
        analysis_mode: str = "summary",
        top_n: int = 20,
        thread_names: str = "",
        scope_keywords: str = "",
        per_thread_top_n: int = 0,
        spike_threshold_ms: float = 0.0,
        max_memory_events: int = 500_000,
        max_files: int = 10,
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
            scope_keywords: Comma-separated keywords to filter scope names
                (e.g. "Tick,Ability,Physics"). Only matching scopes returned.
            per_thread_top_n: When >0, group scopes by thread instead of
                a global list. Returns top N scopes per thread.
            spike_threshold_ms: When >0, report frame spikes exceeding this
                duration (e.g. 16.67 for 60fps target).
            max_memory_events: Cap on stored memory events (default 500K).
            max_files: Max files when given a directory (default 10).
        """
        # Parse filter strings
        thread_list = (
            [t.strip() for t in thread_names.split(",") if t.strip()]
            if thread_names else None
        )
        keyword_list = (
            [k.strip() for k in scope_keywords.split(",") if k.strip()]
            if scope_keywords else None
        )

        # Resolve path
        p = Path(file_path)
        if not p.is_absolute():
            host_root = context.resolve_scope_root("host_project")
            p = host_root / file_path

        if not p.exists():
            return (
                f"Path not found: {file_path}",
                {"error": "path_not_found", "path": str(p)},
            )

        # Directory mode: batch analyze
        if p.is_dir():
            utrace_files = sorted(
                p.glob("*.utrace"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if not utrace_files:
                return (
                    f"No .utrace files found in: {file_path}",
                    {"error": "no_utrace_files", "path": str(p)},
                )

            total_found = len(utrace_files)
            files_to_process = utrace_files[:max_files]
            file_results: list[dict] = []
            errors: list[str] = []
            summary_lines: list[str] = [
                f"Folder: {p.name} — {total_found} .utrace file(s), "
                f"processing {len(files_to_process)}",
            ]

            for uf in files_to_process:
                a, err = _analyze_single_file(
                    uf, analysis_mode, top_n, thread_list, keyword_list,
                    per_thread_top_n, spike_threshold_ms, max_memory_events,
                )
                if err:
                    errors.append(err)
                    summary_lines.append(f"  x {uf.name}: {err}")
                    continue

                file_results.append(a)
                fs = a.get("frame_summary")
                if fs:
                    summary_lines.append(
                        f"  {uf.name}: {fs['total_frames']} frames, "
                        f"avg {fs['avg_ms']}ms, p99 {fs['p99_ms']}ms, "
                        f"max {fs['max_ms']}ms"
                    )
                else:
                    summary_lines.append(f"  {uf.name}: parsed (no frame data)")

            if total_found > max_files:
                summary_lines.append(
                    f"  ... {total_found - max_files} more file(s) skipped"
                )

            return (
                "\n".join(summary_lines),
                {
                    "mode": "batch",
                    "directory": str(p),
                    "file_count": total_found,
                    "processed": len(file_results),
                    "errors": len(errors),
                    "files": file_results,
                },
            )

        # Single file mode
        if p.suffix.lower() != ".utrace":
            return (
                f"Not a utrace file (expected .utrace): {p.name}",
                {"error": "wrong_extension", "path": str(p)},
            )

        # Validate analysis mode
        valid_modes = {"summary", "cpu", "gpu", "memory", "counters", "bookmarks", "all"}
        if analysis_mode not in valid_modes:
            return (
                f"Invalid analysis_mode '{analysis_mode}'. "
                f"Valid: {', '.join(sorted(valid_modes))}",
                {"error": "invalid_mode"},
            )

        analysis, error = _analyze_single_file(
            p, analysis_mode, top_n, thread_list, keyword_list,
            per_thread_top_n, spike_threshold_ms, max_memory_events,
        )
        if error:
            return (error, {"error": "parse_error", "detail": error})

        summary = _format_single_analysis(
            analysis, thread_list, keyword_list,
            per_thread_top_n, spike_threshold_ms,
        )
        return (summary, analysis)

    return utrace_analyze
