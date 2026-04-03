from __future__ import annotations

import sys
from pathlib import Path

# Allow importing optick_parser from the same directory
_tool_dir = str(Path(__file__).parent)
if _tool_dir not in sys.path:
    sys.path.insert(0, _tool_dir)

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata
from optick_parser import analyze_capture, parse_opt_file


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def optick_analyze(
        file_path: str,
        top_n: int = 20,
    ) -> tuple[str, dict]:
        """Parse an Optick .opt capture file and return performance analysis.

        Reads binary .opt files produced by the Optick profiler plugin,
        extracts frame timings, per-thread breakdowns, and identifies
        the hottest profiling scopes by total time.

        Args:
            file_path: Absolute or host-project-relative path to the .opt file.
            top_n: Number of hottest scopes to include (default 20).
        """
        # Resolve path — try absolute first, then relative to host project
        p = Path(file_path)
        if not p.is_absolute():
            host_root = context.resolve_scope_root("host_project")
            p = host_root / file_path

        if not p.exists():
            return (
                f"File not found: {file_path}",
                {"error": "file_not_found", "path": str(p)},
            )

        if not p.suffix.lower() == ".opt":
            return (
                f"Not an Optick capture file (expected .opt): {p.name}",
                {"error": "wrong_extension", "path": str(p)},
            )

        try:
            capture = parse_opt_file(p)
        except ValueError as e:
            return (
                f"Failed to parse Optick capture: {e}",
                {"error": "parse_error", "detail": str(e)},
            )
        except Exception as e:
            return (
                f"Unexpected error parsing {p.name}: {e}",
                {"error": "unexpected_error", "detail": str(e)},
            )

        analysis = analyze_capture(capture, top_n=top_n)

        # Build human-readable summary
        lines = []
        if "frame_summary" in analysis:
            fs = analysis["frame_summary"]
            lines.append(f"Frames: {fs['total_frames']}, "
                         f"Avg: {fs['avg_ms']}ms, "
                         f"P99: {fs['p99_ms']}ms, "
                         f"Max: {fs['max_ms']}ms")
            lines.append(f"Frames >16.67ms (miss 60fps): {fs['frames_above_16ms']}")
            lines.append(f"Frames >33.33ms (miss 30fps): {fs['frames_above_33ms']}")

        if "metadata" in analysis:
            meta = analysis["metadata"]
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items())
            lines.append(f"Platform: {meta_str}")

        if "hottest_scopes" in analysis:
            lines.append(f"Top {len(analysis['hottest_scopes'])} hottest scopes by total time:")
            for i, scope in enumerate(analysis["hottest_scopes"][:5], 1):
                lines.append(f"  {i}. {scope['name']} — {scope['total_ms']}ms total, "
                             f"{scope['avg_ms']}ms avg, {scope['calls']} calls")
            if len(analysis["hottest_scopes"]) > 5:
                lines.append(f"  ... and {len(analysis['hottest_scopes']) - 5} more in artifact")

        summary = "\n".join(lines) if lines else "Capture parsed but no frame data found."

        return (summary, analysis)

    return optick_analyze
