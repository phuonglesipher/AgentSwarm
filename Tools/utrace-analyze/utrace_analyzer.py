"""Analysis functions for parsed UTrace captures.

Produces structured dicts with performance metrics from each trace channel:
CPU scope timing, GPU pass analysis, memory allocation tracking,
counter values, and frame timing statistics.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from utrace_parser import UTraceCapture


def analyze_utrace(
    capture: UTraceCapture,
    *,
    mode: str = "summary",
    top_n: int = 20,
    thread_names: list[str] | None = None,
    spike_threshold_ms: float = 0.0,
) -> dict:
    """Analyze a parsed UTrace capture.

    Args:
        capture: Parsed UTraceCapture from parse_utrace_file.
        mode: Analysis mode — "summary", "cpu", "gpu", "memory", "counters", "all".
        top_n: Number of top items to return per category.
        thread_names: Filter CPU analysis to these thread names.
        spike_threshold_ms: Report frame spikes above this threshold.

    Returns:
        Dict with analysis results keyed by channel.
    """
    result: dict = {}

    # Always include metadata
    result["stream_info"] = {
        "magic": capture.magic,
        "transport_version": capture.transport_version,
        "protocol_version": capture.protocol_version,
        "total_packets": capture.total_packets,
        "total_events": capture.total_events,
        "unknown_events": capture.unknown_event_count,
        "event_types_count": len(capture.event_types),
        "parse_errors": len(capture.parse_errors),
    }

    if capture.session_info:
        result["session_info"] = capture.session_info

    if capture.parse_errors:
        result["parse_errors"] = capture.parse_errors[:20]

    # Event type catalog
    result["event_types"] = [
        {"uid": et.uid, "name": et.full_name, "fields": len(et.fields)}
        for et in sorted(capture.event_types.values(), key=lambda e: e.uid)
    ]

    is_summary = mode == "summary"
    n = 5 if is_summary else top_n

    if mode in ("cpu", "all", "summary"):
        cpu = _analyze_cpu(capture, n, thread_names)
        if cpu:
            result.update(cpu)

    if mode in ("gpu", "all", "summary"):
        gpu = _analyze_gpu(capture, n)
        if gpu:
            result.update(gpu)

    if mode in ("memory", "all", "summary"):
        mem = _analyze_memory(capture, n)
        if mem:
            result.update(mem)

    if mode in ("counters", "all", "summary"):
        counters = _analyze_counters(capture, n)
        if counters:
            result.update(counters)

    if mode in ("cpu", "gpu", "all", "summary"):
        frames = _analyze_frames(capture, spike_threshold_ms)
        if frames:
            result.update(frames)

    # Thread info
    if capture.thread_info:
        result["threads"] = [
            {"id": tid, "name": name}
            for tid, name in sorted(capture.thread_info.items())
        ]

    # Bookmarks
    if capture.bookmarks and mode in ("all", "summary"):
        bm_list = []
        for bm in capture.bookmarks[:50]:
            spec_id = bm.get("spec_id", 0)
            name = capture.bookmark_specs.get(spec_id, f"Bookmark_{spec_id}")
            bm_list.append({"name": name, "timestamp": bm.get("timestamp", 0)})
        if bm_list:
            result["bookmarks"] = bm_list

    return result


# ---------------------------------------------------------------------------
# CPU analysis
# ---------------------------------------------------------------------------

def _analyze_cpu(
    capture: UTraceCapture,
    top_n: int,
    thread_names: list[str] | None,
) -> dict | None:
    if not capture.cpu_scopes:
        return None

    # Group scopes by thread
    scopes_by_thread: dict[int, list] = defaultdict(list)
    for thread_id, spec_id, timestamp, is_enter in capture.cpu_scopes:
        scopes_by_thread[thread_id].append((spec_id, timestamp, is_enter))

    # Build allowed thread IDs from names filter
    allowed_threads: set[int] | None = None
    if thread_names:
        thread_name_lower = {n.lower().strip() for n in thread_names}
        allowed_threads = set()
        for tid, name in capture.thread_info.items():
            if name.lower().strip() in thread_name_lower:
                allowed_threads.add(tid)

    # Calculate scope durations per thread using stack matching
    scope_stats: dict[int, dict] = defaultdict(lambda: {
        "total_ticks": 0, "count": 0, "min_ticks": float("inf"), "max_ticks": 0,
    })
    per_thread_total: dict[int, int] = defaultdict(int)

    for tid, scopes in scopes_by_thread.items():
        if allowed_threads is not None and tid not in allowed_threads:
            continue

        stack: list[tuple[int, int]] = []  # (spec_id, enter_timestamp)
        for spec_id, timestamp, is_enter in scopes:
            if is_enter:
                stack.append((spec_id, timestamp))
            elif stack:
                enter_spec_id, enter_ts = stack.pop()
                duration = timestamp - enter_ts
                if duration > 0:
                    stats = scope_stats[enter_spec_id]
                    stats["total_ticks"] += duration
                    stats["count"] += 1
                    if duration < stats["min_ticks"]:
                        stats["min_ticks"] = duration
                    if duration > stats["max_ticks"]:
                        stats["max_ticks"] = duration
                    per_thread_total[tid] += duration

    if not scope_stats:
        return None

    # Sort by total time, take top_n
    sorted_specs = sorted(scope_stats.items(), key=lambda x: x[1]["total_ticks"], reverse=True)

    hottest = []
    for spec_id, stats in sorted_specs[:top_n]:
        name = capture.cpu_scope_specs.get(spec_id, f"Scope_{spec_id}")
        count = stats["count"]
        hottest.append({
            "name": name,
            "spec_id": spec_id,
            "total_ticks": stats["total_ticks"],
            "count": count,
            "avg_ticks": stats["total_ticks"] // count if count else 0,
            "min_ticks": stats["min_ticks"] if stats["min_ticks"] != float("inf") else 0,
            "max_ticks": stats["max_ticks"],
        })

    result: dict = {"cpu_hottest_scopes": hottest, "cpu_total_scopes": len(scope_stats)}

    # Per-thread breakdown
    thread_breakdown = []
    for tid, total in sorted(per_thread_total.items(), key=lambda x: x[1], reverse=True):
        name = capture.thread_info.get(tid, f"Thread_{tid}")
        thread_breakdown.append({"name": name, "thread_id": tid, "total_ticks": total})
    if thread_breakdown:
        result["cpu_per_thread_breakdown"] = thread_breakdown

    result["cpu_scope_spec_count"] = len(capture.cpu_scope_specs)
    result["cpu_scope_event_count"] = len(capture.cpu_scopes)

    return result


# ---------------------------------------------------------------------------
# GPU analysis
# ---------------------------------------------------------------------------

def _analyze_gpu(capture: UTraceCapture, top_n: int) -> dict | None:
    if not capture.gpu_events:
        return None

    # Pair begin/end events by queue
    queue_stacks: dict[int, list] = defaultdict(list)
    pass_durations: dict[int, list[int]] = defaultdict(list)

    for evt in capture.gpu_events:
        kind = evt.get("kind", "")
        queue_id = evt.get("queue_id", 0)
        ts = evt.get("timestamp", 0)
        spec_id = evt.get("spec_id")

        if "Begin" in kind:
            queue_stacks[queue_id].append((spec_id, ts))
        elif "End" in kind and queue_stacks[queue_id]:
            begin_spec, begin_ts = queue_stacks[queue_id].pop()
            duration = ts - begin_ts
            if duration > 0 and begin_spec is not None:
                pass_durations[begin_spec].append(duration)

    if not pass_durations:
        return {"gpu_event_count": len(capture.gpu_events), "gpu_queues": list(capture.gpu_queue_names.values())}

    # Aggregate pass timing
    pass_stats = []
    for spec_id, durations in sorted(pass_durations.items(), key=lambda x: sum(x[1]), reverse=True)[:top_n]:
        name = capture.gpu_breadcrumb_specs.get(spec_id, f"GpuPass_{spec_id}")
        total = sum(durations)
        count = len(durations)
        pass_stats.append({
            "name": name,
            "spec_id": spec_id,
            "total_ticks": total,
            "count": count,
            "avg_ticks": total // count if count else 0,
        })

    return {
        "gpu_pass_timing": pass_stats,
        "gpu_event_count": len(capture.gpu_events),
        "gpu_queues": list(capture.gpu_queue_names.values()),
    }


# ---------------------------------------------------------------------------
# Memory analysis
# ---------------------------------------------------------------------------

def _analyze_memory(capture: UTraceCapture, top_n: int) -> dict | None:
    if not capture.memory_events:
        return None

    # Walk events, track live allocations
    live: dict[int, int] = {}  # address -> size
    current_bytes = 0
    peak_bytes = 0
    alloc_count = 0
    free_count = 0
    heap_usage: dict[int, int] = defaultdict(int)
    tag_usage: dict[int, int] = defaultdict(int)

    for evt in capture.memory_events:
        kind = evt.get("kind", "")
        addr = evt.get("address", 0)
        size = evt.get("size", 0)

        if kind in ("alloc", "realloc_alloc"):
            if isinstance(size, (int, float)) and size > 0:
                live[addr] = int(size)
                current_bytes += int(size)
                alloc_count += 1
                if current_bytes > peak_bytes:
                    peak_bytes = current_bytes

        elif kind in ("free", "realloc_free"):
            freed = live.pop(addr, 0)
            current_bytes -= freed
            free_count += 1

    result: dict = {
        "memory_peak_bytes": peak_bytes,
        "memory_peak_mb": round(peak_bytes / (1024 * 1024), 2),
        "memory_current_bytes": current_bytes,
        "memory_current_mb": round(current_bytes / (1024 * 1024), 2),
        "memory_alloc_count": alloc_count,
        "memory_free_count": free_count,
        "memory_live_allocations": len(live),
        "memory_events_total": len(capture.memory_events),
        "memory_events_truncated": capture.memory_events_truncated,
    }

    # Heap breakdown
    if capture.heap_specs:
        result["memory_heap_specs"] = {
            str(k): v for k, v in capture.heap_specs.items()
        }

    # Tag breakdown
    if capture.tag_specs:
        result["memory_tag_specs"] = {
            str(k): v for k, v in capture.tag_specs.items()
        }

    return result


# ---------------------------------------------------------------------------
# Counter analysis
# ---------------------------------------------------------------------------

def _analyze_counters(capture: UTraceCapture, top_n: int) -> dict | None:
    if not capture.counter_values:
        return None

    # Group by counter ID
    by_counter: dict[int, list] = defaultdict(list)
    for cv in capture.counter_values:
        cid = cv.get("counter_id", 0)
        val = cv.get("value", 0)
        by_counter[cid].append(val)

    counter_stats = []
    for cid, values in sorted(by_counter.items(), key=lambda x: len(x[1]), reverse=True)[:top_n]:
        name = capture.counter_specs.get(cid, f"Counter_{cid}")
        numeric = [v for v in values if isinstance(v, (int, float))]
        if not numeric:
            continue
        counter_stats.append({
            "name": name,
            "counter_id": cid,
            "sample_count": len(numeric),
            "min": min(numeric),
            "max": max(numeric),
            "avg": round(sum(numeric) / len(numeric), 4),
            "last": numeric[-1],
        })

    return {
        "counters": counter_stats,
        "counter_spec_count": len(capture.counter_specs),
        "counter_sample_count": len(capture.counter_values),
    }


# ---------------------------------------------------------------------------
# Frame analysis
# ---------------------------------------------------------------------------

def _analyze_frames(
    capture: UTraceCapture,
    spike_threshold_ms: float,
) -> dict | None:
    if not capture.frame_boundaries:
        return None

    # Pair begin/end by frame type
    open_frames: dict[int, int] = {}  # frame_type -> begin_cycle
    frame_durations: list[int] = []  # in ticks

    for frame_type, cycle, is_begin in capture.frame_boundaries:
        if is_begin:
            open_frames[frame_type] = cycle
        elif frame_type in open_frames:
            begin = open_frames.pop(frame_type)
            duration = cycle - begin
            if duration > 0:
                frame_durations.append(duration)

    if not frame_durations:
        return None

    # Stats in ticks (caller must convert to ms using cycle frequency if available)
    sorted_durations = sorted(frame_durations)
    total = len(sorted_durations)
    p95_idx = int(total * 0.95)
    p99_idx = int(total * 0.99)

    frame_summary = {
        "total_frames": total,
        "avg_ticks": sum(sorted_durations) // total,
        "min_ticks": sorted_durations[0],
        "max_ticks": sorted_durations[-1],
        "median_ticks": sorted_durations[total // 2],
        "p95_ticks": sorted_durations[min(p95_idx, total - 1)],
        "p99_ticks": sorted_durations[min(p99_idx, total - 1)],
    }

    result: dict = {"frame_summary": frame_summary}

    # Spike detection (in ticks — threshold conversion requires frequency)
    if spike_threshold_ms > 0 and capture.cpu_cycle_frequency > 0:
        threshold_ticks = int(spike_threshold_ms * capture.cpu_cycle_frequency / 1000)
        spikes = []
        for i, d in enumerate(frame_durations):
            if d > threshold_ticks:
                spikes.append({
                    "frame_index": i,
                    "duration_ticks": d,
                    "duration_ms": round(d * 1000 / capture.cpu_cycle_frequency, 3),
                })
        result["frame_spikes"] = spikes[:50]
        result["frame_spike_count"] = len([d for d in frame_durations if d > threshold_ticks])
        result["frame_spike_pct"] = round(
            result["frame_spike_count"] / total * 100, 2
        ) if total > 0 else 0

    return result
