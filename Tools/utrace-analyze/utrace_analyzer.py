"""Analysis functions for parsed UTrace captures.

Produces structured dicts with performance metrics from each trace channel:
CPU scope timing (ms), frame summary, spike detection, per-thread breakdown,
and memory/counter/GPU data.

Output structure mirrors optick-analyze for consistency.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from utrace_parser import UTraceCapture


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def analyze_utrace(
    capture: UTraceCapture,
    *,
    mode: str = "summary",
    top_n: int = 20,
    thread_names: list[str] | None = None,
    scope_keywords: list[str] | None = None,
    per_thread_top_n: int = 0,
    spike_threshold_ms: float = 0.0,
) -> dict:
    """Analyze a parsed UTrace capture.

    Args:
        capture: Parsed UTraceCapture from parse_utrace_file.
        mode: Analysis mode — "summary", "cpu", "gpu", "memory", "counters", "all".
        top_n: Number of top items to return per category.
        thread_names: Filter CPU analysis to these thread names.
        scope_keywords: Only include scopes whose name contains any keyword.
        per_thread_top_n: When >0, group scopes by thread instead of global.
        spike_threshold_ms: Report frame spikes above this threshold (e.g. 16.67).

    Returns:
        Dict with analysis results keyed by channel.
    """
    result: dict = {}
    freq = capture.cpu_cycle_frequency

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

    result["cpu_cycle_frequency"] = freq

    is_summary = mode == "summary"
    n = 5 if is_summary else top_n

    if mode in ("cpu", "all", "summary"):
        cpu = _analyze_cpu(capture, n, thread_names, scope_keywords, per_thread_top_n)
        if cpu:
            result.update(cpu)

    if mode in ("gpu", "all", "summary"):
        gpu = _analyze_gpu(capture, n)
        if gpu:
            result.update(gpu)

    if mode in ("memory", "all"):
        mem = _analyze_memory(capture, n)
        if mem:
            result.update(mem)

    if mode in ("counters", "all"):
        counters = _analyze_counters(capture, n)
        if counters:
            result.update(counters)

    if mode in ("bookmarks", "all"):
        bookmarks = _analyze_bookmarks(capture, n)
        if bookmarks:
            result.update(bookmarks)

    if mode == "summary" and capture.bookmarks:
        result["bookmark_count"] = len(capture.bookmarks)

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

    return result


# ---------------------------------------------------------------------------
# CPU analysis
# ---------------------------------------------------------------------------

def _ticks_to_ms(ticks: int, freq: int) -> float:
    if freq <= 0:
        return 0.0
    return (ticks / freq) * 1000.0


def _analyze_cpu(
    capture: UTraceCapture,
    top_n: int,
    thread_names: list[str] | None,
    scope_keywords: list[str] | None,
    per_thread_top_n: int,
) -> dict | None:
    if not capture.cpu_scopes:
        return None

    freq = capture.cpu_cycle_frequency
    if freq <= 0:
        return None

    kw_lower = [kw.lower() for kw in scope_keywords if kw.strip()] if scope_keywords else []

    # Group scopes by thread
    scopes_by_thread: dict[int, list] = defaultdict(list)
    for thread_id, spec_id, timestamp, is_enter in capture.cpu_scopes:
        scopes_by_thread[thread_id].append((spec_id, timestamp, is_enter))

    # Build allowed thread IDs from names filter
    allowed_threads: set[int] | None = None
    if thread_names:
        thread_name_lower = {n.lower().strip('\x00').strip() for n in thread_names}
        allowed_threads = set()
        for tid, name in capture.thread_info.items():
            clean = name.lower().strip('\x00').strip()
            if clean in thread_name_lower:
                allowed_threads.add(tid)

    # Maximum valid duration: 60 seconds (filters out startup/pre-trace artifacts)
    max_valid_ticks = 60 * freq

    # Calculate scope durations per thread using stack matching
    # Key: (thread_key, spec_id) where thread_key is tid or None (global)
    accum: dict[tuple[int | None, int], list] = {}
    per_thread_total: dict[int, int] = defaultdict(int)

    for tid, scopes in scopes_by_thread.items():
        if allowed_threads is not None and tid not in allowed_threads:
            continue

        thread_key = tid if per_thread_top_n > 0 else None
        stack: list[tuple[int, int]] = []  # (spec_id, enter_timestamp)

        for spec_id, timestamp, is_enter in scopes:
            if is_enter:
                stack.append((spec_id, timestamp))
            elif stack:
                enter_spec_id, enter_ts = stack.pop()
                duration = timestamp - enter_ts
                if duration <= 0 or duration > max_valid_ticks:
                    continue

                per_thread_total[tid] += duration

                key = (thread_key, enter_spec_id)
                entry = accum.get(key)
                if entry is None:
                    accum[key] = [duration, 1, duration]  # [total, count, max]
                else:
                    entry[0] += duration
                    entry[1] += 1
                    if duration > entry[2]:
                        entry[2] = duration

    if not accum:
        return None

    def _build_scope_entry(spec_id: int, total_ticks: int, count: int, max_ticks: int) -> dict | None:
        name = capture.cpu_scope_specs.get(spec_id, f"Scope_{spec_id}")
        if kw_lower and not any(kw in name.lower() for kw in kw_lower):
            return None
        total_ms = _ticks_to_ms(total_ticks, freq)
        avg_ms = total_ms / count
        max_ms = _ticks_to_ms(max_ticks, freq)
        return {
            "name": name,
            "spec_id": spec_id,
            "total_ms": round(total_ms, 3),
            "avg_ms": round(avg_ms, 3),
            "max_ms": round(max_ms, 3),
            "calls": count,
        }

    result: dict = {}

    if per_thread_top_n > 0:
        # Group by thread
        thread_groups: dict[int, list[dict]] = {}
        for (tidx, spec_id), (total_ticks, count, max_ticks) in accum.items():
            entry = _build_scope_entry(spec_id, total_ticks, count, max_ticks)
            if entry is None:
                continue
            assert tidx is not None
            thread_groups.setdefault(tidx, []).append(entry)

        per_thread_scopes: dict[str, list[dict]] = {}
        for tidx, scopes in thread_groups.items():
            name = capture.thread_info.get(tidx, f"Thread_{tidx}").strip('\x00')
            scopes.sort(key=lambda x: x["total_ms"], reverse=True)
            per_thread_scopes[name] = scopes[:per_thread_top_n]
        result["per_thread_scopes"] = per_thread_scopes
    else:
        # Global hottest scopes
        hottest = []
        for (_tidx, spec_id), (total_ticks, count, max_ticks) in accum.items():
            entry = _build_scope_entry(spec_id, total_ticks, count, max_ticks)
            if entry is not None:
                hottest.append(entry)
        hottest.sort(key=lambda x: x["total_ms"], reverse=True)
        result["hottest_scopes"] = hottest[:top_n]

    # Per-thread time breakdown
    thread_breakdown = []
    for tid, total in sorted(per_thread_total.items(), key=lambda x: x[1], reverse=True):
        name = capture.thread_info.get(tid, f"Thread_{tid}").strip('\x00')
        thread_breakdown.append({
            "name": name,
            "total_ms": round(_ticks_to_ms(total, freq), 2),
        })
    if thread_breakdown:
        result["thread_breakdown"] = thread_breakdown

    result["cpu_scope_spec_count"] = len(capture.cpu_scope_specs)
    result["cpu_scope_event_count"] = len(capture.cpu_scopes)

    return result


# ---------------------------------------------------------------------------
# GPU analysis
# ---------------------------------------------------------------------------

def _analyze_gpu(capture: UTraceCapture, top_n: int) -> dict | None:
    if not capture.gpu_events:
        return None

    freq = capture.cpu_cycle_frequency

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
        queue_ids = set()
        for e in capture.gpu_events:
            qid = e.get("queue_id", 0)
            if isinstance(qid, int):
                queue_ids.add(qid)
        return {
            "gpu_event_count": len(capture.gpu_events),
            "gpu_queues": [
                capture.gpu_queue_names.get(qid, f"Queue_{qid}")
                for qid in sorted(queue_ids)
            ],
        }

    pass_stats = []
    for spec_id, durations in sorted(
        pass_durations.items(), key=lambda x: sum(x[1]), reverse=True
    )[:top_n]:
        name = capture.gpu_breadcrumb_specs.get(spec_id, f"GpuPass_{spec_id}")
        total = sum(durations)
        count = len(durations)
        pass_stats.append({
            "name": name,
            "spec_id": spec_id,
            "total_ms": round(_ticks_to_ms(total, freq), 3) if freq > 0 else 0,
            "count": count,
            "avg_ms": round(_ticks_to_ms(total // count, freq), 3) if freq > 0 and count else 0,
        })

    queue_ids = set()
    for e in capture.gpu_events:
        qid = e.get("queue_id", 0)
        if isinstance(qid, int):
            queue_ids.add(qid)

    return {
        "gpu_pass_timing": pass_stats,
        "gpu_event_count": len(capture.gpu_events),
        "gpu_queues": [
            capture.gpu_queue_names.get(qid, f"Queue_{qid}")
            for qid in sorted(queue_ids)
        ],
    }


# ---------------------------------------------------------------------------
# Memory analysis
# ---------------------------------------------------------------------------

def _analyze_memory(capture: UTraceCapture, top_n: int) -> dict | None:
    if not capture.memory_events:
        return None

    live: dict[int, int] = {}
    current_bytes = 0
    peak_bytes = 0
    alloc_count = 0
    free_count = 0

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

    return {
        "memory_peak_mb": round(peak_bytes / (1024 * 1024), 2),
        "memory_current_mb": round(current_bytes / (1024 * 1024), 2),
        "memory_alloc_count": alloc_count,
        "memory_free_count": free_count,
        "memory_live_allocations": len(live),
        "memory_events_truncated": capture.memory_events_truncated,
    }


# ---------------------------------------------------------------------------
# Counter analysis
# ---------------------------------------------------------------------------

def _analyze_counters(capture: UTraceCapture, top_n: int) -> dict | None:
    if not capture.counter_values:
        return None

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

    freq = capture.cpu_cycle_frequency
    if freq <= 0:
        return None

    # Pair begin/end by frame type, compute durations in ms
    open_frames: dict[int, int] = {}
    frame_times_ms: list[float] = []
    max_valid_ticks = 10 * freq  # 10 seconds max frame (filters corrupt data)

    for frame_type, cycle, is_begin in capture.frame_boundaries:
        if is_begin:
            open_frames[frame_type] = cycle
        elif frame_type in open_frames:
            begin = open_frames.pop(frame_type)
            duration = cycle - begin
            if 0 < duration < max_valid_ticks:
                frame_times_ms.append(_ticks_to_ms(duration, freq))

    if not frame_times_ms:
        return None

    sorted_times = sorted(frame_times_ms)
    total = len(sorted_times)
    avg = sum(frame_times_ms) / total
    p95_idx = min(int(total * 0.95), total - 1)
    p99_idx = min(int(total * 0.99), total - 1)

    frame_summary = {
        "total_frames": total,
        "avg_ms": round(avg, 2),
        "min_ms": round(sorted_times[0], 2),
        "max_ms": round(sorted_times[-1], 2),
        "median_ms": round(sorted_times[total // 2], 2),
        "p95_ms": round(sorted_times[p95_idx], 2),
        "p99_ms": round(sorted_times[p99_idx], 2),
        "frames_above_16ms": sum(1 for t in frame_times_ms if t > 16.667),
        "frames_above_33ms": sum(1 for t in frame_times_ms if t > 33.333),
    }

    result: dict = {"frame_summary": frame_summary}

    # Spike detection
    if spike_threshold_ms > 0:
        spikes = []
        for i, t in enumerate(frame_times_ms):
            if t > spike_threshold_ms:
                spikes.append({"frame_index": i, "duration_ms": round(t, 2)})
                if len(spikes) >= 50:
                    break
        spike_total = sum(1 for t in frame_times_ms if t > spike_threshold_ms)
        result["frame_spikes"] = spikes
        result["spike_count"] = spike_total
        result["spike_pct"] = round(
            (spike_total / total) * 100, 1
        ) if total > 0 else 0.0

    # Raw frame times (truncated for large captures)
    max_raw = 2000
    result["frame_times_ms"] = [round(t, 2) for t in frame_times_ms[:max_raw]]
    if len(frame_times_ms) > max_raw:
        result["frame_times_truncated"] = True
        result["frame_times_total"] = len(frame_times_ms)

    return result


# ---------------------------------------------------------------------------
# Bookmark analysis
# ---------------------------------------------------------------------------

def _analyze_bookmarks(capture: UTraceCapture, top_n: int) -> dict | None:
    if not capture.bookmarks:
        return None

    freq = capture.cpu_cycle_frequency

    by_name: dict[str, list[int]] = defaultdict(list)
    timeline: list[dict] = []

    for bm in capture.bookmarks:
        bp = bm.get("bookmark_point", 0)
        ts = bm.get("timestamp", 0)
        name = capture.bookmark_specs.get(bp, f"Bookmark_{bp}")

        by_name[name].append(ts)
        timeline.append({
            "name": name,
            "timestamp": ts,
            "time_ms": round(_ticks_to_ms(ts, freq), 3) if freq > 0 else 0,
        })

    timeline.sort(key=lambda x: x["timestamp"])

    bookmark_counts = sorted(
        [{"name": name, "count": len(timestamps)}
         for name, timestamps in by_name.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:top_n]

    return {
        "bookmark_count": len(capture.bookmarks),
        "bookmark_spec_count": len(capture.bookmark_specs),
        "bookmark_counts_by_name": bookmark_counts,
        "bookmark_timeline": timeline[:200],
        "bookmark_timeline_truncated": len(timeline) > 200,
    }
