"""Parser for Optick .opt binary capture files.

Binary format reverse-engineered from OptickPlugin C++ source:
- optick_server.cpp:251-266  — OptickHeader (magic, version, flags)
- optick_server.cpp:306-383  — ZLibCompressor (zlib deflate)
- optick_message.h:17-51     — DataResponse header and chunk types
- optick_core.cpp:989-1177   — DumpFrames/DumpBoard/DumpSummary serialization
- optick_serialization.cpp   — primitive type encoding (strings = u32 len + bytes)
"""

from __future__ import annotations

import gzip
import io
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

# --- Constants from optick_message.h / optick_server.cpp ---

OPTICK_MAGIC = 0xB50FB50F
HEADER_FLAG_ISZIP = 0x01    # gzip compression
HEADER_FLAG_ISMINIZ = 0x02  # raw deflate compression

# DataResponse::Type enum
CHUNK_FRAME_DESCRIPTION_BOARD = 0
CHUNK_EVENT_FRAME = 1
CHUNK_NULL_FRAME = 3
CHUNK_SYNCHRONIZATION_DATA = 7
CHUNK_TAGS_PACK = 8
CHUNK_CALLSTACK_DESC_BOARD = 9
CHUNK_CALLSTACK_PACK = 10
CHUNK_SUMMARY_PACK = 258
CHUNK_FRAMES_PACK = 259


# --- Binary reader helpers matching optick_serialization.cpp ---

class BinaryReader:
    """Read binary data matching Optick's OutputDataStream encoding."""

    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    def remaining(self) -> int:
        return len(self._data) - self._pos

    def read_bytes(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise EOFError(f"Need {n} bytes at offset {self._pos}, only {self.remaining()} left")
        result = self._data[self._pos : self._pos + n]
        self._pos += n
        return result

    def skip(self, n: int) -> None:
        self._pos += n

    def read_u8(self) -> int:
        return struct.unpack_from("<B", self._data, self._advance(1))[0]

    def read_i8(self) -> int:
        return struct.unpack_from("<b", self._data, self._advance(1))[0]

    def read_u16(self) -> int:
        return struct.unpack_from("<H", self._data, self._advance(2))[0]

    def read_i16(self) -> int:
        return struct.unpack_from("<h", self._data, self._advance(2))[0]

    def read_u32(self) -> int:
        return struct.unpack_from("<I", self._data, self._advance(4))[0]

    def read_i32(self) -> int:
        return struct.unpack_from("<i", self._data, self._advance(4))[0]

    def read_u64(self) -> int:
        return struct.unpack_from("<Q", self._data, self._advance(8))[0]

    def read_i64(self) -> int:
        return struct.unpack_from("<q", self._data, self._advance(8))[0]

    def read_f32(self) -> float:
        return struct.unpack_from("<f", self._data, self._advance(4))[0]

    def read_string(self) -> str:
        """String encoding: u32 length + raw bytes (no null terminator)."""
        length = self.read_u32()
        if length == 0:
            return ""
        raw = self.read_bytes(length)
        return raw.decode("utf-8", errors="replace")

    def _advance(self, n: int) -> int:
        pos = self._pos
        if pos + n > len(self._data):
            raise EOFError(f"Need {n} bytes at offset {pos}")
        self._pos = pos + n
        return pos


# --- Parsed data structures ---

@dataclass
class EventDescription:
    """Matches C++ EventDescription serialization: name, file, line, filter, color, budget, flags."""
    index: int
    name: str
    file: str
    line: int
    filter: int
    color: int
    flags: int


@dataclass
class ThreadInfo:
    """Matches C++ ThreadDescription: threadID, processID, name, maxDepth, priority, mask."""
    thread_id: int
    process_id: int
    name: str
    max_depth: int
    priority: int
    mask: int


@dataclass
class ScopeEvent:
    """Single profiling event with timing and description index."""
    start: int
    finish: int
    description_index: int


@dataclass
class ScopeBlock:
    """A block of events for one thread in one frame."""
    board_number: int
    thread_number: int
    fiber_number: int
    event_start: int
    event_finish: int
    frame_type: int
    categories: list[ScopeEvent]
    events: list[ScopeEvent]


@dataclass
class OptickCapture:
    """All parsed data from an .opt file."""
    # From SummaryPack
    frame_times_ms: list[float] = field(default_factory=list)
    summary: dict[str, str] = field(default_factory=dict)

    # From FrameDescriptionBoard
    cpu_frequency: int = 0
    threads: list[ThreadInfo] = field(default_factory=list)
    event_descriptions: list[EventDescription] = field(default_factory=list)

    # From EventFrame chunks
    scope_blocks: list[ScopeBlock] = field(default_factory=list)

    # Raw frames from FramesPack
    frame_events: list[list[ScopeEvent]] = field(default_factory=list)


# --- Chunk parsers ---

def _parse_summary_pack(reader: BinaryReader, capture: OptickCapture) -> None:
    """Parse SummaryPack: boardNumber, frame times, summary k/v pairs, attachments."""
    _board_number = reader.read_u32()

    # Frame times as floats (ms)
    frame_count = reader.read_u32()
    for _ in range(frame_count):
        capture.frame_times_ms.append(reader.read_f32())

    # Summary key-value pairs
    summary_count = reader.read_u32()
    for _ in range(summary_count):
        key = reader.read_string()
        value = reader.read_string()
        capture.summary[key] = value

    # Attachments (type, name, data) — skip for MVP
    attachment_count = reader.read_u32()
    for _ in range(attachment_count):
        _att_type = reader.read_u32()
        _att_name = reader.read_string()
        # Attachment data is a string (binary blob as length-prefixed bytes)
        data_len = reader.read_u32()
        reader.skip(data_len)


def _parse_event_description(reader: BinaryReader, index: int) -> EventDescription:
    """Parse single EventDescription: name, file, line, filter, color, budget(float), flags(byte)."""
    name = reader.read_string()
    file = reader.read_string()
    line = reader.read_u32()
    filter_val = reader.read_u32()
    color = reader.read_u32()
    _budget = reader.read_f32()  # always 0.0f in serialization
    flags = reader.read_u8()
    return EventDescription(index=index, name=name, file=file, line=line,
                            filter=filter_val, color=color, flags=flags)


def _parse_thread_description(reader: BinaryReader) -> ThreadInfo:
    """Parse ThreadDescription: threadID(u64), processID(u32), name, maxDepth(i32), priority(i32), mask(u32)."""
    thread_id = reader.read_u64()
    process_id = reader.read_u32()
    name = reader.read_string()
    max_depth = reader.read_i32()
    priority = reader.read_i32()
    mask = reader.read_u32()
    return ThreadInfo(thread_id=thread_id, process_id=process_id, name=name,
                      max_depth=max_depth, priority=priority, mask=mask)


def _parse_frame_description_board(reader: BinaryReader, capture: OptickCapture) -> None:
    """Parse FrameDescriptionBoard — see optick_core.cpp:1152-1177 DumpBoard()."""
    _board_number = reader.read_u32()
    capture.cpu_frequency = reader.read_u64()
    _origin = reader.read_u64()
    _precision = reader.read_u32()

    # timeSlice (EventTime: start i64, finish i64)
    _time_start = reader.read_i64()
    _time_finish = reader.read_i64()

    # threads: vector<ThreadEntry*> — serialized as vector of ThreadDescription
    thread_count = reader.read_u32()
    for _ in range(thread_count):
        capture.threads.append(_parse_thread_description(reader))

    # fibers: vector<FiberEntry*> — serialized as vector of fiber IDs (u64 each)
    fiber_count = reader.read_u32()
    for _ in range(fiber_count):
        _fiber_id = reader.read_u64()

    # forcedMainThreadIndex
    _forced_main = reader.read_u32()

    # EventDescriptionBoard: vector<EventDescription*>
    desc_count = reader.read_u32()
    for i in range(desc_count):
        capture.event_descriptions.append(_parse_event_description(reader, i))

    # Remaining fields: tags(u32), run(u32), filters(u32), threadDescs(u32), mode(u32),
    # processDescs(vector), threadDescs(vector), processID(u32), hardwareConcurrency(u32)
    # Skip these — not needed for performance analysis MVP


def _read_scope_event(reader: BinaryReader) -> ScopeEvent:
    """EventData = EventTime(start i64, finish i64) + description index (u32)."""
    start = reader.read_i64()
    finish = reader.read_i64()
    desc_idx = reader.read_u32()
    return ScopeEvent(start=start, finish=finish, description_index=desc_idx)


def _parse_event_frame(reader: BinaryReader, capture: OptickCapture) -> None:
    """Parse EventFrame (ScopeData): header + categories + events."""
    # ScopeHeader: boardNumber(u32), threadNumber(i32), fiberNumber(i32),
    #              event(EventTime: start i64, finish i64), type(i32 — FrameType)
    board_number = reader.read_u32()
    thread_number = reader.read_i32()
    fiber_number = reader.read_i32()
    event_start = reader.read_i64()
    event_finish = reader.read_i64()
    frame_type = reader.read_i32()

    # categories: vector<EventData>
    cat_count = reader.read_u32()
    categories = [_read_scope_event(reader) for _ in range(cat_count)]

    # events: vector<EventData>
    event_count = reader.read_u32()
    events = [_read_scope_event(reader) for _ in range(event_count)]

    capture.scope_blocks.append(ScopeBlock(
        board_number=board_number,
        thread_number=thread_number,
        fiber_number=fiber_number,
        event_start=event_start,
        event_finish=event_finish,
        frame_type=frame_type,
        categories=categories,
        events=events,
    ))


def _parse_frames_pack(reader: BinaryReader, capture: OptickCapture) -> None:
    """Parse FramesPack: boardNumber, then per-FrameType arrays of FrameData."""
    _board_number = reader.read_u32()
    frame_type_count = reader.read_u32()
    for _ in range(frame_type_count):
        # Each frame type has a vector of FrameData (EventData + threadID)
        # FrameData = EventTime(i64 start, i64 finish) + descIndex(u32) + threadID(u64)
        frame_count = reader.read_u32()
        frames = []
        for _ in range(frame_count):
            start = reader.read_i64()
            finish = reader.read_i64()
            desc_idx = reader.read_u32()
            _thread_id = reader.read_u64()
            frames.append(ScopeEvent(start=start, finish=finish, description_index=desc_idx))
        capture.frame_events.append(frames)


# --- Main parser ---

def _open_payload_stream(path: Path) -> io.RawIOBase:
    """Open an .opt file and return a readable stream over the decompressed payload.

    Validates the OptickHeader and selects the right decompression method.
    Uses streaming decompression to avoid loading multi-GB data into memory.
    """
    f = open(path, "rb")
    header = f.read(8)
    if len(header) < 8:
        f.close()
        raise ValueError(f"File too small: {len(header)} bytes")

    magic, version, flags = struct.unpack_from("<IHH", header, 0)
    if magic != OPTICK_MAGIC:
        f.close()
        raise ValueError(f"Invalid Optick magic: 0x{magic:08X} (expected 0x{OPTICK_MAGIC:08X})")

    if flags & HEADER_FLAG_ISZIP:
        # gzip — re-open from offset 8 via GzipFile over remaining bytes
        return gzip.GzipFile(fileobj=f)
    elif flags & HEADER_FLAG_ISMINIZ:
        # raw deflate — use DecompressObj streaming
        return _DeflateStream(f)
    else:
        # Uncompressed — just return the file (already past header)
        return f


class _DeflateStream(io.RawIOBase):
    """Streaming reader for raw-deflate compressed data."""

    def __init__(self, fileobj):
        self._file = fileobj
        self._decompressor = zlib.decompressobj(-15)
        self._buf = b""

    def readable(self):
        return True

    def readinto(self, b):
        while len(self._buf) < len(b):
            chunk = self._file.read(65536)
            if not chunk:
                if self._decompressor:
                    self._buf += self._decompressor.flush()
                    self._decompressor = None
                break
            self._buf += self._decompressor.decompress(chunk)

        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n

    def close(self):
        self._file.close()
        super().close()


def _read_exact(stream, n: int) -> bytes:
    """Read exactly n bytes from a stream, raising on short read."""
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if len(buf) == 0:
                return b""
            raise EOFError(f"Short read: wanted {n}, got {len(buf)}")
        buf += chunk
    return buf


def parse_opt_file(path: str | Path) -> OptickCapture:
    """Parse an Optick .opt file and return structured capture data.

    Uses streaming decompression to handle large captures (multi-GB) without
    loading the entire decompressed payload into memory.

    Args:
        path: Path to the .opt capture file.

    Returns:
        OptickCapture with frame timings, thread info, event descriptions, and scope data.

    Raises:
        ValueError: If file has invalid magic or unsupported format.
    """
    stream = _open_payload_stream(Path(path))
    capture = OptickCapture()

    try:
        while True:
            # DataResponse header: version(u32) + size(u32) + type(u16) + application(u16)
            header_bytes = _read_exact(stream, 12)
            if len(header_bytes) < 12:
                break

            _resp_version, chunk_size, chunk_type, _app_id = struct.unpack_from("<IIHH", header_bytes, 0)

            if chunk_type == CHUNK_NULL_FRAME:
                break

            # Read the full chunk payload
            chunk_data = _read_exact(stream, chunk_size)
            if len(chunk_data) < chunk_size:
                break

            reader = BinaryReader(chunk_data)

            try:
                if chunk_type == CHUNK_SUMMARY_PACK:
                    _parse_summary_pack(reader, capture)
                elif chunk_type == CHUNK_FRAME_DESCRIPTION_BOARD:
                    _parse_frame_description_board(reader, capture)
                elif chunk_type == CHUNK_EVENT_FRAME:
                    _parse_event_frame(reader, capture)
                elif chunk_type == CHUNK_FRAMES_PACK:
                    _parse_frames_pack(reader, capture)
                # else: skip (chunk_data already consumed from stream)
            except (EOFError, struct.error):
                pass  # Chunk parse error — skip to next chunk
    except EOFError:
        pass
    finally:
        stream.close()

    return capture


# --- Analysis helpers ---

def _resolve_thread_indices(
    capture: OptickCapture,
    thread_names: list[str] | None,
) -> set[int] | None:
    """Map thread name filters to thread indices.  Returns ``None`` when no
    filter is active (i.e. all threads should be included)."""
    if not thread_names or not capture.threads:
        return None
    lowered = {n.lower().replace(" ", "") for n in thread_names if n.strip()}
    if not lowered:
        return None
    indices: set[int] = set()
    for idx, t in enumerate(capture.threads):
        if t.name.lower().replace(" ", "") in lowered:
            indices.add(idx)
    return indices if indices else None


def analyze_capture(
    capture: OptickCapture,
    top_n: int = 20,
    thread_names: list[str] | None = None,
    scope_keywords: list[str] | None = None,
    per_thread_top_n: int = 0,
    spike_threshold_ms: float = 0.0,
) -> dict:
    """Produce an LLM-friendly analysis dict from parsed capture data.

    Args:
        capture: Parsed OptickCapture.
        top_n: Number of hottest scopes to include (global mode).
        thread_names: Only include scopes/threads matching these names (case-insensitive).
        scope_keywords: Only include scopes whose name contains any of these substrings.
        per_thread_top_n: When >0, group hottest scopes by thread (top N each)
            instead of returning a single global list.
        spike_threshold_ms: When >0, report frames exceeding this duration.

    Returns:
        Dict with summary stats, thread breakdown, hottest scopes, and raw frame times.
    """
    result: dict = {}
    thread_idx_filter = _resolve_thread_indices(capture, thread_names)
    kw_lower = [kw.lower() for kw in scope_keywords if kw.strip()] if scope_keywords else []

    # --- Frame timing summary ---
    if capture.frame_times_ms:
        times = capture.frame_times_ms
        sorted_times = sorted(times)
        total = len(times)
        avg = sum(times) / total
        p99_idx = min(int(total * 0.99), total - 1)
        p95_idx = min(int(total * 0.95), total - 1)

        result["frame_summary"] = {
            "total_frames": total,
            "avg_ms": round(avg, 2),
            "min_ms": round(sorted_times[0], 2),
            "max_ms": round(sorted_times[-1], 2),
            "median_ms": round(sorted_times[total // 2], 2),
            "p95_ms": round(sorted_times[p95_idx], 2),
            "p99_ms": round(sorted_times[p99_idx], 2),
            "frames_above_16ms": sum(1 for t in times if t > 16.667),
            "frames_above_33ms": sum(1 for t in times if t > 33.333),
        }

    # --- Summary metadata ---
    if capture.summary:
        result["metadata"] = dict(capture.summary)

    # --- Thread list ---
    if capture.threads:
        result["threads"] = [
            {"name": t.name, "thread_id": t.thread_id, "mask": t.mask}
            for t in capture.threads
        ]

    # --- Hottest scopes (aggregate events by description) ---
    if capture.scope_blocks and capture.event_descriptions and capture.cpu_frequency > 0:
        freq = capture.cpu_frequency
        desc_map = {d.index: d for d in capture.event_descriptions}

        # Accumulate per-description, optionally per-thread
        # Key: (thread_number, desc_idx) when per_thread_top_n > 0, else (None, desc_idx)
        accum: dict[tuple[int | None, int], list] = {}
        for block in capture.scope_blocks:
            if thread_idx_filter is not None and block.thread_number not in thread_idx_filter:
                continue
            thread_key = block.thread_number if per_thread_top_n > 0 else None
            for ev in block.events:
                dur = ev.finish - ev.start
                if dur <= 0 or ev.description_index == 0xFFFFFFFF:
                    continue
                key = (thread_key, ev.description_index)
                entry = accum.get(key)
                if entry is None:
                    accum[key] = [dur, 1]
                else:
                    entry[0] += dur
                    entry[1] += 1

        def _build_scope_entry(desc_idx: int, total_ticks: int, count: int) -> dict | None:
            desc = desc_map.get(desc_idx)
            if not desc:
                return None
            if kw_lower and not any(kw in desc.name.lower() for kw in kw_lower):
                return None
            total_ms = (total_ticks / freq) * 1000.0
            avg_ms = total_ms / count
            return {
                "name": desc.name,
                "file": desc.file,
                "line": desc.line,
                "total_ms": round(total_ms, 3),
                "avg_ms": round(avg_ms, 3),
                "calls": count,
            }

        if per_thread_top_n > 0:
            # Group by thread
            thread_groups: dict[int, list[dict]] = {}
            for (tidx, desc_idx), (total_ticks, count) in accum.items():
                entry = _build_scope_entry(desc_idx, total_ticks, count)
                if entry is None:
                    continue
                assert tidx is not None
                thread_groups.setdefault(tidx, []).append(entry)

            per_thread_scopes: dict[str, list[dict]] = {}
            for tidx, scopes in thread_groups.items():
                name = capture.threads[tidx].name if 0 <= tidx < len(capture.threads) else f"Thread#{tidx}"
                scopes.sort(key=lambda x: x["total_ms"], reverse=True)
                per_thread_scopes[name] = scopes[:per_thread_top_n]
            result["per_thread_scopes"] = per_thread_scopes
        else:
            # Global hottest scopes
            hottest = []
            for (_tidx, desc_idx), (total_ticks, count) in accum.items():
                entry = _build_scope_entry(desc_idx, total_ticks, count)
                if entry is not None:
                    hottest.append(entry)
            hottest.sort(key=lambda x: x["total_ms"], reverse=True)
            result["hottest_scopes"] = hottest[:top_n]

    # --- Per-thread time breakdown ---
    if capture.scope_blocks and capture.threads and capture.cpu_frequency > 0:
        freq = capture.cpu_frequency
        thread_totals: dict[int, float] = {}
        for block in capture.scope_blocks:
            if thread_idx_filter is not None and block.thread_number not in thread_idx_filter:
                continue
            dur_ticks = block.event_finish - block.event_start
            if dur_ticks > 0:
                dur_ms = (dur_ticks / freq) * 1000.0
                thread_totals[block.thread_number] = thread_totals.get(block.thread_number, 0.0) + dur_ms

        thread_breakdown = []
        for tidx, total_ms in sorted(thread_totals.items(), key=lambda x: x[1], reverse=True):
            name = capture.threads[tidx].name if 0 <= tidx < len(capture.threads) else f"Thread#{tidx}"
            thread_breakdown.append({"name": name, "total_ms": round(total_ms, 2)})
        result["thread_breakdown"] = thread_breakdown

    # --- Frame spike detection ---
    if spike_threshold_ms > 0 and capture.frame_times_ms:
        max_spikes = 50
        spikes = []
        for i, t in enumerate(capture.frame_times_ms):
            if t > spike_threshold_ms:
                spikes.append({"frame_index": i, "duration_ms": round(t, 2)})
                if len(spikes) >= max_spikes:
                    break
        total_frames = len(capture.frame_times_ms)
        spike_total = sum(1 for t in capture.frame_times_ms if t > spike_threshold_ms)
        result["frame_spikes"] = spikes
        result["spike_count"] = spike_total
        result["spike_pct"] = round((spike_total / total_frames) * 100, 1) if total_frames else 0.0

    # --- Raw frame times (truncated for large captures) ---
    if capture.frame_times_ms:
        max_raw = 2000
        result["frame_times_ms"] = [
            round(t, 2) for t in capture.frame_times_ms[:max_raw]
        ]
        if len(capture.frame_times_ms) > max_raw:
            result["frame_times_truncated"] = True
            result["frame_times_total"] = len(capture.frame_times_ms)

    return result
