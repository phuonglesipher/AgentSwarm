"""Parser for Unreal Insights .utrace binary trace files.

Binary format reverse-engineered from UE 5.7.1 engine source:
- TraceLog/Public/Trace/Detail/Transport.h         — FTidPacketBase, ETransport
- TraceLog/Public/Trace/Detail/Protocols/Protocol5.h — FEventHeader, FAuxHeader, packed UIDs
- TraceLog/Public/Trace/Detail/Protocols/Protocol7.h — EKnownEventUids (current)
- TraceLog/Private/Trace/Writer.cpp:768-798         — Handshake/magic + transport header
- TraceAnalysis/Private/Analysis/Engine.cpp:5304-5360 — Magic stage (TRCE/TRC2)
- EpicGames.Tracing C# reference implementation     — Reader.cs, GenericEvent.cs, TransportPacket.cs

Trace channels parsed:
- CpuProfiler   — scope enter/leave with timestamps, spec names
- GpuProfiler   — breadcrumbs, work begin/end, queue specs
- Memory        — alloc/free/realloc, heap/tag specs
- Counters      — named counter values over time
- Frame         — frame begin/end boundaries
- Diagnostics   — session info (platform, command line)
- Misc          — bookmarks
- Log           — text log messages
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

try:
    import lz4.block as _lz4
except ImportError:
    _lz4 = None
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC_TRCE = 0x54524345  # MSVC multi-char 'TRCE' = 0x54524345
MAGIC_TRC2 = 0x54524332  # MSVC multi-char 'TRC2' / Writer: '2'|'C'<<8|'R'<<16|'T'<<24

# Transport versions (ETransport)
TRANSPORT_RAW = 1
TRANSPORT_PACKET = 2
TRANSPORT_TID_PACKET = 3
TRANSPORT_TID_PACKET_SYNC = 4

# TidPacket bit masks
_ENCODED_MARKER = 0x8000
_VERIFICATION_MARKER = 0x4000
_THREAD_ID_MASK = 0x3FFF

# Well-known thread IDs
THREAD_EVENTS = 0       # event type definitions
THREAD_IMPORTANTS = 1   # important/cached events
THREAD_BIAS = 2         # first real thread

# Well-known event UIDs (Protocol5+, after >>1 shift)
UID_NEW_EVENT = 0
UID_AUX_DATA = 1
UID_AUX_DATA_TERMINAL = 3
UID_ENTER_SCOPE = 4
UID_LEAVE_SCOPE = 5
UID_ENTER_SCOPE_T = 8
UID_LEAVE_SCOPE_T = 12
UID_WELL_KNOWN_NUM = 16

# EventType flags
FLAG_IMPORTANT = 0x01
FLAG_MAYBE_HAS_AUX = 0x02
FLAG_NO_SYNC = 0x04

# TypeInfo octal encoding
_FIELD_CATEGORY_MASK = 0o300
_FIELD_INTEGER = 0o000
_FIELD_FLOAT = 0o100
_FIELD_ARRAY = 0o200
_FIELD_POW2_SIZE_MASK = 0o003
_FIELD_SPECIAL_MASK = 0o030
_FIELD_STRING = 0o010
_FIELD_SIGNED = 0o020

# Derived type constants
TYPE_BOOL = 0o000
TYPE_INT8 = 0o000
TYPE_INT16 = 0o001
TYPE_INT32 = 0o002
TYPE_INT64 = 0o003
TYPE_UINT8 = 0o000
TYPE_UINT16 = 0o001
TYPE_UINT32 = 0o002
TYPE_UINT64 = 0o003
TYPE_POINTER = 0o003
TYPE_FLOAT32 = 0o102
TYPE_FLOAT64 = 0o103
TYPE_ANSI_STRING = 0o210
TYPE_WIDE_STRING = 0o211
TYPE_ARRAY = 0o200

# struct format for fixed field types
_TYPE_TO_STRUCT: dict[int, tuple[str, int]] = {
    0o000: ("<B", 1),   # uint8 / bool
    0o001: ("<H", 2),   # uint16
    0o002: ("<I", 4),   # uint32
    0o003: ("<Q", 8),   # uint64 / pointer
    0o020: ("<b", 1),   # int8
    0o021: ("<h", 2),   # int16
    0o022: ("<i", 4),   # int32
    0o023: ("<q", 8),   # int64
    0o102: ("<f", 4),   # float32
    0o103: ("<d", 8),   # float64
}


# ---------------------------------------------------------------------------
# Binary reader
# ---------------------------------------------------------------------------

class BinaryReader:
    """Read little-endian binary data with position tracking."""

    __slots__ = ("_data", "_pos")

    def __init__(self, data: bytes | memoryview) -> None:
        self._data = data
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    @pos.setter
    def pos(self, value: int) -> None:
        self._pos = value

    def remaining(self) -> int:
        return len(self._data) - self._pos

    def read_bytes(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise EOFError(f"Need {n} bytes at offset {self._pos}, only {self.remaining()} left")
        result = self._data[self._pos : self._pos + n]
        self._pos += n
        return bytes(result) if isinstance(result, memoryview) else result

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

    def read_f64(self) -> float:
        return struct.unpack_from("<d", self._data, self._advance(8))[0]

    def read_packed_uid(self) -> int:
        """Read a packed UID (1 or 2 bytes).

        Format: bit0 of first byte = is_two_byte_uid.
        If set, read second byte. UID = (low >> 1) | (high << 7).
        """
        low = self.read_u8()
        is_two_byte = low & 1
        uid_low = low >> 1
        if is_two_byte:
            uid_high = self.read_u8()
            return uid_low | (uid_high << 7)
        return uid_low

    def read_varint7(self) -> int:
        """Read a 7-bit variable-length unsigned integer."""
        value = 0
        shift = 0
        while True:
            b = self.read_u8()
            value |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        return value

    def _advance(self, n: int) -> int:
        pos = self._pos
        if pos + n > len(self._data):
            raise EOFError(f"Need {n} bytes at offset {pos}")
        self._pos = pos + n
        return pos


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class FieldDef(NamedTuple):
    """Field definition within an event type."""
    name: str
    offset: int
    size: int        # 0 for array/string (aux data)
    type_info: int   # octal-encoded category|special|size


@dataclass
class EventType:
    """Declared event type from a NewEvent on thread 0."""
    uid: int
    logger_name: str
    event_name: str
    fields: list[FieldDef]
    flags: int

    @property
    def is_important(self) -> bool:
        return bool(self.flags & FLAG_IMPORTANT)

    @property
    def maybe_has_aux(self) -> bool:
        return bool(self.flags & FLAG_MAYBE_HAS_AUX)

    @property
    def is_no_sync(self) -> bool:
        return bool(self.flags & FLAG_NO_SYNC)

    @property
    def has_serial(self) -> bool:
        return not self.is_no_sync and not self.is_important

    @property
    def fixed_size(self) -> int:
        return sum(f.size for f in self.fields)

    @property
    def full_name(self) -> str:
        return f"{self.logger_name}.{self.event_name}"


@dataclass
class UTraceCapture:
    """All parsed data from a .utrace file."""

    # Stream metadata
    magic: str = ""
    transport_version: int = 0
    protocol_version: int = 0

    # Event type registry (uid -> EventType)
    event_types: dict[int, EventType] = field(default_factory=dict)

    # CPU profiler
    cpu_scope_specs: dict[int, str] = field(default_factory=dict)
    cpu_scopes: list[tuple[int, int, int, bool]] = field(default_factory=list)
    # (thread_id, spec_id, timestamp, is_enter)
    cpu_cycle_frequency: int = 0
    # Per-thread running timestamp for EventBatch delta decoding (carries across batches)
    _thread_cpu_timestamps: dict[int, int] = field(default_factory=dict)

    # GPU profiler
    gpu_events: list[dict] = field(default_factory=list)
    gpu_queue_names: dict[int, str] = field(default_factory=dict)
    gpu_breadcrumb_specs: dict[int, str] = field(default_factory=dict)

    # Memory
    memory_events: list[dict] = field(default_factory=list)
    memory_events_truncated: bool = False
    heap_specs: dict[int, str] = field(default_factory=dict)
    tag_specs: dict[int, str] = field(default_factory=dict)
    memory_size_shift: int = 0

    # Counters
    counter_specs: dict[int, str] = field(default_factory=dict)
    counter_values: list[dict] = field(default_factory=list)

    # Frame timing
    frame_boundaries: list[tuple[int, int, bool]] = field(default_factory=list)
    # (frame_type, cycle, is_begin)

    # Session info
    session_info: dict[str, str] = field(default_factory=dict)

    # Bookmarks
    bookmark_specs: dict[int, str] = field(default_factory=dict)
    bookmarks: list[dict] = field(default_factory=list)

    # Log messages
    log_messages: list[dict] = field(default_factory=list)

    # Thread info
    thread_info: dict[int, str] = field(default_factory=dict)

    # Parsing stats
    total_packets: int = 0
    total_events: int = 0
    unknown_event_count: int = 0
    parse_errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Event field reading helpers
# ---------------------------------------------------------------------------

def _read_fixed_field(reader: BinaryReader, type_info: int) -> int | float:
    """Read a fixed-size field value based on its type_info."""
    key = type_info & (_FIELD_CATEGORY_MASK | _FIELD_SPECIAL_MASK | _FIELD_POW2_SIZE_MASK)

    # Check for array/string types first — these are aux data, skip
    if type_info & _FIELD_ARRAY:
        return 0  # placeholder, real data in aux

    fmt_info = _TYPE_TO_STRUCT.get(key)
    if fmt_info:
        fmt, size = fmt_info
        return struct.unpack_from(fmt, reader._data, reader._advance(size))[0]

    # Fallback: try to determine size from low bits
    size_bits = type_info & _FIELD_POW2_SIZE_MASK
    byte_size = 1 << size_bits
    reader.skip(byte_size)
    return 0


def _read_event_fields(reader: BinaryReader, event_type: EventType) -> dict[str, int | float | str | bytes]:
    """Read all fields of an event, including aux data."""
    fields: dict[str, int | float | str | bytes] = {}

    # Read fixed fields
    for f in event_type.fields:
        if f.size == 0:
            # Array/string field — data comes in aux blocks
            fields[f.name] = b""
            continue
        fields[f.name] = _read_fixed_field(reader, f.type_info)

    # Read aux data if present
    if event_type.maybe_has_aux:
        _read_aux_data(reader, event_type, fields)

    return fields


def _read_aux_data(
    reader: BinaryReader,
    event_type: EventType,
    fields: dict[str, int | float | str | bytes],
) -> None:
    """Read aux data blocks until AuxDataTerminal."""
    while reader.remaining() > 0:
        # For important events, UIDs are not shifted
        # For normal events, well-known UIDs are shifted left by 1
        first_byte = reader.read_u8()

        if event_type.is_important:
            aux_uid = first_byte
        else:
            aux_uid = first_byte >> 1

        if aux_uid == UID_AUX_DATA_TERMINAL:
            break

        if aux_uid == UID_AUX_DATA:
            # Read rest of aux header as u32 (we already read first byte)
            reader.pos -= 1
            aux_header = reader.read_u32()
            field_index = (aux_header >> 8) & 0x1F
            data_size = (aux_header >> 13) & 0x7FFFF

            aux_bytes = reader.read_bytes(data_size)

            if field_index < len(event_type.fields):
                f = event_type.fields[field_index]
                ti = f.type_info
                if ti == TYPE_ANSI_STRING:
                    fields[f.name] = aux_bytes.decode("ascii", errors="replace")
                elif ti == TYPE_WIDE_STRING:
                    fields[f.name] = aux_bytes.decode("utf-16-le", errors="replace")
                else:
                    fields[f.name] = aux_bytes
        else:
            # Unknown aux UID — bail out to avoid misalignment
            break


# ---------------------------------------------------------------------------
# NewEvent parsing (thread 0)
# ---------------------------------------------------------------------------

def _parse_new_event(reader: BinaryReader, protocol_version: int = 7) -> EventType:
    """Parse a NewEvent declaration.

    Protocol 0-5: field descriptors are 6 bytes each (offset, size, type, nameSize).
    Protocol 6+:  field descriptors are 7 bytes each (fieldFamily + offset, size, type, nameSize).
    """
    event_uid = reader.read_u16()
    field_count = reader.read_u8()
    flags = reader.read_u8()
    logger_name_size = reader.read_u8()
    event_name_size = reader.read_u8()

    # Read field descriptors
    # Protocol 6+: each field has a FieldType byte + padding + 6-byte union = 8 bytes
    # Protocol 0-5: each field is 6 bytes (Offset, Size, TypeInfo, NameSize)
    raw_fields = []
    for _ in range(field_count):
        if protocol_version >= 6:
            _field_family = reader.read_u8()  # Regular=0, Reference=1, DefinitionId=2
            reader.skip(1)  # padding byte for uint16 alignment
        offset = reader.read_u16()
        size = reader.read_u16()
        type_info = reader.read_u8()
        name_size = reader.read_u8()
        raw_fields.append((offset, size, type_info, name_size))

    # Read names
    logger_name = reader.read_bytes(logger_name_size).decode("utf-8", errors="replace")
    event_name = reader.read_bytes(event_name_size).decode("utf-8", errors="replace")

    fields = []
    for offset, size, type_info, name_size in raw_fields:
        name = reader.read_bytes(name_size).decode("utf-8", errors="replace")
        fields.append(FieldDef(name=name, offset=offset, size=size, type_info=type_info))

    return EventType(
        uid=event_uid,
        logger_name=logger_name,
        event_name=event_name,
        fields=fields,
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Channel handlers
# ---------------------------------------------------------------------------

def _handle_cpu_event_spec(fields: dict, capture: UTraceCapture, **_) -> None:
    spec_id = fields.get("Id", 0)
    name = fields.get("Name", "")
    if isinstance(spec_id, (int, float)):
        capture.cpu_scope_specs[int(spec_id)] = str(name)


def _handle_cpu_event_batch(
    fields: dict, capture: UTraceCapture, thread_id: int, **_
) -> None:
    data = fields.get("Data", b"")
    if not isinstance(data, (bytes, bytearray)) or len(data) == 0:
        return

    reader = BinaryReader(data)
    last_timestamp = 0
    first = True

    try:
        while reader.remaining() > 0:
            value = reader.read_varint7()
            is_enter = bool(value & 1)
            timestamp = value >> 1

            if first:
                # First timestamp per batch is absolute (QPC cycle count).
                last_timestamp = timestamp
                first = False

                # Detect pre-trace batches (timestamp before session start
                # or backwards from the thread's last known good timestamp).
                prev_good = capture._thread_cpu_timestamps.get(thread_id, 0)
                if prev_good > 0 and timestamp < prev_good - 10_000_000:
                    # Batch timestamp is more than 1 second before the last
                    # good timestamp — skip entire batch (likely pre-trace
                    # init events or stale buffer data).
                    return
            else:
                # Delta encoding from previous event
                timestamp = last_timestamp + timestamp
                last_timestamp = timestamp

            spec_id = 0
            if is_enter:
                spec_id = reader.read_varint7()

            capture.cpu_scopes.append((thread_id, spec_id, timestamp, is_enter))
            capture.total_events += 1
    except EOFError:
        pass  # Truncated batch is fine

    # Track last valid timestamp for anomaly detection in subsequent batches
    if last_timestamp > capture._thread_cpu_timestamps.get(thread_id, 0):
        capture._thread_cpu_timestamps[thread_id] = last_timestamp


def _handle_memory_init(fields: dict, capture: UTraceCapture, **_) -> None:
    capture.memory_size_shift = int(fields.get("SizeShift", 0))
    capture.memory_events.append({"kind": "init", **{k: v for k, v in fields.items()}})


def _handle_memory_alloc(
    fields: dict, capture: UTraceCapture, max_memory_events: int, **_
) -> None:
    if len(capture.memory_events) >= max_memory_events:
        capture.memory_events_truncated = True
        return
    capture.memory_events.append({
        "kind": "alloc",
        "address": fields.get("Address", fields.get("Ptr", 0)),
        "size": fields.get("Size", 0),
        "callstack_id": fields.get("CallstackId", 0),
    })


def _handle_memory_free(
    fields: dict, capture: UTraceCapture, max_memory_events: int, **_
) -> None:
    if len(capture.memory_events) >= max_memory_events:
        capture.memory_events_truncated = True
        return
    capture.memory_events.append({
        "kind": "free",
        "address": fields.get("Address", fields.get("Ptr", 0)),
    })


def _handle_memory_realloc_alloc(
    fields: dict, capture: UTraceCapture, max_memory_events: int, **_
) -> None:
    if len(capture.memory_events) >= max_memory_events:
        capture.memory_events_truncated = True
        return
    capture.memory_events.append({
        "kind": "realloc_alloc",
        "address": fields.get("Address", fields.get("Ptr", 0)),
        "size": fields.get("Size", 0),
        "callstack_id": fields.get("CallstackId", 0),
    })


def _handle_memory_realloc_free(
    fields: dict, capture: UTraceCapture, max_memory_events: int, **_
) -> None:
    if len(capture.memory_events) >= max_memory_events:
        capture.memory_events_truncated = True
        return
    capture.memory_events.append({
        "kind": "realloc_free",
        "address": fields.get("Address", fields.get("Ptr", 0)),
    })


def _handle_heap_spec(fields: dict, capture: UTraceCapture, **_) -> None:
    heap_id = int(fields.get("Id", 0))
    name = str(fields.get("Name", f"Heap_{heap_id}"))
    capture.heap_specs[heap_id] = name


def _handle_tag_spec(fields: dict, capture: UTraceCapture, **_) -> None:
    tag_id = int(fields.get("Id", 0))
    name = str(fields.get("Name", f"Tag_{tag_id}"))
    capture.tag_specs[tag_id] = name


def _handle_memory_marker(fields: dict, capture: UTraceCapture, **_) -> None:
    # Memory marker provides cycle/time correlation
    pass


def _handle_gpu_queue_spec(fields: dict, capture: UTraceCapture, **_) -> None:
    queue_id = int(fields.get("QueueId", fields.get("Id", 0)))
    name = str(fields.get("TypeString", fields.get("Name", f"Queue_{queue_id}")))
    capture.gpu_queue_names[queue_id] = name


def _handle_gpu_breadcrumb_spec(fields: dict, capture: UTraceCapture, **_) -> None:
    spec_id = int(fields.get("SpecId", fields.get("Id", 0)))
    name = str(fields.get("StaticName", fields.get("NameFormat", fields.get("Name", ""))))
    capture.gpu_breadcrumb_specs[spec_id] = name


def _handle_gpu_event(
    fields: dict, capture: UTraceCapture, event_name: str, **_
) -> None:
    # Begin events use GPUTimestampTOP, End events use GPUTimestampBOP,
    # EventWait uses StartTime, fences use CPUTimestamp
    timestamp = fields.get(
        "GPUTimestampTOP", fields.get(
        "GPUTimestampBOP", fields.get(
        "CPUTimestamp", fields.get(
        "StartTime", 0))))
    capture.gpu_events.append({
        "kind": event_name,
        "queue_id": fields.get("QueueId", 0),
        "timestamp": timestamp,
        "spec_id": fields.get("SpecId", None),
    })


def _handle_counter_spec(fields: dict, capture: UTraceCapture, **_) -> None:
    counter_id = int(fields.get("Id", 0))
    name = str(fields.get("Name", f"Counter_{counter_id}"))
    capture.counter_specs[counter_id] = name


def _handle_counter_set_value(fields: dict, capture: UTraceCapture, **_) -> None:
    capture.counter_values.append({
        "counter_id": int(fields.get("CounterId", fields.get("Id", 0))),
        "value": fields.get("Value", 0),
        "timestamp": fields.get("Cycle", 0),
    })


def _handle_frame_begin(fields: dict, capture: UTraceCapture, **_) -> None:
    capture.frame_boundaries.append((
        int(fields.get("FrameType", 0)),
        int(fields.get("Cycle", fields.get("Timestamp", 0))),
        True,
    ))


def _handle_frame_end(fields: dict, capture: UTraceCapture, **_) -> None:
    capture.frame_boundaries.append((
        int(fields.get("FrameType", 0)),
        int(fields.get("Cycle", fields.get("Timestamp", 0))),
        False,
    ))


def _handle_session(fields: dict, capture: UTraceCapture, **_) -> None:
    for key, value in fields.items():
        if isinstance(value, (str, int, float)):
            capture.session_info[key] = str(value)


def _handle_bookmark_spec(fields: dict, capture: UTraceCapture, **_) -> None:
    bookmark_point = fields.get("BookmarkPoint", fields.get("Id", 0))
    name = str(fields.get("FormatString", fields.get("Name", "")))
    capture.bookmark_specs[bookmark_point] = name


def _handle_bookmark(fields: dict, capture: UTraceCapture, **_) -> None:
    capture.bookmarks.append({
        "bookmark_point": fields.get("BookmarkPoint", fields.get("SpecId", 0)),
        "timestamp": fields.get("Cycle", 0),
    })


def _handle_log(fields: dict, capture: UTraceCapture, **_) -> None:
    capture.log_messages.append({
        "category": fields.get("Category", ""),
        "message": fields.get("Message", ""),
        "verbosity": fields.get("Verbosity", 0),
        "timestamp": fields.get("Cycle", 0),
    })


def _handle_new_trace_event(fields: dict, capture: UTraceCapture, **_) -> None:
    # Only accept the first valid frequency.  Later calls from regular threads
    # may dispatch here with misaligned data, producing garbage values.
    if capture.cpu_cycle_frequency > 0:
        return
    freq = fields.get("CycleFrequency", 0)
    if isinstance(freq, (int, float)) and 1_000_000 <= freq <= 100_000_000_000:
        capture.cpu_cycle_frequency = int(freq)


def _handle_thread_info_event(fields: dict, capture: UTraceCapture, **_) -> None:
    thread_id = int(fields.get("ThreadId", 0))
    name = str(fields.get("Name", ""))
    if name:
        capture.thread_info[thread_id] = name


# Channel dispatch table: (logger_name, event_name) -> handler
_CHANNEL_HANDLERS: dict[tuple[str, str], object] = {
    ("CpuProfiler", "EventSpec"): _handle_cpu_event_spec,
    ("CpuProfiler", "EventBatch"): _handle_cpu_event_batch,
    ("CpuProfiler", "EventBatchV3"): _handle_cpu_event_batch,
    ("Memory", "Init"): _handle_memory_init,
    ("Memory", "Alloc"): _handle_memory_alloc,
    ("Memory", "AllocSystem"): _handle_memory_alloc,
    ("Memory", "AllocVideo"): _handle_memory_alloc,
    ("Memory", "Free"): _handle_memory_free,
    ("Memory", "FreeSystem"): _handle_memory_free,
    ("Memory", "FreeVideo"): _handle_memory_free,
    ("Memory", "ReallocAlloc"): _handle_memory_realloc_alloc,
    ("Memory", "ReallocAllocSystem"): _handle_memory_realloc_alloc,
    ("Memory", "ReallocFree"): _handle_memory_realloc_free,
    ("Memory", "ReallocFreeSystem"): _handle_memory_realloc_free,
    ("Memory", "HeapSpec"): _handle_heap_spec,
    ("Memory", "HeapMarkAlloc"): _handle_memory_alloc,
    ("Memory", "TagSpec"): _handle_tag_spec,
    ("Memory", "MemScopeTag"): _handle_tag_spec,
    ("Memory", "Marker"): _handle_memory_marker,
    ("GpuProfiler", "QueueSpec"): _handle_gpu_queue_spec,
    ("GpuProfiler", "EventBreadcrumbSpec"): _handle_gpu_breadcrumb_spec,
    ("GpuProfiler", "EventBeginBreadcrumb"): _handle_gpu_event,
    ("GpuProfiler", "EventEndBreadcrumb"): _handle_gpu_event,
    ("GpuProfiler", "EventBeginWork"): _handle_gpu_event,
    ("GpuProfiler", "EventEndWork"): _handle_gpu_event,
    ("GpuProfiler", "EventWait"): _handle_gpu_event,
    ("GpuProfiler", "EventStats"): _handle_gpu_event,
    ("GpuProfiler", "EventFrameBoundary"): _handle_gpu_event,
    ("GpuProfiler", "Init"): _handle_gpu_event,
    ("GpuProfiler", "SignalFence"): _handle_gpu_event,
    ("GpuProfiler", "WaitFence"): _handle_gpu_event,
    ("Counters", "Spec"): _handle_counter_spec,
    ("Counters", "SetValueInt"): _handle_counter_set_value,
    ("Counters", "SetValueFloat"): _handle_counter_set_value,
    ("Counters", "SetValue"): _handle_counter_set_value,
    ("Counters", "IncrementValue"): _handle_counter_set_value,
    ("Frame", "BeginFrame"): _handle_frame_begin,
    ("Frame", "EndFrame"): _handle_frame_end,
    ("Misc", "BeginFrame"): _handle_frame_begin,
    ("Misc", "EndFrame"): _handle_frame_end,
    ("Diagnostics", "Session"): _handle_session,
    ("Diagnostics", "Session2"): _handle_session,
    ("Misc", "BookmarkSpec"): _handle_bookmark_spec,
    ("Misc", "Bookmark"): _handle_bookmark,
    ("Thread", "Info"): _handle_thread_info_event,
    ("$Trace", "ThreadInfo"): _handle_thread_info_event,
    ("$Trace", "NewTrace"): _handle_new_trace_event,
}


# ---------------------------------------------------------------------------
# Packet-level parsing
# ---------------------------------------------------------------------------

def _process_events_thread(
    data: bytes,
    capture: UTraceCapture,
    protocol_version: int = 7,
) -> None:
    """Process event type definitions from thread 0 (Events thread)."""
    reader = BinaryReader(data)
    while reader.remaining() >= 4:
        try:
            uid = reader.read_u16()
            size = reader.read_u16()

            if uid == UID_NEW_EVENT and size > 0:
                start = reader.pos
                event_type = _parse_new_event(reader, protocol_version)
                capture.event_types[event_type.uid] = event_type
                # Ensure we consumed exactly `size` bytes
                consumed = reader.pos - start
                if consumed < size:
                    reader.skip(size - consumed)
            else:
                # Skip unknown events on thread 0
                if size > 0:
                    reader.skip(size)
        except EOFError:
            break
        except Exception as e:
            capture.parse_errors.append(f"Thread0 parse error at offset {reader.pos}: {e}")
            break


def _process_important_events(
    data: bytes,
    capture: UTraceCapture,
    max_memory_events: int,
    protocol_version: int = 7,
) -> None:
    """Process important/cached events from thread 1."""
    reader = BinaryReader(data)
    while reader.remaining() >= 4:
        try:
            uid = reader.read_u16()
            size = reader.read_u16()

            if uid == UID_NEW_EVENT:
                # NewEvent can appear on importants thread too
                start = reader.pos
                event_type = _parse_new_event(reader, protocol_version)
                capture.event_types[event_type.uid] = event_type
                consumed = reader.pos - start
                if consumed < size:
                    reader.skip(size - consumed)
                continue

            event_type = capture.event_types.get(uid)
            if not event_type:
                # Skip unknown
                if size > 0:
                    reader.skip(size)
                capture.unknown_event_count += 1
                continue

            start = reader.pos
            fields = _read_event_fields(reader, event_type)
            consumed = reader.pos - start
            if consumed < size:
                reader.skip(size - consumed)

            capture.total_events += 1
            _dispatch_event(event_type, fields, capture, 1, max_memory_events)

        except EOFError:
            break
        except Exception as e:
            capture.parse_errors.append(f"Importants parse error at offset {reader.pos}: {e}")
            break


def _process_thread_events(
    data: bytes,
    thread_id: int,
    capture: UTraceCapture,
    max_memory_events: int,
) -> None:
    """Process events from a regular thread (thread_id >= THREAD_BIAS)."""
    reader = BinaryReader(data)
    while reader.remaining() > 0:
        try:
            uid = reader.read_packed_uid()

            # Well-known scope events
            if uid == UID_ENTER_SCOPE or uid == UID_LEAVE_SCOPE:
                # No data, just scope markers
                continue

            if uid == UID_ENTER_SCOPE_T:
                # 7-byte timestamp follows (already consumed 1-2 bytes for UID)
                # Actually for EnterScope_T in the packed UID stream,
                # the timestamp is encoded as remaining 7 bytes of a u64
                # Re-read: the C# code reads a full u64 and extracts UID from low byte
                # But our packed_uid already consumed the UID bytes.
                # In protocol5+, EnterScope_T has a 7-byte absolute timestamp
                ts_bytes = reader.read_bytes(7)
                timestamp = int.from_bytes(ts_bytes, "little")
                # We don't have a spec_id for raw scope events
                continue

            if uid == UID_LEAVE_SCOPE_T:
                ts_bytes = reader.read_bytes(7)
                continue

            if uid == UID_AUX_DATA or uid == UID_AUX_DATA_TERMINAL:
                # Shouldn't appear standalone, skip
                continue

            event_type = capture.event_types.get(uid)
            if not event_type:
                capture.unknown_event_count += 1
                # Can't skip accurately without knowing event size.
                # Try to re-sync by scanning for the next valid packed UID.
                # This may misalign but is better than losing all remaining data.
                continue

            # Read serial if applicable
            if event_type.has_serial:
                serial_low = reader.read_u8()
                serial_high = reader.read_u16()
                # serial = serial_low | (serial_high << 16)  # unused currently

            fields = _read_event_fields(reader, event_type)
            capture.total_events += 1
            _dispatch_event(event_type, fields, capture, thread_id, max_memory_events)

        except EOFError:
            break
        except Exception as e:
            capture.parse_errors.append(
                f"Thread {thread_id} parse error at offset {reader.pos}: {e}"
            )
            break


def _dispatch_event(
    event_type: EventType,
    fields: dict,
    capture: UTraceCapture,
    thread_id: int,
    max_memory_events: int,
) -> None:
    """Route a parsed event to its channel handler."""
    key = (event_type.logger_name, event_type.event_name)
    handler = _CHANNEL_HANDLERS.get(key)
    if not handler:
        capture.unknown_event_count += 1
        return

    try:
        handler(
            fields=fields,
            capture=capture,
            thread_id=thread_id,
            event_name=event_type.event_name,
            max_memory_events=max_memory_events,
        )
    except TypeError:
        # Handler doesn't accept all kwargs — call with minimal args
        try:
            handler(fields=fields, capture=capture)
        except Exception:
            pass
    except Exception as e:
        capture.parse_errors.append(
            f"Handler error for {event_type.full_name}: {e}"
        )


# ---------------------------------------------------------------------------
# Top-level parser
# ---------------------------------------------------------------------------

def parse_utrace_file(
    path: str | Path,
    *,
    max_memory_events: int = 500_000,
) -> UTraceCapture:
    """Parse a .utrace file and return structured capture data.

    Args:
        path: Path to the .utrace file.
        max_memory_events: Cap on stored memory events to prevent OOM.

    Returns:
        UTraceCapture with all parsed channel data.

    Raises:
        ValueError: If the file is not a valid .utrace file.
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = path.read_bytes()
    if len(data) < 6:
        raise ValueError(f"File too small to be a valid utrace: {len(data)} bytes")

    capture = UTraceCapture()
    reader = BinaryReader(data)

    # --- Magic ---
    magic = reader.read_u32()
    if magic == MAGIC_TRCE:
        capture.magic = "TRCE"
    elif magic == MAGIC_TRC2:
        capture.magic = "TRC2"
        # TRC2 has metadata section
        metadata_size = reader.read_u16()
        if metadata_size > 0:
            reader.skip(metadata_size)  # Skip metadata fields for now
    elif magic == 0x00000001:
        # Legacy format: protocol 0, transport 1 — no magic header
        capture.magic = "LEGACY"
        reader.pos = 0  # Reset — transport header starts at byte 0
    else:
        raise ValueError(
            f"Invalid utrace magic: 0x{magic:08X}. "
            f"Expected TRCE (0x{MAGIC_TRCE:08X}) or TRC2 (0x{MAGIC_TRC2:08X})"
        )

    # --- Transport header ---
    capture.transport_version = reader.read_u8()
    capture.protocol_version = reader.read_u8()

    if capture.transport_version not in (
        TRANSPORT_RAW, TRANSPORT_PACKET, TRANSPORT_TID_PACKET, TRANSPORT_TID_PACKET_SYNC
    ):
        raise ValueError(f"Unsupported transport version: {capture.transport_version}")

    if capture.protocol_version > 7:
        raise ValueError(f"Unsupported protocol version: {capture.protocol_version}")

    # --- Demux: collect per-thread buffers ---
    # For TidPacket/TidPacketSync, demux packets by thread
    thread_buffers: dict[int, bytearray] = {}

    while reader.remaining() >= 4:
        try:
            packet_size = reader.read_u16()
            thread_id_raw = reader.read_u16()

            thread_id = thread_id_raw & _THREAD_ID_MASK
            is_encoded = bool(thread_id_raw & _ENCODED_MARKER)
            has_verification = bool(thread_id_raw & _VERIFICATION_MARKER)

            header_size = 4  # PacketSize + ThreadId
            decoded_size = 0
            if is_encoded:
                decoded_size = reader.read_u16()
                header_size += 2

            data_size = packet_size - header_size
            if data_size <= 0:
                continue
            if data_size > reader.remaining():
                # Truncated packet at end of file
                capture.parse_errors.append(
                    f"Truncated packet: need {data_size} bytes, have {reader.remaining()}"
                )
                break

            packet_data = reader.read_bytes(data_size)

            # Decompress LZ4-encoded packets
            if is_encoded and decoded_size > 0:
                if _lz4 is None:
                    capture.parse_errors.append(
                        "Encoded packet requires lz4 package: pip install lz4"
                    )
                    continue
                try:
                    packet_data = _lz4.decompress(
                        packet_data,
                        uncompressed_size=decoded_size,
                    )
                except Exception as e:
                    capture.parse_errors.append(f"LZ4 decompress failed: {e}")
                    continue

            # Verification value follows packet data if flag is set
            if has_verification and reader.remaining() >= 8:
                reader.skip(8)

            if thread_id not in thread_buffers:
                thread_buffers[thread_id] = bytearray()
            thread_buffers[thread_id].extend(packet_data)

            capture.total_packets += 1

        except EOFError:
            break
        except Exception as e:
            capture.parse_errors.append(f"Packet parse error: {e}")
            break

    # --- Process thread 0 first (event definitions) ---
    if THREAD_EVENTS in thread_buffers:
        _process_events_thread(
            bytes(thread_buffers[THREAD_EVENTS]),
            capture,
            capture.protocol_version,
        )

    # --- Process thread 1 (importants) ---
    if THREAD_IMPORTANTS in thread_buffers:
        _process_important_events(
            bytes(thread_buffers[THREAD_IMPORTANTS]),
            capture,
            max_memory_events,
            capture.protocol_version,
        )

    # --- Process remaining threads ---
    for tid in sorted(thread_buffers.keys()):
        if tid <= THREAD_IMPORTANTS:
            continue
        _process_thread_events(
            bytes(thread_buffers[tid]),
            tid,
            capture,
            max_memory_events,
        )

    return capture
