"""Tests for the Optick .opt binary parser.

Constructs synthetic .opt binary data matching the C++ serialization format
and verifies the Python parser can correctly extract all fields.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

import pytest

# Add tool dir to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "Tools" / "optick-analyze"))
from optick_parser import (
    CHUNK_EVENT_FRAME,
    CHUNK_FRAME_DESCRIPTION_BOARD,
    CHUNK_FRAMES_PACK,
    CHUNK_NULL_FRAME,
    CHUNK_SUMMARY_PACK,
    OPTICK_MAGIC,
    BinaryReader,
    OptickCapture,
    analyze_capture,
    parse_opt_file,
)

PROTOCOL_VERSION = 26
APP_ID = 0xB50F


# --- Helpers to build binary chunks matching C++ OutputDataStream ---

def _u8(v: int) -> bytes:
    return struct.pack("<B", v)

def _i8(v: int) -> bytes:
    return struct.pack("<b", v)

def _u16(v: int) -> bytes:
    return struct.pack("<H", v)

def _u32(v: int) -> bytes:
    return struct.pack("<I", v)

def _i32(v: int) -> bytes:
    return struct.pack("<i", v)

def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)

def _i64(v: int) -> bytes:
    return struct.pack("<q", v)

def _f32(v: float) -> bytes:
    return struct.pack("<f", v)

def _string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return _u32(len(encoded)) + encoded

def _data_response(chunk_type: int, payload: bytes) -> bytes:
    header = _u32(PROTOCOL_VERSION) + _u32(len(payload)) + _u16(chunk_type) + _u16(APP_ID)
    return header + payload

def _null_frame() -> bytes:
    return _data_response(CHUNK_NULL_FRAME, b"")


def _build_summary_pack(
    board_number: int = 1,
    frame_times: list[float] | None = None,
    summary: dict[str, str] | None = None,
) -> bytes:
    payload = _u32(board_number)

    times = frame_times or []
    payload += _u32(len(times))
    for t in times:
        payload += _f32(t)

    kvs = summary or {}
    payload += _u32(len(kvs))
    for k, v in kvs.items():
        payload += _string(k) + _string(v)

    # No attachments
    payload += _u32(0)

    return _data_response(CHUNK_SUMMARY_PACK, payload)


def _build_event_description(name: str, file: str, line: int, filter_val: int = 0, color: int = 0, flags: int = 0) -> bytes:
    return _string(name) + _string(file) + _u32(line) + _u32(filter_val) + _u32(color) + _f32(0.0) + _u8(flags)


def _build_thread_description(tid: int, pid: int, name: str, max_depth: int = 1, priority: int = 0, mask: int = 0) -> bytes:
    return _u64(tid) + _u32(pid) + _string(name) + _i32(max_depth) + _i32(priority) + _u32(mask)


def _build_frame_description_board(
    board_number: int = 1,
    frequency: int = 10_000_000,
    threads: list[tuple[int, int, str]] | None = None,
    events: list[tuple[str, str, int]] | None = None,
) -> bytes:
    payload = _u32(board_number)
    payload += _u64(frequency)
    payload += _u64(0)  # origin
    payload += _u32(0)  # precision
    payload += _i64(1000) + _i64(2000)  # timeSlice

    thread_list = threads or []
    payload += _u32(len(thread_list))
    for tid, pid, name in thread_list:
        payload += _build_thread_description(tid, pid, name)

    # fibers
    payload += _u32(0)

    # forcedMainThreadIndex
    payload += _u32(0xFFFFFFFF)

    # EventDescriptionBoard (vector of EventDescription)
    event_list = events or []
    payload += _u32(len(event_list))
    for name, file, line in event_list:
        payload += _build_event_description(name, file, line)

    # Remaining board fields — tags, run, filters, threadDescs count, mode,
    # processDescs, threadDescs, processID, hardwareConcurrency
    # We add minimal trailing data so the parser can skip
    payload += _u32(0)  # tags
    payload += _u32(0)  # run
    payload += _u32(0)  # filters
    payload += _u32(0)  # threadDescs count
    payload += _u32(0)  # mode
    payload += _u32(0)  # processDescs count
    payload += _u32(0)  # threadDescs count (again for processDescs/threadDescs vectors)
    payload += _u32(0)  # processID
    payload += _u32(4)  # hardwareConcurrency

    return _data_response(CHUNK_FRAME_DESCRIPTION_BOARD, payload)


def _build_event_frame(
    board_number: int = 1,
    thread_number: int = 0,
    events: list[tuple[int, int, int]] | None = None,
) -> bytes:
    """Build an EventFrame chunk. events = list of (start, finish, desc_index)."""
    payload = _u32(board_number)
    payload += _i32(thread_number)
    payload += _i32(-1)  # fiberNumber
    payload += _i64(1000) + _i64(2000)  # scope event time
    payload += _i32(0)  # frameType

    # categories (empty for test)
    payload += _u32(0)

    event_list = events or []
    payload += _u32(len(event_list))
    for start, finish, desc_idx in event_list:
        payload += _i64(start) + _i64(finish) + _u32(desc_idx)

    return _data_response(CHUNK_EVENT_FRAME, payload)


def _build_opt_file(chunks: list[bytes], compressed: bool = True) -> bytes:
    """Build a complete .opt file from chunks."""
    flags = 0x02 if compressed else 0x00
    header = struct.pack("<IHH", OPTICK_MAGIC, 0, flags)

    body = b"".join(chunks) + _null_frame()

    if compressed:
        compressor = zlib.compressobj(1, zlib.DEFLATED, -15)
        body = compressor.compress(body) + compressor.flush()

    return header + body


# --- Tests ---

class TestBinaryReader:
    def test_primitives(self):
        data = _u32(42) + _i64(-100) + _f32(3.14) + _string("hello")
        r = BinaryReader(data)
        assert r.read_u32() == 42
        assert r.read_i64() == -100
        assert abs(r.read_f32() - 3.14) < 0.01
        assert r.read_string() == "hello"

    def test_empty_string(self):
        r = BinaryReader(_u32(0))
        assert r.read_string() == ""

    def test_eof_raises(self):
        r = BinaryReader(b"\x00\x00")
        with pytest.raises(EOFError):
            r.read_u32()


class TestParseOptFile:
    def test_invalid_magic(self, tmp_path):
        bad_file = tmp_path / "bad.opt"
        bad_file.write_bytes(struct.pack("<IHH", 0xDEADBEEF, 0, 0))
        with pytest.raises(ValueError, match="Invalid Optick magic"):
            parse_opt_file(bad_file)

    def test_empty_capture(self, tmp_path):
        f = tmp_path / "empty.opt"
        f.write_bytes(_build_opt_file([]))
        capture = parse_opt_file(f)
        assert isinstance(capture, OptickCapture)
        assert capture.frame_times_ms == []

    def test_summary_pack(self, tmp_path):
        chunks = [
            _build_summary_pack(
                frame_times=[16.0, 14.5, 18.2],
                summary={"Platform": "Windows", "GPU": "RTX 4090"},
            ),
        ]
        f = tmp_path / "summary.opt"
        f.write_bytes(_build_opt_file(chunks))
        capture = parse_opt_file(f)
        assert len(capture.frame_times_ms) == 3
        assert abs(capture.frame_times_ms[0] - 16.0) < 0.01
        assert capture.summary["Platform"] == "Windows"
        assert capture.summary["GPU"] == "RTX 4090"

    def test_frame_description_board(self, tmp_path):
        chunks = [
            _build_frame_description_board(
                frequency=10_000_000,
                threads=[(1234, 5678, "GameThread"), (2345, 5678, "RenderThread")],
                events=[("Tick", "GameEngine.cpp", 42), ("Render", "Renderer.cpp", 100)],
            ),
        ]
        f = tmp_path / "board.opt"
        f.write_bytes(_build_opt_file(chunks))
        capture = parse_opt_file(f)
        assert capture.cpu_frequency == 10_000_000
        assert len(capture.threads) == 2
        assert capture.threads[0].name == "GameThread"
        assert capture.threads[1].name == "RenderThread"
        assert len(capture.event_descriptions) == 2
        assert capture.event_descriptions[0].name == "Tick"
        assert capture.event_descriptions[1].file == "Renderer.cpp"

    def test_event_frame(self, tmp_path):
        chunks = [
            _build_event_frame(
                thread_number=0,
                events=[(1000, 1500, 0), (1100, 1400, 1)],
            ),
        ]
        f = tmp_path / "events.opt"
        f.write_bytes(_build_opt_file(chunks))
        capture = parse_opt_file(f)
        assert len(capture.scope_blocks) == 1
        block = capture.scope_blocks[0]
        assert block.thread_number == 0
        assert len(block.events) == 2
        assert block.events[0].start == 1000
        assert block.events[0].finish == 1500

    def test_uncompressed(self, tmp_path):
        chunks = [
            _build_summary_pack(frame_times=[10.0, 20.0]),
        ]
        f = tmp_path / "uncompressed.opt"
        f.write_bytes(_build_opt_file(chunks, compressed=False))
        capture = parse_opt_file(f)
        assert len(capture.frame_times_ms) == 2

    def test_full_capture(self, tmp_path):
        """Test a complete capture with all chunk types together."""
        freq = 10_000_000
        chunks = [
            _build_summary_pack(
                frame_times=[15.0, 16.5, 14.2, 22.0, 33.5],
                summary={"Platform": "Windows", "UnrealVersion": "5.7.1"},
            ),
            _build_frame_description_board(
                frequency=freq,
                threads=[(100, 1, "GameThread")],
                events=[("AI::Update", "AI.cpp", 50), ("Physics::Step", "Physics.cpp", 80)],
            ),
            _build_event_frame(
                thread_number=0,
                events=[
                    (0, 150000, 0),       # AI::Update, 15ms at 10MHz
                    (0, 50000, 1),        # Physics::Step, 5ms
                    (150000, 300000, 0),  # AI::Update again, 15ms
                ],
            ),
        ]
        f = tmp_path / "full.opt"
        f.write_bytes(_build_opt_file(chunks))
        capture = parse_opt_file(f)

        assert len(capture.frame_times_ms) == 5
        assert capture.cpu_frequency == freq
        assert len(capture.threads) == 1
        assert len(capture.event_descriptions) == 2
        assert len(capture.scope_blocks) == 1
        assert len(capture.scope_blocks[0].events) == 3


class TestAnalyzeCapture:
    def test_frame_summary(self):
        capture = OptickCapture(frame_times_ms=[10.0, 15.0, 20.0, 25.0, 35.0])
        analysis = analyze_capture(capture)
        fs = analysis["frame_summary"]
        assert fs["total_frames"] == 5
        assert fs["avg_ms"] == 21.0
        assert fs["min_ms"] == 10.0
        assert fs["max_ms"] == 35.0
        assert fs["frames_above_16ms"] == 3
        assert fs["frames_above_33ms"] == 1

    def test_hottest_scopes(self):
        from optick_parser import EventDescription, ScopeBlock, ScopeEvent

        capture = OptickCapture(
            cpu_frequency=10_000_000,
            event_descriptions=[
                EventDescription(index=0, name="Tick", file="a.cpp", line=1, filter=0, color=0, flags=0),
                EventDescription(index=1, name="Render", file="b.cpp", line=2, filter=0, color=0, flags=0),
            ],
            scope_blocks=[
                ScopeBlock(
                    board_number=1, thread_number=0, fiber_number=-1,
                    event_start=0, event_finish=500000, frame_type=0,
                    categories=[],
                    events=[
                        ScopeEvent(start=0, finish=300000, description_index=0),    # 30ms
                        ScopeEvent(start=0, finish=100000, description_index=1),    # 10ms
                        ScopeEvent(start=300000, finish=500000, description_index=0),  # 20ms
                    ],
                ),
            ],
        )
        analysis = analyze_capture(capture, top_n=10)
        hottest = analysis["hottest_scopes"]
        assert len(hottest) == 2
        # Tick should be first (50ms total)
        assert hottest[0]["name"] == "Tick"
        assert hottest[0]["total_ms"] == 50.0
        assert hottest[0]["calls"] == 2
        assert hottest[1]["name"] == "Render"
        assert hottest[1]["total_ms"] == 10.0
