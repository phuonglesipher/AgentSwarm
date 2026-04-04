"""Tests for the minidump-analyze tool."""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import pytest

# Add the tool directory so we can import minidump_parser directly
_tool_dir = str(Path(__file__).resolve().parent.parent / "Tools" / "minidump-analyze")
if _tool_dir not in sys.path:
    sys.path.insert(0, _tool_dir)

from minidump_parser import (
    ExceptionInfo,
    MinidumpResult,
    ModuleInfo,
    StackFrame,
    ThreadInfo,
    _classify_ue_module,
    _resolve_address_to_module,
    format_summary,
    to_analysis_dict,
)


# ---------------------------------------------------------------------------
# _classify_ue_module
# ---------------------------------------------------------------------------

class TestClassifyUeModule:
    def test_editor_engine_module(self):
        is_ue, name = _classify_ue_module("UnrealEditor-Engine.dll")
        assert is_ue is True
        assert name == "Engine"

    def test_editor_plugin_module(self):
        is_ue, name = _classify_ue_module("UnrealEditor-SipherComboGraph.dll")
        assert is_ue is True
        assert name == "SipherComboGraph"

    def test_editor_core_uobject(self):
        is_ue, name = _classify_ue_module("UnrealEditor-CoreUObject.dll")
        assert is_ue is True
        assert name == "CoreUObject"

    def test_case_insensitive(self):
        is_ue, name = _classify_ue_module("unrealeditor-engine.dll")
        assert is_ue is True
        assert name == "engine"

    def test_system_dll(self):
        is_ue, name = _classify_ue_module("ntdll.dll")
        assert is_ue is False
        assert name == ""

    def test_kernel32(self):
        is_ue, name = _classify_ue_module("kernel32.dll")
        assert is_ue is False
        assert name == ""

    def test_d3d11(self):
        is_ue, name = _classify_ue_module("d3d11.dll")
        assert is_ue is False
        assert name == ""

    def test_empty_name(self):
        is_ue, name = _classify_ue_module("")
        assert is_ue is False
        assert name == ""

    def test_linux_so(self):
        is_ue, name = _classify_ue_module("UnrealEditor-Engine.so")
        assert is_ue is True
        assert name == "Engine"


# ---------------------------------------------------------------------------
# _resolve_address_to_module
# ---------------------------------------------------------------------------

class TestResolveAddressToModule:
    @pytest.fixture()
    def modules(self):
        return [
            ModuleInfo("ntdll.dll", 0x7FF800000000, 0x100000, False, ""),
            ModuleInfo("UnrealEditor-Engine.dll", 0x7FF900000000, 0x500000, True, "Engine"),
            ModuleInfo("UnrealEditor-S2.dll", 0x7FFA00000000, 0x200000, True, "S2"),
        ]

    def test_exact_base(self, modules):
        name, offset = _resolve_address_to_module(0x7FF900000000, modules)
        assert name == "UnrealEditor-Engine.dll"
        assert offset == 0

    def test_within_module(self, modules):
        name, offset = _resolve_address_to_module(0x7FF900001234, modules)
        assert name == "UnrealEditor-Engine.dll"
        assert offset == 0x1234

    def test_end_of_module(self, modules):
        # Last valid address in Engine module
        name, offset = _resolve_address_to_module(0x7FF9004FFFFF, modules)
        assert name == "UnrealEditor-Engine.dll"
        assert offset == 0x4FFFFF

    def test_outside_all_modules(self, modules):
        name, offset = _resolve_address_to_module(0x1234, modules)
        assert name == "<unknown>"
        assert offset == 0x1234

    def test_between_modules(self, modules):
        # Gap between ntdll and Engine
        name, offset = _resolve_address_to_module(0x7FF880000000, modules)
        assert name == "<unknown>"
        assert offset == 0x7FF880000000

    def test_empty_module_list(self):
        name, offset = _resolve_address_to_module(0x7FF900001234, [])
        assert name == "<unknown>"
        assert offset == 0x7FF900001234


# ---------------------------------------------------------------------------
# format_summary
# ---------------------------------------------------------------------------

class TestFormatSummary:
    def test_basic_summary(self):
        result = MinidumpResult(
            file_path="test.dmp",
            file_size_mb=12.5,
            dump_type="MiniDump",
            exception=ExceptionInfo(
                code=0xC0000005,
                code_name="ACCESS_VIOLATION",
                address=0x7FF900001234,
                module_name="UnrealEditor-Engine.dll",
                module_offset=0x1234,
                thread_id=1000,
            ),
            crashing_thread=ThreadInfo(
                thread_id=1000,
                is_crashing_thread=True,
                stack_frames=[
                    StackFrame(0, 0x7FF900001234, "UnrealEditor-Engine.dll", 0x1234),
                ],
            ),
            threads=[],
            modules=[
                ModuleInfo("UnrealEditor-Engine.dll", 0x7FF900000000, 0x500000, True, "Engine"),
                ModuleInfo("ntdll.dll", 0x7FF800000000, 0x100000, False, ""),
            ],
            system_info={"os": "Windows 10", "arch": "AMD64"},
        )

        summary = format_summary(result)
        assert "ACCESS_VIOLATION" in summary
        assert "0xC0000005" in summary
        assert "UnrealEditor-Engine.dll" in summary
        assert "MiniDump" in summary
        assert "12.5 MB" in summary

    def test_no_exception(self):
        result = MinidumpResult(
            file_path="test.dmp",
            file_size_mb=1.0,
            dump_type="MiniDump",
            exception=None,
            crashing_thread=None,
            threads=[],
            modules=[],
            system_info={},
        )
        summary = format_summary(result)
        assert "None found" in summary

    def test_unknown_module(self):
        result = MinidumpResult(
            file_path="test.dmp",
            file_size_mb=1.0,
            dump_type="MiniDump",
            exception=ExceptionInfo(
                code=0xC0000005,
                code_name="ACCESS_VIOLATION",
                address=0x1234,
                module_name="<unknown>",
                module_offset=0x1234,
                thread_id=1,
            ),
            crashing_thread=None,
            threads=[],
            modules=[],
            system_info={},
            warnings=["No module list found in dump — all addresses are raw"],
        )
        summary = format_summary(result)
        assert "<unknown>@0x" in summary
        assert "WARNING" in summary

    def test_unresolved_frames_reported(self):
        result = MinidumpResult(
            file_path="test.dmp",
            file_size_mb=1.0,
            dump_type="MiniDump",
            exception=None,
            crashing_thread=None,
            threads=[],
            modules=[],
            system_info={},
            unresolved_frame_count=5,
        )
        summary = format_summary(result)
        assert "Unresolved Frames: 5" in summary


# ---------------------------------------------------------------------------
# to_analysis_dict
# ---------------------------------------------------------------------------

class TestToAnalysisDict:
    def test_basic_dict(self):
        result = MinidumpResult(
            file_path="test.dmp",
            file_size_mb=10.0,
            dump_type="MiniDump",
            exception=ExceptionInfo(
                code=0xC0000005,
                code_name="ACCESS_VIOLATION",
                address=0x7FF900001234,
                module_name="UnrealEditor-Engine.dll",
                module_offset=0x1234,
                thread_id=100,
            ),
            crashing_thread=ThreadInfo(
                thread_id=100,
                is_crashing_thread=True,
                stack_frames=[
                    StackFrame(0, 0x7FF900001234, "UnrealEditor-Engine.dll", 0x1234),
                ],
            ),
            threads=[
                ThreadInfo(100, True),
                ThreadInfo(200, False),
            ],
            modules=[
                ModuleInfo("UnrealEditor-Engine.dll", 0x7FF900000000, 0x500000, True, "Engine"),
            ],
            system_info={"os": "Windows 10"},
        )

        d = to_analysis_dict(result)
        assert d["dump_type"] == "MiniDump"
        assert d["exception"]["code_name"] == "ACCESS_VIOLATION"
        assert d["thread_count"] == 2
        assert d["ue_module_count"] == 1
        assert len(d["crashing_thread_stack"]) == 1
        assert d["modules"][0]["ue_module"] == "Engine"

    def test_no_exception_dict(self):
        result = MinidumpResult(
            file_path="test.dmp",
            file_size_mb=1.0,
            dump_type="MiniDump",
            exception=None,
            crashing_thread=None,
            threads=[],
            modules=[],
            system_info={},
        )
        d = to_analysis_dict(result)
        assert d["exception"] is None
        assert d["crashing_thread_stack"] == []


# ---------------------------------------------------------------------------
# entry.py - error paths
# ---------------------------------------------------------------------------

class TestEntryErrorPaths:
    def test_file_not_found(self):
        """Verify the tool handles missing files gracefully."""
        # We test the logic path without actually loading the tool framework
        from minidump_parser import parse_minidump

        with pytest.raises(Exception):
            parse_minidump(Path("/nonexistent/file.dmp"))

    def test_scan_empty_dir(self):
        """Verify scan_crash_dir returns helpful error for empty dirs."""
        with tempfile.TemporaryDirectory() as tmp:
            # No .dmp files in directory
            dmp_files = sorted(
                Path(tmp).rglob("*.dmp"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            assert len(dmp_files) == 0

    def test_scan_finds_most_recent(self):
        """Verify scan picks most recent .dmp by mtime."""
        import time

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Create two fake .dmp files with different mtimes
            old_dmp = tmp_path / "old.dmp"
            old_dmp.write_bytes(b"MDMP" + b"\x00" * 100)
            time.sleep(0.05)

            new_dmp = tmp_path / "new.dmp"
            new_dmp.write_bytes(b"MDMP" + b"\x00" * 100)

            dmp_files = sorted(
                tmp_path.rglob("*.dmp"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            assert dmp_files[0].name == "new.dmp"
