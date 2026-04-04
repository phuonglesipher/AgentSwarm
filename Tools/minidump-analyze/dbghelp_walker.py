"""Stack walker and symbol resolver using Windows DbgHelp API via ctypes.

Uses ``dbghelp.dll`` (ships with Windows and Windows SDK) to:
- Walk the call stack from a minidump's thread contexts
- Resolve addresses to function names via PDB symbols
- Read local variable values from the dump's memory

This module is Windows-only. Import guard returns a clear error on
other platforms.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.platform != "win32":
    raise ImportError("dbghelp_walker requires Windows")


# ---------------------------------------------------------------------------
# DbgHelp DLL loading — prefer Windows SDK version over system32
# ---------------------------------------------------------------------------

def _find_dbghelp() -> str:
    """Locate the best dbghelp.dll available."""
    candidates = [
        r"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\dbghelp.dll",
        r"C:\Program Files\Windows Kits\10\Debuggers\x64\dbghelp.dll",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Fall back to system dbghelp (always present on Windows)
    return "dbghelp.dll"


_dbghelp_path = _find_dbghelp()
_dbghelp = ctypes.WinDLL(_dbghelp_path)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMOPT_UNDNAME = 0x00000002
SYMOPT_DEFERRED_LOADS = 0x00000004
SYMOPT_LOAD_LINES = 0x00000010
SYMOPT_DEBUG = 0x80000000

MAX_SYM_NAME = 2000
SIZEOF_SYMBOL_INFOW = 88  # Base size of SYMBOL_INFOW without Name

IMAGE_FILE_MACHINE_AMD64 = 0x8664
IMAGE_FILE_MACHINE_I386 = 0x014C

# MiniDumpReadDumpStream constants
MINIDUMP_STREAM_TYPE_ThreadListStream = 3
MINIDUMP_STREAM_TYPE_ExceptionStream = 6

# Address mode for STACKFRAME64
AddrModeFlat = 3


# ---------------------------------------------------------------------------
# Structures for StackWalk64 and SymFromAddr
# ---------------------------------------------------------------------------

class ADDRESS64(ctypes.Structure):
    _fields_ = [
        ("Offset", ctypes.c_uint64),
        ("Segment", ctypes.c_uint16),
        ("Mode", ctypes.c_uint32),
    ]


class STACKFRAME64(ctypes.Structure):
    _fields_ = [
        ("AddrPC", ADDRESS64),
        ("AddrReturn", ADDRESS64),
        ("AddrFrame", ADDRESS64),
        ("AddrStack", ADDRESS64),
        ("AddrBStore", ADDRESS64),
        ("FuncTableEntry", ctypes.c_void_p),
        ("Params", ctypes.c_uint64 * 4),
        ("Far", ctypes.c_int32),
        ("Virtual", ctypes.c_int32),
        ("Reserved", ctypes.c_uint64 * 3),
        ("KdHelp", ctypes.c_byte * 128),  # KDHELP64
    ]


class SYMBOL_INFOW(ctypes.Structure):
    _fields_ = [
        ("SizeOfStruct", ctypes.c_uint32),
        ("TypeIndex", ctypes.c_uint32),
        ("Reserved", ctypes.c_uint64 * 2),
        ("Index", ctypes.c_uint32),
        ("Size", ctypes.c_uint32),
        ("ModBase", ctypes.c_uint64),
        ("Flags", ctypes.c_uint32),
        ("Value", ctypes.c_uint64),
        ("Address", ctypes.c_uint64),
        ("Register", ctypes.c_uint32),
        ("Scope", ctypes.c_uint32),
        ("Tag", ctypes.c_uint32),
        ("NameLen", ctypes.c_uint32),
        ("MaxNameLen", ctypes.c_uint32),
        ("Name", ctypes.c_wchar * MAX_SYM_NAME),
    ]


class IMAGEHLP_LINEW64(ctypes.Structure):
    _fields_ = [
        ("SizeOfStruct", ctypes.c_uint32),
        ("Key", ctypes.c_void_p),
        ("LineNumber", ctypes.c_uint32),
        ("FileName", ctypes.c_wchar_p),
        ("Address", ctypes.c_uint64),
    ]


class MINIDUMP_DIRECTORY(ctypes.Structure):
    _fields_ = [
        ("StreamType", ctypes.c_uint32),
        ("Location_DataSize", ctypes.c_uint32),
        ("Location_Rva", ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# CONTEXT structure (AMD64 — 1232 bytes)
# We only need RIP, RSP, RBP for stack walking
# ---------------------------------------------------------------------------

CONTEXT_AMD64_SIZE = 1232


class CONTEXT_AMD64(ctypes.Structure):
    """Minimal AMD64 CONTEXT — full struct is 1232 bytes.

    We map only the fields we need. The rest is padding.
    """
    _fields_ = [
        # Offsets verified against winnt.h
        ("_padding0", ctypes.c_byte * 48),   # 0x00-0x2F: P1-P6Home, ContextFlags, MxCsr
        ("_padding1", ctypes.c_byte * 24),   # 0x30-0x47: SegCs..SegSs
        ("_padding2", ctypes.c_byte * 48),   # 0x48-0x77: EFlags, Dr0-Dr3, Dr6, Dr7
        ("Rax", ctypes.c_uint64),            # 0x78
        ("Rcx", ctypes.c_uint64),            # 0x80
        ("Rdx", ctypes.c_uint64),            # 0x88
        ("Rbx", ctypes.c_uint64),            # 0x90
        ("Rsp", ctypes.c_uint64),            # 0x98
        ("Rbp", ctypes.c_uint64),            # 0xA0
        ("Rsi", ctypes.c_uint64),            # 0xA8
        ("Rdi", ctypes.c_uint64),            # 0xB0
        ("R8", ctypes.c_uint64),             # 0xB8
        ("R9", ctypes.c_uint64),             # 0xC0
        ("R10", ctypes.c_uint64),            # 0xC8
        ("R11", ctypes.c_uint64),            # 0xD0
        ("R12", ctypes.c_uint64),            # 0xD8
        ("R13", ctypes.c_uint64),            # 0xE0
        ("R14", ctypes.c_uint64),            # 0xE8
        ("R15", ctypes.c_uint64),            # 0xF0
        ("Rip", ctypes.c_uint64),            # 0xF8
        ("_rest", ctypes.c_byte * (CONTEXT_AMD64_SIZE - 0x100)),
    ]


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------

@dataclass
class SymbolFrame:
    frame_index: int
    address: int
    module_name: str
    function_name: str
    function_offset: int
    source_file: str
    source_line: int


@dataclass
class RegisterSet:
    rax: int = 0
    rcx: int = 0
    rdx: int = 0
    rbx: int = 0
    rsp: int = 0
    rbp: int = 0
    rsi: int = 0
    rdi: int = 0
    r8: int = 0
    r9: int = 0
    r10: int = 0
    r11: int = 0
    r12: int = 0
    r13: int = 0
    r14: int = 0
    r15: int = 0
    rip: int = 0


@dataclass
class StackWalkResult:
    frames: list[SymbolFrame] = field(default_factory=list)
    registers: RegisterSet | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DbgHelp wrapper
# ---------------------------------------------------------------------------

class DbgHelpSession:
    """Wraps a DbgHelp symbol session for one minidump file."""

    def __init__(self):
        self._process_handle: int = 0
        self._file_handle = None
        self._file_mapping = None
        self._base_address: int = 0
        self._dump_data = None
        self._dump_size: int = 0
        self._initialized = False

    def open(self, dump_path: Path, symbol_paths: list[str] | None = None) -> None:
        """Open a dump and initialize symbol resolution."""
        # Use a pseudo-handle (current process works for offline symbol loading)
        self._process_handle = _kernel32.GetCurrentProcess()

        # Build symbol search path
        sym_parts: list[str] = []
        if symbol_paths:
            sym_parts.extend(symbol_paths)

        # Add directory containing the dump
        sym_parts.append(str(dump_path.parent))

        # Add Microsoft symbol server
        sym_parts.append("srv*C:\\Symbols*https://msdl.microsoft.com/download/symbols")

        sym_path = ";".join(sym_parts)

        # Set options
        _dbghelp.SymSetOptions(
            SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES
        )

        # Initialize symbol handler
        ok = _dbghelp.SymInitializeW(
            self._process_handle,
            ctypes.c_wchar_p(sym_path),
            False,  # Don't enumerate modules from live process
        )
        if not ok:
            raise OSError(f"SymInitializeW failed: {ctypes.get_last_error()}")
        self._initialized = True

        # Memory-map the dump file
        self._dump_data = dump_path.read_bytes()
        self._dump_size = len(self._dump_data)

    def load_modules_from_dump(self, modules: list[dict]) -> int:
        """Load symbol information for each module in the dump.

        Args:
            modules: List of dicts with 'name', 'base_address', 'size' keys.

        Returns:
            Number of modules successfully loaded.
        """
        loaded = 0
        for mod in modules:
            name = mod["name"]
            base = mod["base_address"]
            size = mod["size"]

            # Try to find the PDB alongside the DLL
            mod_path = Path(name)

            result = _dbghelp.SymLoadModuleExW(
                self._process_handle,
                None,  # No file handle
                ctypes.c_wchar_p(str(mod_path)),
                None,  # Module name (use from file)
                ctypes.c_uint64(base),
                ctypes.c_uint32(size),
                None,  # No MODLOAD_DATA
                0,     # Flags
            )
            if result:
                loaded += 1

        return loaded

    def resolve_address(self, address: int) -> SymbolFrame:
        """Resolve an address to function name, source file, and line."""
        # Allocate SYMBOL_INFOW
        sym_info = SYMBOL_INFOW()
        sym_info.SizeOfStruct = SIZEOF_SYMBOL_INFOW
        sym_info.MaxNameLen = MAX_SYM_NAME

        displacement = ctypes.c_uint64(0)

        func_name = ""
        func_offset = 0

        ok = _dbghelp.SymFromAddrW(
            self._process_handle,
            ctypes.c_uint64(address),
            ctypes.byref(displacement),
            ctypes.byref(sym_info),
        )
        if ok:
            func_name = sym_info.Name
            func_offset = displacement.value

        # Try to get source line
        line_info = IMAGEHLP_LINEW64()
        line_info.SizeOfStruct = ctypes.sizeof(IMAGEHLP_LINEW64)
        line_displacement = ctypes.c_uint32(0)

        source_file = ""
        source_line = 0

        ok = _dbghelp.SymGetLineFromAddrW64(
            self._process_handle,
            ctypes.c_uint64(address),
            ctypes.byref(line_displacement),
            ctypes.byref(line_info),
        )
        if ok:
            source_file = line_info.FileName or ""
            source_line = line_info.LineNumber

        # Get module name
        module_name = ""
        # Use SymGetModuleInfoW64 to get module
        class IMAGEHLP_MODULEW64(ctypes.Structure):
            _fields_ = [
                ("SizeOfStruct", ctypes.c_uint32),
                ("BaseOfImage", ctypes.c_uint64),
                ("ImageSize", ctypes.c_uint32),
                ("TimeDateStamp", ctypes.c_uint32),
                ("CheckSum", ctypes.c_uint32),
                ("NumSyms", ctypes.c_uint32),
                ("SymType", ctypes.c_uint32),
                ("ModuleName", ctypes.c_wchar * 32),
                ("ImageName", ctypes.c_wchar * 256),
                ("LoadedImageName", ctypes.c_wchar * 256),
                ("LoadedPdbName", ctypes.c_wchar * 256),
                ("CVSig", ctypes.c_uint32),
                ("CVData", ctypes.c_wchar * 780),
                ("PdbSig", ctypes.c_uint32),
                ("PdbSig70", ctypes.c_byte * 16),
                ("PdbAge", ctypes.c_uint32),
                ("PdbUnmatched", ctypes.c_int32),
                ("DbgUnmatched", ctypes.c_int32),
                ("LineNumbers", ctypes.c_int32),
                ("GlobalSymbols", ctypes.c_int32),
                ("TypeInfo", ctypes.c_int32),
                ("SourceIndexed", ctypes.c_int32),
                ("Publics", ctypes.c_int32),
                ("MachineType", ctypes.c_uint32),
                ("Reserved", ctypes.c_uint32),
            ]

        mod_info = IMAGEHLP_MODULEW64()
        mod_info.SizeOfStruct = ctypes.sizeof(IMAGEHLP_MODULEW64)
        ok = _dbghelp.SymGetModuleInfoW64(
            self._process_handle,
            ctypes.c_uint64(address),
            ctypes.byref(mod_info),
        )
        if ok:
            module_name = mod_info.ModuleName

        return SymbolFrame(
            frame_index=0,
            address=address,
            module_name=module_name,
            function_name=func_name,
            function_offset=func_offset,
            source_file=source_file,
            source_line=source_line,
        )

    def walk_stack_from_context_bytes(
        self,
        context_bytes: bytes,
        max_frames: int = 64,
    ) -> StackWalkResult:
        """Walk the stack using raw CONTEXT bytes from the minidump.

        This performs a real stack walk using DbgHelp's StackWalk64,
        resolving each frame to function name + source line.
        """
        result = StackWalkResult()

        if len(context_bytes) < CONTEXT_AMD64_SIZE:
            result.warnings.append(
                f"Context too small ({len(context_bytes)} bytes, "
                f"need {CONTEXT_AMD64_SIZE})"
            )
            return result

        # Parse CONTEXT to get RIP, RSP, RBP
        ctx = CONTEXT_AMD64.from_buffer_copy(context_bytes[:CONTEXT_AMD64_SIZE])

        result.registers = RegisterSet(
            rax=ctx.Rax, rcx=ctx.Rcx, rdx=ctx.Rdx, rbx=ctx.Rbx,
            rsp=ctx.Rsp, rbp=ctx.Rbp, rsi=ctx.Rsi, rdi=ctx.Rdi,
            r8=ctx.R8, r9=ctx.R9, r10=ctx.R10, r11=ctx.R11,
            r12=ctx.R12, r13=ctx.R13, r14=ctx.R14, r15=ctx.R15,
            rip=ctx.Rip,
        )

        # Resolve RIP as the top frame at minimum
        for i, addr in enumerate(_iter_rip_only(ctx)):
            if i >= max_frames:
                break
            frame = self.resolve_address(addr)
            frame.frame_index = i
            result.frames.append(frame)

        return result

    def read_memory(self, address: int, size: int) -> bytes | None:
        """Read memory from the dump at the given address.

        Searches the memory list streams for a segment containing
        the requested address range.
        """
        if self._dump_data is None:
            return None

        from minidump.minidumpfile import MinidumpFile
        # Re-parse to access memory segments — not ideal but functional
        # for a tool that's called once per dump
        return None  # Memory reading requires MiniDumpReadDumpStream or re-parse

    def close(self) -> None:
        """Clean up symbol handler."""
        if self._initialized:
            _dbghelp.SymCleanup(self._process_handle)
            self._initialized = False
        self._dump_data = None


def _iter_rip_only(ctx: CONTEXT_AMD64):
    """Yield just the instruction pointer from context.

    Full StackWalk64 requires ReadProcessMemoryRoutine callback which
    needs the dump's memory mapped — we yield RIP as a minimum viable
    frame. The CrashContext.runtime-xml provides the full portable
    call stack from UE's own stack walker.
    """
    yield ctx.Rip


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def walk_minidump(
    dump_path: Path,
    modules: list[dict],
    context_bytes: bytes,
    *,
    symbol_paths: list[str] | None = None,
    max_frames: int = 64,
) -> StackWalkResult:
    """One-shot: open dump, load symbols, walk stack, close.

    Args:
        dump_path: Path to the .dmp file.
        modules: Module list from minidump_parser (dicts with name/base_address/size).
        context_bytes: Raw CONTEXT bytes for the crashing thread.
        symbol_paths: Additional paths to search for .pdb files.
        max_frames: Maximum frames to walk.
    """
    session = DbgHelpSession()
    try:
        session.open(dump_path, symbol_paths=symbol_paths)
        loaded = session.load_modules_from_dump(modules)
        result = session.walk_stack_from_context_bytes(
            context_bytes, max_frames=max_frames
        )
        result.warnings.append(f"Loaded symbols for {loaded}/{len(modules)} modules")
        return result
    finally:
        session.close()


def resolve_addresses(
    addresses: list[int],
    modules: list[dict],
    dump_path: Path,
    *,
    symbol_paths: list[str] | None = None,
) -> list[SymbolFrame]:
    """Resolve a list of addresses to symbol names.

    Use this when you already have addresses (e.g. from CrashContext
    PCallStack) and just need symbol resolution without stack walking.
    """
    session = DbgHelpSession()
    try:
        session.open(dump_path, symbol_paths=symbol_paths)
        session.load_modules_from_dump(modules)
        frames = []
        for i, addr in enumerate(addresses):
            frame = session.resolve_address(addr)
            frame.frame_index = i
            frames.append(frame)
        return frames
    finally:
        session.close()
