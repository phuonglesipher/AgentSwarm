"""Minidump (.dmp) parser for UE crash investigation.

Wraps the ``minidump`` PyPI package and adds UE-specific module
classification, address resolution, and structured output formatting.

When ``dbghelp.dll`` is available (Windows), uses it to resolve
addresses to function names + source lines via PDB symbols.
Also parses ``CrashContext.runtime-xml`` for UE's own portable
call stack when present alongside the .dmp file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Windows SEH exception codes (superset of what minidump.streams.ExceptionCode
# covers, plus UE-specific codes)
# ---------------------------------------------------------------------------

EXCEPTION_NAMES: dict[int, str] = {
    0x80000003: "BREAKPOINT",
    0x80000004: "SINGLE_STEP",
    0xC0000005: "ACCESS_VIOLATION",
    0xC0000006: "IN_PAGE_ERROR",
    0xC0000008: "INVALID_HANDLE",
    0xC000001D: "ILLEGAL_INSTRUCTION",
    0xC0000025: "NONCONTINUABLE_EXCEPTION",
    0xC0000026: "INVALID_DISPOSITION",
    0xC000008C: "ARRAY_BOUNDS_EXCEEDED",
    0xC000008D: "FLOAT_DENORMAL_OPERAND",
    0xC000008E: "FLOAT_DIVIDE_BY_ZERO",
    0xC0000090: "FLOAT_INVALID_OPERATION",
    0xC0000091: "FLOAT_OVERFLOW",
    0xC0000092: "FLOAT_STACK_CHECK",
    0xC0000093: "FLOAT_UNDERFLOW",
    0xC0000094: "INTEGER_DIVIDE_BY_ZERO",
    0xC0000095: "INTEGER_OVERFLOW",
    0xC0000096: "PRIVILEGED_INSTRUCTION",
    0xC00000FD: "STACK_OVERFLOW",
    0xC00000C5: "STACK_BUFFER_OVERRUN",
    0xC0000135: "DLL_NOT_FOUND",
    0xC0000142: "DLL_INIT_FAILED",
    0xC0000409: "FAST_FAIL",
    0xE06D7363: "CPP_EXCEPTION",
    # UE-specific
    0x00000001: "UE_FATAL_ERROR",
}

# Patterns for UE module DLL names
_UE_MODULE_RE = re.compile(
    r"^UnrealEditor-(.+)\.dll$", re.IGNORECASE
)
_UE_MODULE_PLATFORM_RE = re.compile(
    r"^UnrealEditor-(.+)\.so$", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExceptionInfo:
    code: int
    code_name: str
    address: int
    module_name: str
    module_offset: int
    thread_id: int = 0


@dataclass
class StackFrame:
    frame_index: int
    address: int
    module_name: str
    module_offset: int
    symbol_name: str = ""
    source_file: str = ""
    source_line: int = 0


@dataclass
class ThreadInfo:
    thread_id: int
    is_crashing_thread: bool
    stack_frames: list[StackFrame] = field(default_factory=list)


@dataclass
class ModuleInfo:
    name: str
    base_address: int
    size: int
    is_ue_module: bool
    ue_module_name: str


@dataclass
class MinidumpResult:
    file_path: str
    file_size_mb: float
    dump_type: str
    exception: ExceptionInfo | None
    crashing_thread: ThreadInfo | None
    threads: list[ThreadInfo]
    modules: list[ModuleInfo]
    system_info: dict[str, str]
    unresolved_frame_count: int = 0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module classification
# ---------------------------------------------------------------------------

def _classify_ue_module(dll_name: str) -> tuple[bool, str]:
    """Determine if a DLL is a UE editor module and extract the module name.

    Handles both bare filenames (``UnrealEditor-Engine.dll``) and full
    paths (``D:\\UnrealEngine\\...\\UnrealEditor-Engine.dll``).

    Returns (is_ue_module, ue_module_name).
    """
    # Extract basename — module names from minidump are often full paths
    basename = dll_name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    m = _UE_MODULE_RE.match(basename)
    if not m:
        m = _UE_MODULE_PLATFORM_RE.match(basename)
    if m:
        return True, m.group(1)
    return False, ""


def _resolve_address_to_module(
    address: int,
    modules: list[ModuleInfo],
) -> tuple[str, int]:
    """Map an instruction address to a module name and offset.

    Returns ("<unknown>", raw_address) when no module contains the address.
    """
    for mod in modules:
        if mod.base_address <= address < mod.base_address + mod.size:
            return mod.name, address - mod.base_address
    return "<unknown>", address


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_minidump(
    path: Path,
    *,
    max_stack_frames: int = 64,
    include_all_threads: bool = False,
    symbol_paths: list[str] | None = None,
) -> MinidumpResult:
    """Parse a .dmp file and return structured crash data.

    Attempts three layers of stack trace enrichment:
    1. CrashContext.runtime-xml (UE's own portable call stack)
    2. DbgHelp symbol resolution (PDB-based, Windows only)
    3. Raw address + module offset (always available)
    """

    from minidump.minidumpfile import MinidumpFile

    file_size = path.stat().st_size
    file_size_mb = round(file_size / (1024 * 1024), 2)
    dump_type = "FullDump" if file_size > 100 * 1024 * 1024 else "MiniDump"

    warnings: list[str] = []
    if file_size > 500 * 1024 * 1024:
        warnings.append(
            f"Large dump file ({file_size_mb} MB). "
            "Parsing may be slow; only metadata streams are read."
        )

    mdf = MinidumpFile.parse(str(path))

    # --- Modules ---
    modules: list[ModuleInfo] = []
    if mdf.modules is not None:
        for mod in mdf.modules.modules:
            is_ue, ue_name = _classify_ue_module(mod.name or "")
            modules.append(ModuleInfo(
                name=mod.name or "<unnamed>",
                base_address=mod.baseaddress or 0,
                size=mod.size or 0,
                is_ue_module=is_ue,
                ue_module_name=ue_name,
            ))
    else:
        warnings.append(
            "No module list found in dump — all addresses are raw."
        )

    # Sort by base address for resolution
    modules.sort(key=lambda m: m.base_address)

    # --- Exception ---
    exception: ExceptionInfo | None = None
    crashing_thread_id: int | None = None

    if mdf.exception is not None and mdf.exception.exception_records:
        rec = mdf.exception.exception_records[0]
        exc = rec.ExceptionRecord

        raw_code = exc.ExceptionCode_raw if exc.ExceptionCode_raw is not None else 0
        code_name = EXCEPTION_NAMES.get(
            raw_code,
            exc.ExceptionCode.name if exc.ExceptionCode else f"UNKNOWN_0x{raw_code:08X}",
        )
        exc_address = exc.ExceptionAddress or 0
        mod_name, mod_offset = _resolve_address_to_module(exc_address, modules)
        crashing_thread_id = rec.ThreadId

        exception = ExceptionInfo(
            code=raw_code,
            code_name=code_name,
            address=exc_address,
            module_name=mod_name,
            module_offset=mod_offset,
            thread_id=crashing_thread_id or 0,
        )

    # --- Parse CrashContext for UE's portable call stack ---
    crash_ctx = _parse_crash_context(path)
    ctx_frames = crash_ctx.get("callstack", [])

    # --- Build crashing thread stack from CrashContext (best source) ---
    threads: list[ThreadInfo] = []
    crashing_thread: ThreadInfo | None = None
    unresolved_count = 0

    if ctx_frames:
        # Use UE's own stack trace — it has all frames
        frames: list[StackFrame] = []
        for cf in ctx_frames[:max_stack_frames]:
            addr = cf.get("address", 0)
            mod_name_full, mod_off = _resolve_address_to_module(addr, modules)
            _, ue_name = _classify_ue_module(cf.get("module", ""))
            display_mod = cf.get("module", mod_name_full)
            frames.append(StackFrame(
                frame_index=cf["index"],
                address=addr,
                module_name=display_mod,
                module_offset=cf.get("offset", mod_off),
            ))
            if display_mod == "<unknown>":
                unresolved_count += 1

        crashing_thread = ThreadInfo(
            thread_id=crashing_thread_id or 0,
            is_crashing_thread=True,
            stack_frames=frames,
        )
    else:
        # Fallback: only the crash address from exception record
        if mdf.threads is not None:
            for t in mdf.threads.threads:
                is_crashing = (
                    crashing_thread_id is not None
                    and t.ThreadId == crashing_thread_id
                )

                if not include_all_threads and not is_crashing:
                    threads.append(ThreadInfo(
                        thread_id=t.ThreadId or 0,
                        is_crashing_thread=False,
                    ))
                    continue

                frames = []
                if is_crashing and exception is not None:
                    frames.append(StackFrame(
                        frame_index=0,
                        address=exception.address,
                        module_name=exception.module_name,
                        module_offset=exception.module_offset,
                    ))

                ti = ThreadInfo(
                    thread_id=t.ThreadId or 0,
                    is_crashing_thread=is_crashing,
                    stack_frames=frames[:max_stack_frames],
                )
                threads.append(ti)
                if is_crashing:
                    crashing_thread = ti

                for f in frames:
                    if f.module_name == "<unknown>":
                        unresolved_count += 1

    # Add crashing thread to threads list
    if crashing_thread and crashing_thread not in threads:
        threads.insert(0, crashing_thread)

    # Record total thread count from minidump even if we don't walk them
    if mdf.threads is not None and not threads:
        for t in mdf.threads.threads:
            threads.append(ThreadInfo(
                thread_id=t.ThreadId or 0,
                is_crashing_thread=(
                    crashing_thread_id is not None
                    and t.ThreadId == crashing_thread_id
                ),
            ))

    # --- Try DbgHelp symbol resolution on stack frames ---
    if crashing_thread and crashing_thread.stack_frames:
        sym_warnings = _try_resolve_symbols(
            path, crashing_thread.stack_frames, modules,
            symbol_paths=symbol_paths,
        )
        warnings.extend(sym_warnings)

    # --- System Info ---
    sys_info: dict[str, str] = {}
    if mdf.sysinfo is not None:
        si = mdf.sysinfo
        if si.OperatingSystem:
            sys_info["os"] = str(si.OperatingSystem)
        if si.ProcessorArchitecture:
            sys_info["arch"] = si.ProcessorArchitecture.name
        if si.NumberOfProcessors:
            sys_info["processors"] = str(si.NumberOfProcessors)
        if si.BuildNumber:
            sys_info["build"] = str(si.BuildNumber)
        if si.MajorVersion is not None and si.MinorVersion is not None:
            sys_info["version"] = f"{si.MajorVersion}.{si.MinorVersion}"

    # Enrich with CrashContext metadata
    if crash_ctx:
        if crash_ctx.get("error_message"):
            sys_info["ue_error"] = crash_ctx["error_message"]
        if crash_ctx.get("command_line"):
            sys_info["command_line"] = crash_ctx["command_line"]
        if crash_ctx.get("cpu"):
            sys_info["cpu"] = crash_ctx["cpu"]
        if crash_ctx.get("gpu"):
            sys_info["gpu"] = crash_ctx["gpu"]

    # Thread count: use minidump thread count as authoritative
    thread_count = len(mdf.threads.threads) if mdf.threads else len(threads)

    return MinidumpResult(
        file_path=str(path),
        file_size_mb=file_size_mb,
        dump_type=dump_type,
        exception=exception,
        crashing_thread=crashing_thread,
        threads=threads if len(threads) > 1 else [ThreadInfo(t.ThreadId or 0, t.ThreadId == crashing_thread_id) for t in (mdf.threads.threads if mdf.threads else [])],
        modules=modules,
        system_info=sys_info,
        unresolved_frame_count=unresolved_count,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# CrashContext.runtime-xml parser
# ---------------------------------------------------------------------------

def _parse_crash_context(dump_path: Path) -> dict[str, Any]:
    """Parse UE CrashContext.runtime-xml alongside a .dmp file.

    Returns a dict with 'error_message', 'callstack' (list of dicts
    with module/offset), 'command_line', etc. Returns empty dict if
    file not found.
    """
    ctx_path = dump_path.parent / "CrashContext.runtime-xml"
    if not ctx_path.exists():
        return {}

    try:
        raw = ctx_path.read_bytes()
    except OSError:
        return {}

    # The file is UTF-16LE with BOM or wide chars with nulls between
    # Decode it
    text = ""
    try:
        text = raw.decode("utf-16-le", errors="replace")
    except Exception:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            return {}

    # Remove null chars that may linger
    text = text.replace("\x00", "")

    result: dict[str, Any] = {}

    # Extract key fields using simple regex on the XML-like content
    def _extract(tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    result["error_message"] = _extract("ErrorMessage")
    result["crash_type"] = _extract("CrashType")
    result["game_name"] = _extract("GameName")
    result["build_configuration"] = _extract("BuildConfiguration")
    result["engine_version"] = _extract("EngineVersion")
    result["command_line"] = _extract("CommandLine")
    result["engine_mode"] = _extract("EngineMode")
    result["platform"] = _extract("PlatformFullName")
    result["cpu"] = _extract("Misc.CPUBrand")
    result["gpu"] = _extract("Misc.PrimaryGPUBrand")
    result["os"] = _extract("Misc.OSVersionMajor")

    # Parse PCallStack — the portable call stack from UE
    pcallstack_raw = _extract("PCallStack")
    frames: list[dict[str, Any]] = []
    if pcallstack_raw:
        for i, line in enumerate(pcallstack_raw.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            # Format: "ModuleName  0xBASE + OFFSET"
            parts = line.split()
            if len(parts) >= 3 and "+" in parts:
                plus_idx = parts.index("+")
                mod_name = " ".join(parts[:plus_idx - 1])
                base_str = parts[plus_idx - 1]
                offset_str = parts[plus_idx + 1] if plus_idx + 1 < len(parts) else "0"
                try:
                    base = int(base_str, 16)
                    offset = int(offset_str, 16)
                    frames.append({
                        "index": i,
                        "module": mod_name,
                        "base": base,
                        "offset": offset,
                        "address": base + offset,
                    })
                except ValueError:
                    pass

    result["callstack"] = frames
    return result


# ---------------------------------------------------------------------------
# Symbol resolution via DbgHelp (Windows only)
# ---------------------------------------------------------------------------

def _try_resolve_symbols(
    dump_path: Path,
    frames: list[StackFrame],
    modules: list[ModuleInfo],
    symbol_paths: list[str] | None = None,
) -> list[str]:
    """Attempt to resolve frame addresses to function names via dbghelp.

    Updates frames in-place with symbol_name, source_file, source_line.
    Returns list of warnings.
    """
    warnings: list[str] = []

    try:
        from dbghelp_walker import resolve_addresses
    except ImportError:
        warnings.append("DbgHelp not available (Windows-only). No symbol resolution.")
        return warnings
    except Exception as e:
        warnings.append(f"DbgHelp init failed: {e}")
        return warnings

    addresses = [f.address for f in frames]
    if not addresses:
        return warnings

    mod_dicts = [
        {"name": m.name, "base_address": m.base_address, "size": m.size}
        for m in modules
    ]

    try:
        sym_paths = list(symbol_paths or [])
        # Add engine binaries dir if we can detect it from module paths
        for m in modules:
            if "UnrealEditor-Engine" in m.name:
                engine_bin = str(Path(m.name).parent)
                if engine_bin not in sym_paths:
                    sym_paths.append(engine_bin)
                break

        resolved = resolve_addresses(
            addresses, mod_dicts, dump_path, symbol_paths=sym_paths
        )

        resolved_count = 0
        for frame, sym in zip(frames, resolved):
            if sym.function_name:
                frame.symbol_name = sym.function_name
                frame.source_file = sym.source_file
                frame.source_line = sym.source_line
                resolved_count += 1

        warnings.append(f"Symbols resolved: {resolved_count}/{len(frames)} frames")
    except Exception as e:
        warnings.append(f"Symbol resolution failed: {e}")

    return warnings


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_summary(result: MinidumpResult) -> str:
    """Build a human-readable crash summary."""

    lines: list[str] = []

    # Warnings
    for w in result.warnings:
        lines.append(f"WARNING: {w}")

    # Dump info
    lines.append(f"Dump: {result.dump_type} ({result.file_size_mb} MB)")

    # System info
    if result.system_info:
        parts = ", ".join(f"{k}={v}" for k, v in result.system_info.items())
        lines.append(f"System: {parts}")

    # Exception
    if result.exception:
        exc = result.exception
        if exc.module_name == "<unknown>":
            loc = f"<unknown>@0x{exc.address:016X}"
        else:
            loc = f"{exc.module_name}+0x{exc.module_offset:X}"
        lines.append(
            f"Exception: {exc.code_name} (0x{exc.code:08X}) at {loc}"
        )
        lines.append(f"Crashing Thread ID: {exc.thread_id}")
    else:
        lines.append("Exception: None found in dump")

    # Crashing thread stack
    if result.crashing_thread and result.crashing_thread.stack_frames:
        lines.append("Crashing Thread Stack:")
        for f in result.crashing_thread.stack_frames:
            if f.module_name == "<unknown>":
                loc = f"<unknown>@0x{f.address:016X}"
            else:
                loc = f"{f.module_name}+0x{f.module_offset:X}"
            if f.symbol_name:
                loc += f" ({f.symbol_name}+0x{0:X})"
                if f.source_file:
                    loc += f" [{f.source_file}:{f.source_line}]"
            lines.append(f"  {f.frame_index}. {loc}")

    # Module summary
    ue_modules = [m for m in result.modules if m.is_ue_module]
    sys_modules = [m for m in result.modules if not m.is_ue_module]

    lines.append(
        f"Loaded Modules: {len(result.modules)} total, "
        f"{len(ue_modules)} UE modules"
    )

    if ue_modules:
        names = ", ".join(m.ue_module_name for m in ue_modules[:20])
        suffix = f" ... and {len(ue_modules) - 20} more" if len(ue_modules) > 20 else ""
        lines.append(f"UE Modules: {names}{suffix}")

    lines.append(f"Thread Count: {len(result.threads)}")

    if result.unresolved_frame_count > 0:
        lines.append(
            f"Unresolved Frames: {result.unresolved_frame_count} "
            "(address not in any loaded module)"
        )

    return "\n".join(lines)


def to_analysis_dict(result: MinidumpResult) -> dict[str, Any]:
    """Convert to a dict suitable for LLM consumption / artifact output."""

    exc_dict: dict[str, Any] | None = None
    if result.exception:
        e = result.exception
        exc_dict = {
            "code": f"0x{e.code:08X}",
            "code_name": e.code_name,
            "address": f"0x{e.address:016X}",
            "module": e.module_name,
            "module_offset": f"0x{e.module_offset:X}",
            "thread_id": e.thread_id,
        }

    crashing_stack: list[dict[str, Any]] = []
    if result.crashing_thread:
        for f in result.crashing_thread.stack_frames:
            fd: dict[str, Any] = {
                "index": f.frame_index,
                "address": f"0x{f.address:016X}",
                "module": f.module_name,
                "module_offset": f"0x{f.module_offset:X}",
            }
            if f.symbol_name:
                fd["symbol"] = f.symbol_name
            if f.source_file:
                fd["source"] = f"{f.source_file}:{f.source_line}"
            crashing_stack.append(fd)

    module_list = []
    for m in result.modules:
        d: dict[str, Any] = {
            "name": m.name,
            "base": f"0x{m.base_address:016X}",
            "size": m.size,
        }
        if m.is_ue_module:
            d["ue_module"] = m.ue_module_name
        module_list.append(d)

    return {
        "file_path": result.file_path,
        "file_size_mb": result.file_size_mb,
        "dump_type": result.dump_type,
        "system_info": result.system_info,
        "exception": exc_dict,
        "crashing_thread_stack": crashing_stack,
        "thread_count": len(result.threads),
        "modules": module_list,
        "ue_module_count": sum(1 for m in result.modules if m.is_ue_module),
        "unresolved_frame_count": result.unresolved_frame_count,
        "warnings": result.warnings,
    }
