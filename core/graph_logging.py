from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from threading import Lock, local
from time import perf_counter
from typing import Any, Callable
import sys


GRAPH_TRACE_FILE = "graph_traversal.log"
GRAPH_TIMELINE_FILE = "graph_timeline.md"
GRAPH_DEBUG_TRACE_FILE = "graph_state_debug.jsonl"
LLM_PROMPT_TRACE_FILE = "llm_prompt_trace.md"
_TRACE_LOCK = Lock()
_TRACE_CONTEXT = local()
_DEBUG_VALUE_CHAR_LIMIT = 240
_DEBUG_COLLECTION_ITEM_LIMIT = 8
_DEBUG_MAPPING_KEY_LIMIT = 16
_DEBUG_RECURSION_LIMIT = 2
_DEBUG_KEY_SUMMARY_LIMIT = 8
_PROMPT_TEXT_CHAR_LIMIT = 12000
_TRACE_EVENT_ID = 0


@dataclass(frozen=True)
class _ActiveTraceContext:
    run_dir: Path
    graph_name: str
    node_name: str
    state_context: str


def _resolve_run_dir(state: Mapping[str, Any] | None) -> Path | None:
    if not state:
        return None
    run_dir = state.get("run_dir")
    if isinstance(run_dir, str) and run_dir.strip():
        return Path(run_dir)
    return None


def _summarize_state(state: Mapping[str, Any] | None) -> str:
    if not state:
        return ""

    details: list[str] = []

    task_id = state.get("task_id")
    if isinstance(task_id, str) and task_id.strip():
        details.append(f"task_id={task_id}")

    active_task = state.get("active_task")
    if isinstance(active_task, Mapping):
        active_task_id = active_task.get("id")
        if isinstance(active_task_id, str) and active_task_id.strip():
            details.append(f"active_task={active_task_id}")

        workflow_name = active_task.get("workflow_name")
        if isinstance(workflow_name, str) and workflow_name.strip():
            details.append(f"active_workflow={workflow_name}")

    review_round = state.get("review_round")
    if isinstance(review_round, int) and review_round > 0:
        details.append(f"review_round={review_round}")

    return " ".join(details)


def _clip_debug_text(value: str, limit: int = _DEBUG_VALUE_CHAR_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def _clip_multiline_text(value: str, limit: int = _PROMPT_TEXT_CHAR_LIMIT) -> str:
    clean_value = value.strip()
    if len(clean_value) <= limit:
        return clean_value
    overflow = len(clean_value) - limit
    return f"{clean_value[:limit].rstrip()}\n\n... [truncated {overflow} chars]"


def _normalize_debug_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value

    if isinstance(value, str):
        return _clip_debug_text(value.replace("\n", "\\n"))

    if isinstance(value, Path):
        return str(value)

    if depth >= _DEBUG_RECURSION_LIMIT:
        return _clip_debug_text(repr(value))

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        items = list(value.items())
        for index, (key, item) in enumerate(items):
            if index >= _DEBUG_MAPPING_KEY_LIMIT:
                normalized["..."] = f"{len(items) - _DEBUG_MAPPING_KEY_LIMIT} more keys"
                break
            normalized[str(key)] = _normalize_debug_value(item, depth=depth + 1)
        return normalized

    if isinstance(value, list | tuple | set):
        items = list(value)
        normalized_items = [
            _normalize_debug_value(item, depth=depth + 1)
            for item in items[:_DEBUG_COLLECTION_ITEM_LIMIT]
        ]
        if len(items) > _DEBUG_COLLECTION_ITEM_LIMIT:
            normalized_items.append(f"... ({len(items) - _DEBUG_COLLECTION_ITEM_LIMIT} more items)")
        return normalized_items

    return _clip_debug_text(repr(value))


def _serialize_debug_payload(payload: Mapping[str, Any] | None) -> str:
    if not payload:
        return "{}"
    normalized = _normalize_debug_value(payload)
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True)


def _summarize_payload_keys(payload: Mapping[str, Any] | None) -> str:
    if not payload:
        return "(none)"
    keys = [str(key) for key in payload]
    visible = keys[:_DEBUG_KEY_SUMMARY_LIMIT]
    summary = ", ".join(visible)
    if len(keys) > _DEBUG_KEY_SUMMARY_LIMIT:
        summary = f"{summary}, ... ({len(keys)} total)"
    return summary


def _resolve_trace_context_stack() -> list[_ActiveTraceContext]:
    stack = getattr(_TRACE_CONTEXT, "stack", None)
    if stack is None:
        stack = []
        _TRACE_CONTEXT.stack = stack
    return stack


def _get_active_trace_context() -> _ActiveTraceContext | None:
    stack = getattr(_TRACE_CONTEXT, "stack", None)
    if not stack:
        return None
    return stack[-1]


@contextmanager
def bind_active_trace_context(
    *,
    state: Mapping[str, Any] | None,
    graph_name: str,
    node_name: str,
):
    run_dir = _resolve_run_dir(state)
    if run_dir is None:
        yield
        return

    stack = _resolve_trace_context_stack()
    stack.append(
        _ActiveTraceContext(
            run_dir=run_dir,
            graph_name=graph_name,
            node_name=node_name,
            state_context=_summarize_state(state),
        )
    )
    try:
        yield
    finally:
        stack.pop()


def _ensure_markdown_header(path: Path, title: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n", encoding="utf-8")


def _write_llm_trace_entry(
    *,
    active_context: _ActiveTraceContext,
    event_id: int | None,
    entry_kind: str,
    metadata_lines: list[str],
    sections: list[tuple[str, str]],
) -> None:
    trace_path = active_context.run_dir / LLM_PROMPT_TRACE_FILE
    with _TRACE_LOCK:
        _ensure_markdown_header(trace_path, "LLM Prompt Trace")
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(f"## {entry_kind} Event {event_id or '?'} | {datetime.now().isoformat(timespec='seconds')}\n\n")
            handle.write(f"- Graph node: `{active_context.graph_name}.{active_context.node_name}`\n")
            if active_context.state_context:
                handle.write(f"- Context: `{active_context.state_context}`\n")
            for line in metadata_lines:
                handle.write(f"- {line}\n")
            handle.write("\n")
            for title, body in sections:
                handle.write(f"### {title}\n")
                handle.write("```text\n")
                rendered_body = _clip_multiline_text(str(body))
                handle.write(rendered_body + "\n" if rendered_body else "(empty)\n")
                handle.write("```\n\n")


def _append_timeline_entry(
    *,
    run_dir: Path,
    timestamp: str,
    graph_name: str,
    node_name: str,
    phase: str,
    context: str,
    message: str,
) -> None:
    timeline_path = run_dir / GRAPH_TIMELINE_FILE
    with _TRACE_LOCK:
        _ensure_markdown_header(timeline_path, "Graph Timeline")
        details: list[str] = [f"- {timestamp} | `{graph_name}.{node_name}` | `{phase}`"]
        if context:
            details.append(f"| {context}")
        if message:
            details.append(f"| {message}")
        with timeline_path.open("a", encoding="utf-8") as handle:
            handle.write(" ".join(details).rstrip() + "\n")


def _write_debug_trace_record(
    *,
    state: Mapping[str, Any] | None,
    graph_name: str,
    node_name: str,
    phase: str,
    payload_label: str,
    payload: Mapping[str, Any] | None,
    message: str = "",
    run_dir_override: Path | None = None,
    state_context_override: str | None = None,
) -> int | None:
    run_dir = run_dir_override or _resolve_run_dir(state)
    if run_dir is None:
        return None

    record = {
        "graph": graph_name,
        "node": node_name,
        "phase": phase,
        "payload_label": payload_label,
        "payload": _normalize_debug_value(payload),
        "state_context": state_context_override if state_context_override is not None else _summarize_state(state),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if message:
        record["message"] = message

    with _TRACE_LOCK:
        global _TRACE_EVENT_ID
        _TRACE_EVENT_ID += 1
        event_id = _TRACE_EVENT_ID
        record["event_id"] = event_id
        debug_path = run_dir / GRAPH_DEBUG_TRACE_FILE
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    return event_id


def log_graph_payload_event(
    *,
    state: Mapping[str, Any] | None,
    graph_name: str,
    node_name: str,
    phase: str,
    payload_label: str,
    payload: Mapping[str, Any] | None,
    message: str = "",
    run_dir_override: Path | None = None,
    state_context_override: str | None = None,
) -> None:
    event_id = _write_debug_trace_record(
        state=state,
        graph_name=graph_name,
        node_name=node_name,
        phase=phase,
        payload_label=payload_label,
        payload=payload,
        message=message,
        run_dir_override=run_dir_override,
        state_context_override=state_context_override,
    )
    details: list[str] = []
    if message:
        details.append(message)
    details.append(f"{payload_label}_keys={_summarize_payload_keys(payload)}")
    if event_id is not None:
        details.append(f"details={GRAPH_DEBUG_TRACE_FILE}#{event_id}")
    log_graph_event(
        state=state,
        graph_name=graph_name,
        node_name=node_name,
        phase=phase,
        message=" ".join(details),
        run_dir_override=run_dir_override,
        state_context_override=state_context_override,
    )


def log_graph_event(
    *,
    state: Mapping[str, Any] | None,
    graph_name: str,
    node_name: str,
    phase: str,
    message: str = "",
    run_dir_override: Path | None = None,
    state_context_override: str | None = None,
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    context = state_context_override if state_context_override is not None else _summarize_state(state)
    parts = [
        f"[{timestamp}]",
        f"[{graph_name}]",
        f"[{node_name}]",
        phase,
    ]
    if context:
        parts.append(context)
    if message:
        parts.append(message)
    line = " ".join(parts)

    print(line, file=sys.stderr, flush=True)

    run_dir = run_dir_override or _resolve_run_dir(state)
    if run_dir is None:
        return

    log_path = run_dir / GRAPH_TRACE_FILE
    with _TRACE_LOCK:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    _append_timeline_entry(
        run_dir=run_dir,
        timestamp=timestamp,
        graph_name=graph_name,
        node_name=node_name,
        phase=phase,
        context=context,
        message=message,
    )


def log_llm_prompt_event(
    *,
    client_label: str,
    transport: str,
    instructions: str,
    input_text: str,
    final_prompt: str,
    require_structured_output: bool,
    schema_name: str | None = None,
    effort: str | None = None,
) -> int | None:
    active_context = _get_active_trace_context()
    if active_context is None:
        return None

    payload = {
        "client": client_label,
        "transport": transport,
        "require_structured_output": require_structured_output,
        "schema_name": schema_name or "",
        "effort": effort or "",
        "instructions_chars": len(instructions.strip()),
        "input_text_chars": len(input_text.strip()),
        "final_prompt_chars": len(final_prompt.strip()),
        "instructions": instructions.strip(),
        "input_text": input_text.strip(),
        "final_prompt": final_prompt.strip(),
    }
    details: list[str] = [
        f"client={client_label}",
        f"transport={transport}",
        f"mode={'json' if require_structured_output else 'text'}",
    ]
    if schema_name:
        details.append(f"schema={schema_name}")
    if effort:
        details.append(f"effort={effort}")
    event_id = _write_debug_trace_record(
        state=None,
        graph_name=active_context.graph_name,
        node_name=active_context.node_name,
        phase="PROMPT",
        payload_label="prompt_recipe",
        payload=payload,
        message=" ".join(details),
        run_dir_override=active_context.run_dir,
        state_context_override=active_context.state_context,
    )
    if event_id is not None:
        details.append(f"details={LLM_PROMPT_TRACE_FILE}#{event_id}")
    log_graph_event(
        state=None,
        graph_name=active_context.graph_name,
        node_name=active_context.node_name,
        phase="PROMPT",
        message=" ".join(details),
        run_dir_override=active_context.run_dir,
        state_context_override=active_context.state_context,
    )
    metadata_lines = [
        f"Client: `{client_label}`",
        f"Transport: `{transport}`",
        f"Mode: `{'json' if require_structured_output else 'text'}`",
    ]
    if schema_name:
        metadata_lines.append(f"Schema: `{schema_name}`")
    if effort:
        metadata_lines.append(f"Effort: `{effort}`")
    metadata_lines.extend(
        [
            f"Instructions chars: `{len(instructions.strip())}`",
            f"Input chars: `{len(input_text.strip())}`",
            f"Final prompt chars: `{len(final_prompt.strip())}`",
        ]
    )
    _write_llm_trace_entry(
        active_context=active_context,
        event_id=event_id,
        entry_kind="Prompt",
        metadata_lines=metadata_lines,
        sections=[
            ("Instructions", instructions),
            ("Input Text", input_text),
            ("Final Prompt", final_prompt),
        ],
    )
    return event_id


def log_llm_response_event(
    *,
    client_label: str,
    transport: str,
    require_structured_output: bool,
    response_text: str,
    elapsed_ms: float | None,
    schema_name: str | None = None,
    request_event_id: int | None = None,
    error: str | None = None,
) -> int | None:
    active_context = _get_active_trace_context()
    if active_context is None:
        return None

    clean_response = response_text.strip()
    clean_error = str(error).strip() if error else ""
    status = "error" if clean_error else "ok"
    payload = {
        "client": client_label,
        "transport": transport,
        "require_structured_output": require_structured_output,
        "schema_name": schema_name or "",
        "request_event_id": request_event_id,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "response_chars": len(clean_response),
        "response_text": clean_response,
        "error": clean_error,
    }
    details = [
        f"client={client_label}",
        f"transport={transport}",
        f"mode={'json' if require_structured_output else 'text'}",
        f"status={status}",
    ]
    if schema_name:
        details.append(f"schema={schema_name}")
    if request_event_id is not None:
        details.append(f"request={request_event_id}")
    if elapsed_ms is not None:
        details.append(f"elapsed_ms={elapsed_ms}")
    event_id = _write_debug_trace_record(
        state=None,
        graph_name=active_context.graph_name,
        node_name=active_context.node_name,
        phase="RESPONSE",
        payload_label="response",
        payload=payload,
        message=" ".join(details),
        run_dir_override=active_context.run_dir,
        state_context_override=active_context.state_context,
    )
    if event_id is not None:
        details.append(f"details={LLM_PROMPT_TRACE_FILE}#{event_id}")
    log_graph_event(
        state=None,
        graph_name=active_context.graph_name,
        node_name=active_context.node_name,
        phase="RESPONSE",
        message=" ".join(details),
        run_dir_override=active_context.run_dir,
        state_context_override=active_context.state_context,
    )
    metadata_lines = [
        f"Client: `{client_label}`",
        f"Transport: `{transport}`",
        f"Mode: `{'json' if require_structured_output else 'text'}`",
        f"Status: `{status}`",
    ]
    if schema_name:
        metadata_lines.append(f"Schema: `{schema_name}`")
    if request_event_id is not None:
        metadata_lines.append(f"Request event: `{request_event_id}`")
    if elapsed_ms is not None:
        metadata_lines.append(f"Elapsed ms: `{elapsed_ms}`")
    metadata_lines.append(f"Response chars: `{len(clean_response)}`")
    sections = [("Model Response", clean_response)]
    if clean_error:
        sections.append(("Error", clean_error))
    _write_llm_trace_entry(
        active_context=active_context,
        event_id=event_id,
        entry_kind="Response",
        metadata_lines=metadata_lines,
        sections=sections,
    )
    return event_id


def trace_graph_node(
    *,
    graph_name: str,
    node_name: str,
    node_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def wrapper(state: dict[str, Any]) -> dict[str, Any]:
        start_time = perf_counter()
        with bind_active_trace_context(state=state, graph_name=graph_name, node_name=node_name):
            log_graph_payload_event(
                state=state,
                graph_name=graph_name,
                node_name=node_name,
                phase="ENTER",
                payload_label="input",
                payload=state,
            )
            try:
                result = node_fn(state)
            except Exception as exc:
                elapsed_ms = round((perf_counter() - start_time) * 1000, 2)
                log_graph_payload_event(
                    state=state,
                    graph_name=graph_name,
                    node_name=node_name,
                    phase="ERROR",
                    payload_label="input",
                    payload=state,
                    message=f"elapsed_ms={elapsed_ms} {type(exc).__name__}: {exc}",
                )
                raise

        elapsed_ms = round((perf_counter() - start_time) * 1000, 2)
        updated_keys = ", ".join(sorted(result)) if result else "no_state_updates"
        log_graph_payload_event(
            state=state,
            graph_name=graph_name,
            node_name=node_name,
            phase="EXIT",
            payload_label="output",
            payload=result,
            message=f"elapsed_ms={elapsed_ms} updated_keys={updated_keys}",
        )
        return result

    return wrapper


def trace_route_decision(
    *,
    graph_name: str,
    router_name: str,
    route_fn: Callable[[dict[str, Any]], str],
) -> Callable[[dict[str, Any]], str]:
    def wrapper(state: dict[str, Any]) -> str:
        start_time = perf_counter()
        log_graph_event(state=state, graph_name=graph_name, node_name=router_name, phase="ROUTE_EVAL")
        try:
            next_node = route_fn(state)
        except Exception as exc:
            elapsed_ms = round((perf_counter() - start_time) * 1000, 2)
            log_graph_payload_event(
                state=state,
                graph_name=graph_name,
                node_name=router_name,
                phase="ERROR",
                payload_label="input",
                payload=state,
                message=f"elapsed_ms={elapsed_ms} {type(exc).__name__}: {exc}",
            )
            raise

        elapsed_ms = round((perf_counter() - start_time) * 1000, 2)
        log_graph_event(
            state=state,
            graph_name=graph_name,
            node_name=router_name,
            phase="ROUTE",
            message=f"elapsed_ms={elapsed_ms} next={next_node}",
        )
        return next_node

    return wrapper
