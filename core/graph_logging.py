from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
from pathlib import Path
from threading import Lock
from typing import Any, Callable
import sys


GRAPH_TRACE_FILE = "graph_traversal.log"
_TRACE_LOCK = Lock()
_DEBUG_VALUE_CHAR_LIMIT = 240
_DEBUG_COLLECTION_ITEM_LIMIT = 8
_DEBUG_MAPPING_KEY_LIMIT = 16
_DEBUG_RECURSION_LIMIT = 2


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

        blueprint_name = active_task.get("blueprint_name")
        if isinstance(blueprint_name, str) and blueprint_name.strip():
            details.append(f"active_blueprint={blueprint_name}")

    review_round = state.get("review_round")
    if isinstance(review_round, int) and review_round > 0:
        details.append(f"review_round={review_round}")

    return " ".join(details)


def _clip_debug_text(value: str, limit: int = _DEBUG_VALUE_CHAR_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


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


def log_graph_event(
    *,
    state: Mapping[str, Any] | None,
    graph_name: str,
    node_name: str,
    phase: str,
    message: str = "",
) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    context = _summarize_state(state)
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

    run_dir = _resolve_run_dir(state)
    if run_dir is None:
        return

    log_path = run_dir / GRAPH_TRACE_FILE
    with _TRACE_LOCK:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def trace_graph_node(
    *,
    graph_name: str,
    node_name: str,
    node_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def wrapper(state: dict[str, Any]) -> dict[str, Any]:
        log_graph_event(
            state=state,
            graph_name=graph_name,
            node_name=node_name,
            phase="ENTER",
            message=f"input={_serialize_debug_payload(state)}",
        )
        try:
            result = node_fn(state)
        except Exception as exc:
            log_graph_event(
                state=state,
                graph_name=graph_name,
                node_name=node_name,
                phase="ERROR",
                message=f"{type(exc).__name__}: {exc} input={_serialize_debug_payload(state)}",
            )
            raise

        updated_keys = ", ".join(sorted(result)) if result else "no_state_updates"
        log_graph_event(
            state=state,
            graph_name=graph_name,
            node_name=node_name,
            phase="EXIT",
            message=f"updated_keys={updated_keys} output={_serialize_debug_payload(result)}",
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
        log_graph_event(state=state, graph_name=graph_name, node_name=router_name, phase="ROUTE_EVAL")
        try:
            next_node = route_fn(state)
        except Exception as exc:
            log_graph_event(
                state=state,
                graph_name=graph_name,
                node_name=router_name,
                phase="ERROR",
                message=f"{type(exc).__name__}: {exc}",
            )
            raise

        log_graph_event(
            state=state,
            graph_name=graph_name,
            node_name=router_name,
            phase="ROUTE",
            message=f"next={next_node}",
        )
        return next_node

    return wrapper
