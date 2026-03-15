from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from core.models import ToolRuntime


def build_tool_call_message(tool_name: str, args: dict[str, Any], call_id: str, *, content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": tool_name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def find_latest_tool_message(
    messages: list[Any],
    *,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
) -> ToolMessage | None:
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        if tool_name is not None and message.name != tool_name:
            continue
        if tool_call_id is not None and message.tool_call_id != tool_call_id:
            continue
        return message
    return None


def build_tool_subgraph(runtime: ToolRuntime, state_schema: type[Any]) -> Any:
    graph = StateGraph(state_schema)
    graph.add_node("execute_tool", ToolNode([runtime.tool], name=runtime.metadata.name))
    graph.add_edge(START, "execute_tool")
    graph.add_edge("execute_tool", END)
    return graph.compile(name=runtime.metadata.name)
