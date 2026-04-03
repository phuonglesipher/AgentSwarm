from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from core.llm import LLMError

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


# ------------------------------------------------------------------ #
#  Data classes
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ToolCall:
    """Record of a single tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    result: str
    error: str | None = None
    turn: int = 0
    artifact: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolEngineConfig:
    """Profile for a ToolEngine run.

    Analogous to ``ReviewProfile`` and ``PlanProfile`` — configures
    behavior without touching implementation.
    """

    system_id: str
    persona: str = ""
    task_framing: str = ""
    max_turns: int = 5
    max_result_chars: int = 3000
    require_tool_use: bool = False
    allow_parallel_calls: bool = True


@dataclass
class ToolEngineResult:
    """Output from a ToolEngine run.

    Workflows consume this to inject tool-gathered context into
    investigation prompts, plan inputs, etc.
    """

    summary: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns_used: int = 0
    success: bool = False

    @property
    def has_tool_output(self) -> bool:
        return any(tc.error is None for tc in self.tool_calls)

    def first_artifact(self, tool_name: str | None = None) -> dict[str, Any] | None:
        """Return the first non-None artifact, optionally filtered by tool name."""
        for tc in self.tool_calls:
            if tc.artifact is not None and (tool_name is None or tc.tool_name == tool_name):
                return tc.artifact
        return None

    def tool_results_text(self, *, max_chars: int = 8000) -> str:
        """Format all tool results as markdown for injection into prompts."""
        blocks: list[str] = []
        for tc in self.tool_calls:
            if tc.error:
                blocks.append(f"### {tc.tool_name} (error)\n{tc.error}")
            else:
                blocks.append(f"### {tc.tool_name}\n{tc.result}")
        text = "\n\n".join(blocks)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated ...]"
        return text


# ------------------------------------------------------------------ #
#  Engine
# ------------------------------------------------------------------ #


class ToolEngine:
    """LLM-driven tool orchestrator.

    Given a set of tools and a task, lets the LLM autonomously decide
    which tools to call.  The engine manages the request/response loop,
    tool execution, result accumulation, and final synthesis.

    Usage in a workflow node::

        tools = [context.get_tool(name).tool for name in metadata.tools]
        engine = ToolEngine(
            config=ToolEngineConfig(system_id="crash-gather", persona="..."),
            tools=tools,
            llm=context.get_llm("investigator"),
        )
        result = engine.gather(task="Gather crash context for: ...", context="...")
    """

    def __init__(
        self,
        config: ToolEngineConfig,
        tools: list[BaseTool],
        llm: Any,
    ) -> None:
        self._config = config
        self._tools = tools
        self._llm = llm
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #

    def gather(self, task: str, *, context: str = "") -> ToolEngineResult:
        """Run the tool-use agent loop.

        Args:
            task: What the LLM should accomplish using the available tools.
            context: Optional background context (project info, prior results).

        Returns:
            ``ToolEngineResult`` with summary, tool call records, and success flag.
        """
        cfg = self._config

        if not self._llm.is_enabled():
            return ToolEngineResult(
                summary="Tool engine skipped: LLM unavailable.",
                turns_used=0,
                success=False,
            )

        if not self._tools:
            return ToolEngineResult(
                summary="Tool engine skipped: no tools available.",
                turns_used=0,
                success=False,
            )

        tool_descriptions = self._build_tool_descriptions()
        schema = self._build_response_schema()
        all_calls: list[ToolCall] = []

        for turn in range(cfg.max_turns):
            input_text = self._build_turn_input(task, context, all_calls)

            try:
                response = self._llm.generate_json(
                    instructions=self._build_instructions(tool_descriptions),
                    input_text=input_text,
                    schema_name=f"{cfg.system_id}_tool_turn_{turn}",
                    schema=schema,
                )
            except LLMError:
                return ToolEngineResult(
                    summary=self._summarize_from_calls(all_calls),
                    tool_calls=all_calls,
                    turns_used=turn + 1,
                    success=bool(all_calls),
                )

            if response.get("done", False) or not response.get("tool_calls"):
                final_answer = str(response.get("final_answer", "")).strip()
                if not final_answer:
                    final_answer = str(response.get("reasoning", "")).strip()
                success = True
                if cfg.require_tool_use and not all_calls:
                    success = False
                return ToolEngineResult(
                    summary=final_answer,
                    tool_calls=all_calls,
                    turns_used=turn + 1,
                    success=success,
                )

            requested_calls = response.get("tool_calls", [])
            if not cfg.allow_parallel_calls and len(requested_calls) > 1:
                requested_calls = requested_calls[:1]

            for call_spec in requested_calls:
                tool_name = str(call_spec.get("tool_name", ""))
                arguments = dict(call_spec.get("arguments", {}))
                tc = self._execute_tool(tool_name, arguments, turn=turn)
                all_calls.append(tc)

        return ToolEngineResult(
            summary=self._summarize_from_calls(all_calls),
            tool_calls=all_calls,
            turns_used=cfg.max_turns,
            success=bool(all_calls),
        )

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _build_tool_descriptions(self) -> str:
        lines: list[str] = []
        for tool in self._tools:
            desc = str(getattr(tool, "description", "No description."))
            args_schema = getattr(tool, "args_schema", None)
            args_desc = "no arguments"
            if args_schema is not None:
                schema_fn = getattr(args_schema, "schema", None) or getattr(args_schema, "model_json_schema", None)
                if callable(schema_fn):
                    try:
                        schema_dict = schema_fn()
                        props = schema_dict.get("properties", {})
                        if props:
                            args_desc = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
                    except Exception:
                        pass
            lines.append(f"- **{tool.name}** ({args_desc}): {desc}")
        return "\n".join(lines)

    def _build_response_schema(self) -> dict[str, Any]:
        tool_names = [t.name for t in self._tools]
        return {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning about what tools to use and why, or why you are done.",
                },
                "tool_calls": {
                    "type": "array",
                    "description": "Tools to invoke this turn. Empty array or omit if done.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string", "enum": tool_names},
                            "arguments": {"type": "object"},
                        },
                        "required": ["tool_name", "arguments"],
                        "additionalProperties": False,
                    },
                },
                "final_answer": {
                    "type": "string",
                    "description": "Final synthesized answer when done. Only provide when done=true.",
                },
                "done": {
                    "type": "boolean",
                    "description": "Set to true when you have enough information and are providing final_answer.",
                },
            },
            "required": ["reasoning", "done"],
            "additionalProperties": False,
        }

    def _build_instructions(self, tool_descriptions: str) -> str:
        cfg = self._config
        persona = cfg.persona or "You are a tool-calling agent."
        return (
            f"{persona}\n\n"
            f"You have access to these tools:\n{tool_descriptions}\n\n"
            "On each turn, decide which tools to call based on what information you still need. "
            "When you have gathered enough information, set done=true and provide a final_answer "
            "that synthesizes all tool results into a structured summary.\n\n"
            "Rules:\n"
            "- Only call tools that exist in the list above.\n"
            "- Provide valid arguments matching each tool's expected parameters.\n"
            "- Do not fabricate tool results — only report what tools actually return.\n"
            "- Be efficient: don't call the same tool with the same arguments twice.\n"
            f"- You have at most {cfg.max_turns} turns before the engine stops.\n"
        )

    def _build_turn_input(self, task: str, context: str, prior_calls: list[ToolCall]) -> str:
        cfg = self._config
        parts = [f"## Task\n{task}"]

        if context.strip():
            parts.append(f"\n## Context\n{context}")

        if cfg.task_framing:
            parts.append(f"\n## Focus\n{cfg.task_framing}")

        if prior_calls:
            parts.append("\n## Tool Results So Far")
            for tc in prior_calls:
                if tc.error:
                    parts.append(f"\n### {tc.tool_name} (turn {tc.turn}, ERROR)\n{tc.error}")
                else:
                    truncated = tc.result[:cfg.max_result_chars]
                    if len(tc.result) > cfg.max_result_chars:
                        truncated += "\n[... truncated ...]"
                    parts.append(f"\n### {tc.tool_name} (turn {tc.turn})\n{truncated}")
        else:
            parts.append("\n## Tool Results So Far\nNo tools called yet. This is the first turn.")

        return "\n".join(parts)

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any], *, turn: int) -> ToolCall:
        cfg = self._config
        tool = self._tool_map.get(tool_name)
        if tool is None:
            return ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result="",
                error=f"Unknown tool: {tool_name}. Available: {', '.join(sorted(self._tool_map))}",
                turn=turn,
            )
        try:
            raw_result = tool.invoke(arguments)
            artifact: dict[str, Any] | None = None
            if isinstance(raw_result, tuple):
                result_text = str(raw_result[0])
                if len(raw_result) >= 2 and isinstance(raw_result[1], dict):
                    artifact = raw_result[1]
            else:
                result_text = str(raw_result)
            if len(result_text) > cfg.max_result_chars:
                result_text = result_text[:cfg.max_result_chars] + "\n[... truncated ...]"
            return ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=result_text,
                turn=turn,
                artifact=artifact,
            )
        except Exception as exc:
            return ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result="",
                error=f"Tool execution failed: {type(exc).__name__}: {str(exc)[:300]}",
                turn=turn,
            )

    def _summarize_from_calls(self, calls: list[ToolCall]) -> str:
        if not calls:
            return "No tools were called."
        successful = [tc for tc in calls if tc.error is None]
        failed = [tc for tc in calls if tc.error is not None]
        parts: list[str] = []
        if successful:
            parts.append(f"Called {len(successful)} tool(s) successfully: {', '.join(tc.tool_name for tc in successful)}.")
        if failed:
            parts.append(f"{len(failed)} tool call(s) failed: {', '.join(tc.tool_name for tc in failed)}.")
        return " ".join(parts)
