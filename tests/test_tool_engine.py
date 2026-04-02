from __future__ import annotations

import unittest

from core.tool_engine import ToolCall, ToolEngine, ToolEngineConfig, ToolEngineResult
from core.llm import LLMError


# ------------------------------------------------------------------ #
#  Test doubles
# ------------------------------------------------------------------ #


class MockTool:
    """Minimal BaseTool stand-in for testing."""

    def __init__(
        self,
        name: str,
        result: str = "ok",
        *,
        description: str | None = None,
        raise_on_invoke: Exception | None = None,
    ) -> None:
        self.name = name
        self.description = description or f"Mock tool: {name}"
        self.args_schema = None
        self._result = result
        self._raise = raise_on_invoke
        self.invocations: list[dict] = []

    def invoke(self, args: dict) -> str:
        self.invocations.append(args)
        if self._raise:
            raise self._raise
        return self._result


class DisabledLLM:
    def is_enabled(self) -> bool:
        return False

    def describe(self) -> str:
        return "disabled"

    def generate_json(self, **kwargs) -> dict:
        raise AssertionError("Should not be called when disabled")


class MockLLMForTools:
    """Returns scripted responses, one per ``generate_json`` call."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.calls: list[dict] = []

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "mock-tool-llm"

    def generate_json(self, *, instructions: str, input_text: str, schema_name: str, schema: dict) -> dict:
        self.calls.append({"instructions": instructions, "input_text": input_text, "schema_name": schema_name})
        if self._call_index >= len(self._responses):
            return {"reasoning": "exhausted", "done": True, "final_answer": "no more scripted responses"}
        response = self._responses[self._call_index]
        self._call_index += 1
        if isinstance(response, Exception):
            raise response
        return response


class FailingLLM:
    """LLM that always raises LLMError."""

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "failing-llm"

    def generate_json(self, **kwargs) -> dict:
        raise LLMError("simulated failure")


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestToolEngine(unittest.TestCase):

    def _make_config(self, **overrides) -> ToolEngineConfig:
        defaults = {"system_id": "test-engine"}
        defaults.update(overrides)
        return ToolEngineConfig(**defaults)

    # -- Core flow --

    def test_gather_calls_tool_then_finishes(self):
        """LLM requests a tool on turn 0, then returns done on turn 1."""
        tool = MockTool("scanner", result="crash log found at Saved/Crashes/crash_001.log")
        llm = MockLLMForTools([
            {
                "reasoning": "Need crash logs",
                "done": False,
                "tool_calls": [{"tool_name": "scanner", "arguments": {"path": "/logs"}}],
            },
            {
                "reasoning": "Got what I need",
                "done": True,
                "final_answer": "Found crash log at Saved/Crashes/crash_001.log",
            },
        ])
        engine = ToolEngine(config=self._make_config(), tools=[tool], llm=llm)
        result = engine.gather(task="Find crash logs")

        self.assertTrue(result.success)
        self.assertEqual(result.turns_used, 2)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].tool_name, "scanner")
        self.assertIn("crash log", result.tool_calls[0].result)
        self.assertIn("crash log", result.summary)
        self.assertEqual(len(tool.invocations), 1)

    def test_gather_done_immediately(self):
        """LLM decides no tools needed, returns done on first turn."""
        tool = MockTool("scanner")
        llm = MockLLMForTools([
            {
                "reasoning": "Task prompt already has all the info",
                "done": True,
                "final_answer": "No tools needed.",
            },
        ])
        engine = ToolEngine(config=self._make_config(), tools=[tool], llm=llm)
        result = engine.gather(task="Already have context")

        self.assertTrue(result.success)
        self.assertEqual(result.turns_used, 1)
        self.assertEqual(len(result.tool_calls), 0)
        self.assertEqual(result.summary, "No tools needed.")
        self.assertEqual(len(tool.invocations), 0)

    def test_gather_done_with_empty_tool_calls(self):
        """LLM returns done=False but empty tool_calls — treated as done."""
        llm = MockLLMForTools([
            {
                "reasoning": "Nothing to call",
                "done": False,
                "tool_calls": [],
                "final_answer": "Done via empty calls.",
            },
        ])
        engine = ToolEngine(config=self._make_config(), tools=[MockTool("t")], llm=llm)
        result = engine.gather(task="test")

        self.assertTrue(result.success)
        self.assertEqual(result.turns_used, 1)

    # -- Max turns --

    def test_gather_max_turns_exhausted(self):
        """LLM keeps requesting tools beyond max_turns."""
        tool = MockTool("scanner", result="data")
        call_response = {
            "reasoning": "Need more",
            "done": False,
            "tool_calls": [{"tool_name": "scanner", "arguments": {}}],
        }
        llm = MockLLMForTools([call_response, call_response, call_response])
        engine = ToolEngine(config=self._make_config(max_turns=2), tools=[tool], llm=llm)
        result = engine.gather(task="test")

        self.assertEqual(result.turns_used, 2)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertTrue(result.success)  # Has calls, even though max turns
        self.assertIn("scanner", result.summary)

    # -- Error handling --

    def test_gather_unknown_tool_handled(self):
        """LLM requests a tool not in the tool list."""
        tool = MockTool("real_tool")
        llm = MockLLMForTools([
            {
                "reasoning": "Call nonexistent",
                "done": False,
                "tool_calls": [{"tool_name": "ghost_tool", "arguments": {}}],
            },
            {
                "reasoning": "Done",
                "done": True,
                "final_answer": "Finished despite error.",
            },
        ])
        engine = ToolEngine(config=self._make_config(), tools=[tool], llm=llm)
        result = engine.gather(task="test")

        self.assertTrue(result.success)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertIsNotNone(result.tool_calls[0].error)
        self.assertIn("Unknown tool", result.tool_calls[0].error)
        self.assertEqual(len(tool.invocations), 0)

    def test_gather_tool_execution_error(self):
        """Tool raises exception during invoke."""
        tool = MockTool("broken", raise_on_invoke=RuntimeError("disk full"))
        llm = MockLLMForTools([
            {
                "reasoning": "Call broken tool",
                "done": False,
                "tool_calls": [{"tool_name": "broken", "arguments": {}}],
            },
            {
                "reasoning": "Done",
                "done": True,
                "final_answer": "Tool failed but we continue.",
            },
        ])
        engine = ToolEngine(config=self._make_config(), tools=[tool], llm=llm)
        result = engine.gather(task="test")

        self.assertTrue(result.success)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertIsNotNone(result.tool_calls[0].error)
        self.assertIn("disk full", result.tool_calls[0].error)

    def test_gather_llm_error_mid_loop(self):
        """LLM fails on turn 1 after successful turn 0."""
        tool = MockTool("scanner", result="data")
        llm = MockLLMForTools([
            {
                "reasoning": "Call tool",
                "done": False,
                "tool_calls": [{"tool_name": "scanner", "arguments": {}}],
            },
            LLMError("connection reset"),
        ])
        engine = ToolEngine(config=self._make_config(), tools=[tool], llm=llm)
        result = engine.gather(task="test")

        self.assertTrue(result.success)  # Has successful calls
        self.assertEqual(result.turns_used, 2)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertIn("scanner", result.summary)

    # -- Guards --

    def test_gather_llm_disabled(self):
        """LLM is not enabled."""
        engine = ToolEngine(config=self._make_config(), tools=[MockTool("t")], llm=DisabledLLM())
        result = engine.gather(task="test")

        self.assertFalse(result.success)
        self.assertEqual(result.turns_used, 0)
        self.assertIn("unavailable", result.summary)

    def test_gather_no_tools(self):
        """Empty tool list."""
        llm = MockLLMForTools([{"reasoning": "x", "done": True, "final_answer": "y"}])
        engine = ToolEngine(config=self._make_config(), tools=[], llm=llm)
        result = engine.gather(task="test")

        self.assertFalse(result.success)
        self.assertEqual(result.turns_used, 0)
        self.assertIn("no tools", result.summary)

    def test_gather_require_tool_use_enforced(self):
        """Config require_tool_use=True, LLM returns done without calling tools."""
        llm = MockLLMForTools([
            {"reasoning": "Skip tools", "done": True, "final_answer": "No need."},
        ])
        engine = ToolEngine(
            config=self._make_config(require_tool_use=True),
            tools=[MockTool("t")],
            llm=llm,
        )
        result = engine.gather(task="test")

        self.assertFalse(result.success)
        self.assertEqual(result.turns_used, 1)
        self.assertEqual(len(result.tool_calls), 0)

    # -- Parallel calls --

    def test_gather_parallel_calls_disabled(self):
        """Config allow_parallel_calls=False, LLM requests 3 tools."""
        tools = [MockTool("a", result="ra"), MockTool("b", result="rb"), MockTool("c", result="rc")]
        llm = MockLLMForTools([
            {
                "reasoning": "Call all three",
                "done": False,
                "tool_calls": [
                    {"tool_name": "a", "arguments": {}},
                    {"tool_name": "b", "arguments": {}},
                    {"tool_name": "c", "arguments": {}},
                ],
            },
            {"reasoning": "Done", "done": True, "final_answer": "ok"},
        ])
        engine = ToolEngine(
            config=self._make_config(allow_parallel_calls=False),
            tools=tools,
            llm=llm,
        )
        result = engine.gather(task="test")

        self.assertTrue(result.success)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].tool_name, "a")

    def test_gather_parallel_calls_enabled(self):
        """Default: LLM can request multiple tools per turn."""
        tools = [MockTool("a", result="ra"), MockTool("b", result="rb")]
        llm = MockLLMForTools([
            {
                "reasoning": "Call both",
                "done": False,
                "tool_calls": [
                    {"tool_name": "a", "arguments": {}},
                    {"tool_name": "b", "arguments": {}},
                ],
            },
            {"reasoning": "Done", "done": True, "final_answer": "ok"},
        ])
        engine = ToolEngine(config=self._make_config(), tools=tools, llm=llm)
        result = engine.gather(task="test")

        self.assertTrue(result.success)
        self.assertEqual(len(result.tool_calls), 2)

    # -- Truncation --

    def test_tool_result_truncation(self):
        """Tool returns very long result, gets truncated."""
        long_result = "x" * 500
        tool = MockTool("verbose", result=long_result)
        llm = MockLLMForTools([
            {
                "reasoning": "Call",
                "done": False,
                "tool_calls": [{"tool_name": "verbose", "arguments": {}}],
            },
            {"reasoning": "Done", "done": True, "final_answer": "ok"},
        ])
        engine = ToolEngine(
            config=self._make_config(max_result_chars=100),
            tools=[tool],
            llm=llm,
        )
        result = engine.gather(task="test")

        self.assertEqual(len(result.tool_calls), 1)
        self.assertIn("[... truncated ...]", result.tool_calls[0].result)
        self.assertTrue(len(result.tool_calls[0].result) < 200)

    # -- ToolEngineResult helpers --

    def test_tool_results_text_formatting(self):
        """ToolEngineResult.tool_results_text() formats correctly."""
        result = ToolEngineResult(
            summary="done",
            tool_calls=[
                ToolCall(tool_name="a", arguments={}, result="output_a", turn=0),
                ToolCall(tool_name="b", arguments={}, result="", error="failed", turn=0),
            ],
            turns_used=1,
            success=True,
        )
        text = result.tool_results_text()
        self.assertIn("### a\noutput_a", text)
        self.assertIn("### b (error)\nfailed", text)

    def test_tool_results_text_truncation(self):
        """tool_results_text() respects max_chars."""
        long_result = "y" * 10000
        result = ToolEngineResult(
            summary="done",
            tool_calls=[ToolCall(tool_name="big", arguments={}, result=long_result, turn=0)],
            turns_used=1,
            success=True,
        )
        text = result.tool_results_text(max_chars=200)
        self.assertIn("[... truncated ...]", text)
        self.assertTrue(len(text) < 300)

    def test_has_tool_output_property(self):
        """has_tool_output is True only when at least one call succeeded."""
        empty = ToolEngineResult(summary="x")
        self.assertFalse(empty.has_tool_output)

        only_errors = ToolEngineResult(
            summary="x",
            tool_calls=[ToolCall(tool_name="t", arguments={}, result="", error="err", turn=0)],
        )
        self.assertFalse(only_errors.has_tool_output)

        with_success = ToolEngineResult(
            summary="x",
            tool_calls=[ToolCall(tool_name="t", arguments={}, result="ok", turn=0)],
        )
        self.assertTrue(with_success.has_tool_output)

    # -- Reasoning fallback --

    def test_gather_uses_reasoning_when_no_final_answer(self):
        """When done=True but no final_answer, use reasoning as summary."""
        llm = MockLLMForTools([
            {"reasoning": "Fallback reasoning text", "done": True},
        ])
        engine = ToolEngine(config=self._make_config(), tools=[MockTool("t")], llm=llm)
        result = engine.gather(task="test")

        self.assertTrue(result.success)
        self.assertEqual(result.summary, "Fallback reasoning text")

    # -- LLM receives tool results in context --

    def test_llm_receives_prior_tool_results(self):
        """Verify that turn 1's input_text includes turn 0's tool results."""
        tool = MockTool("scanner", result="found_data_xyz")
        llm = MockLLMForTools([
            {
                "reasoning": "Call tool",
                "done": False,
                "tool_calls": [{"tool_name": "scanner", "arguments": {"q": "crash"}}],
            },
            {"reasoning": "Done", "done": True, "final_answer": "ok"},
        ])
        engine = ToolEngine(config=self._make_config(), tools=[tool], llm=llm)
        engine.gather(task="test")

        # The second LLM call should have the tool result in its input
        self.assertEqual(len(llm.calls), 2)
        second_input = llm.calls[1]["input_text"]
        self.assertIn("found_data_xyz", second_input)
        self.assertIn("scanner", second_input)


if __name__ == "__main__":
    unittest.main()
