from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from core.graph_logging import GRAPH_TIMELINE_FILE, LLM_PROMPT_TRACE_FILE, bind_active_trace_context
from core.llm import ClaudeCodeConfig, ClaudeCodeLLMClient, CodexCLIConfig, CodexCliLLMClient, ensure_traced_llm_client


class CodexCliLLMClientTests(unittest.TestCase):
    def test_generate_text_uses_resolved_executable_path(self) -> None:
        client = CodexCliLLMClient(
            CodexCLIConfig(
                command="codex",
                model="gpt-5.3-codex",
                timeout_seconds=30,
            )
        )
        resolved_command = r"C:\Users\phuong.le\AppData\Roaming\npm\codex.cmd"
        captured: list[list[str]] = []

        def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
            del kwargs
            captured.append(command)
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_text("resolved command works", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with mock.patch("core.llm.shutil.which", return_value=resolved_command):
            with mock.patch("core.llm.subprocess.run", side_effect=fake_run):
                result = client.generate_text(instructions="Test", input_text="Hello")

        self.assertEqual(result, "resolved command works")
        self.assertEqual(captured[0][0], resolved_command)

    def test_generate_text_runs_in_configured_working_directory_and_uses_stdin_prompt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="codex-llm-host-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            host_root.mkdir(parents=True, exist_ok=True)
            client = CodexCliLLMClient(
                CodexCLIConfig(
                    command="codex",
                    model="gpt-5.3-codex",
                    timeout_seconds=30,
                    working_directory=str(host_root),
                )
            )
            resolved_command = r"C:\Users\phuong.le\AppData\Roaming\npm\codex.cmd"
            captured: dict[str, object] = {}

            def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
                captured["command"] = command
                captured["cwd"] = kwargs["cwd"]
                captured["input"] = kwargs["input"]
                captured["encoding"] = kwargs["encoding"]
                captured["errors"] = kwargs["errors"]
                output_path = Path(command[command.index("-o") + 1])
                output_path.write_text("host context works", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with mock.patch("core.llm.shutil.which", return_value=resolved_command):
                with mock.patch("core.llm.subprocess.run", side_effect=fake_run):
                    result = client.generate_text(instructions="Plan", input_text="Fix the movement bug")

        self.assertEqual(result, "host context works")
        self.assertEqual(captured["cwd"], str(host_root))
        self.assertIn("--cd", captured["command"])
        command = captured["command"]
        assert isinstance(command, list)
        self.assertEqual(command[command.index("--cd") + 1], str(host_root))
        self.assertEqual(command[-1], "-")
        self.assertEqual(command[command.index("--sandbox") + 1], "read-only")
        self.assertIsInstance(captured["input"], str)
        self.assertIn("Fix the movement bug", str(captured["input"]))
        self.assertEqual(captured["encoding"], "utf-8")
        self.assertEqual(captured["errors"], "replace")

    def test_with_overrides_can_switch_codex_sandbox_mode(self) -> None:
        client = CodexCliLLMClient(
            CodexCLIConfig(
                command="codex",
                model="gpt-5.3-codex",
                timeout_seconds=30,
                sandbox_mode="read-only",
            )
        )
        resolved_command = r"C:\Users\phuong.le\AppData\Roaming\npm\codex.cmd"
        captured: list[list[str]] = []

        def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
            del kwargs
            captured.append(command)
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_text("sandbox override works", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with mock.patch("core.llm.shutil.which", return_value=resolved_command):
            with mock.patch("core.llm.subprocess.run", side_effect=fake_run):
                result = client.with_overrides(sandbox_mode="workspace-write").generate_text(
                    instructions="Investigate",
                    input_text="Find the root cause",
                )

        self.assertEqual(result, "sandbox override works")
        self.assertEqual(captured[0][captured[0].index("--sandbox") + 1], "workspace-write")

    def test_generate_json_uses_natural_language_prompt_wrapper(self) -> None:
        client = CodexCliLLMClient(
            CodexCLIConfig(
                command="codex",
                model="gpt-5.3-codex",
                timeout_seconds=30,
            )
        )
        resolved_command = r"C:\Users\phuong.le\AppData\Roaming\npm\codex.cmd"
        captured: dict[str, object] = {}

        def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
            captured["command"] = command
            captured["input"] = kwargs["input"]
            output_path = Path(command[command.index("-o") + 1])
            output_path.write_text('{"value":"ok"}', encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        with mock.patch("core.llm.shutil.which", return_value=resolved_command):
            with mock.patch("core.llm.subprocess.run", side_effect=fake_run):
                result = client.generate_json(
                    instructions="Classify the gameplay request and keep the response concise.",
                    input_text="The player cannot move after spawning.",
                    schema_name="tiny",
                    schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                )

        self.assertEqual(result, {"value": "ok"})
        prompt = str(captured["input"])
        self.assertNotIn("System instructions:", prompt)
        self.assertNotIn("Task input:", prompt)
        self.assertIn("Here is the current working context:", prompt)
        self.assertIn("configured structured output channel", prompt)

    def test_generate_json_writes_prompt_trace_when_graph_context_is_bound(self) -> None:
        client = CodexCliLLMClient(
            CodexCLIConfig(
                command="codex",
                model="gpt-5.3-codex",
                timeout_seconds=30,
            )
        )
        resolved_command = r"C:\Users\phuong.le\AppData\Roaming\npm\codex.cmd"

        with tempfile.TemporaryDirectory(prefix="codex-llm-trace-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "run_dir": str(run_dir),
                "task_id": "task-1",
                "active_task": {
                    "id": "task-1",
                    "workflow_name": "agentswarm::gameplay-engineer-workflow",
                },
            }

            def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
                del kwargs
                output_path = Path(command[command.index("-o") + 1])
                output_path.write_text('{"value":"ok"}', encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with mock.patch("core.llm.shutil.which", return_value=resolved_command):
                with mock.patch("core.llm.subprocess.run", side_effect=fake_run):
                    with bind_active_trace_context(state=state, graph_name="main_graph", node_name="route_tasks"):
                        result = client.generate_json(
                            instructions="Classify the gameplay request and keep the response concise.",
                            input_text="The player cannot move after spawning.",
                            schema_name="tiny",
                            schema={
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                                "required": ["value"],
                                "additionalProperties": False,
                            },
                        )

            self.assertEqual(result, {"value": "ok"})
            prompt_trace = (run_dir / LLM_PROMPT_TRACE_FILE).read_text(encoding="utf-8")
            self.assertIn("# LLM Prompt Trace", prompt_trace)
            self.assertIn("`main_graph.route_tasks`", prompt_trace)
            self.assertIn("### Instructions", prompt_trace)
            self.assertIn("### Input Text", prompt_trace)
            self.assertIn("### Final Prompt", prompt_trace)
            self.assertIn("### Model Response", prompt_trace)
            self.assertIn("Classify the gameplay request and keep the response concise.", prompt_trace)
            self.assertIn("The player cannot move after spawning.", prompt_trace)
            self.assertIn('{"value":"ok"}', prompt_trace)

            timeline_trace = (run_dir / GRAPH_TIMELINE_FILE).read_text(encoding="utf-8")
            self.assertIn("# Graph Timeline", timeline_trace)
            self.assertIn("`main_graph.route_tasks`", timeline_trace)
            self.assertIn("`PROMPT`", timeline_trace)
            self.assertIn("`RESPONSE`", timeline_trace)
            self.assertIn(f"details={LLM_PROMPT_TRACE_FILE}#", timeline_trace)

    def test_wrapped_fake_client_writes_prompt_and_response_trace_when_graph_context_is_bound(self) -> None:
        class FakeClient:
            def is_enabled(self) -> bool:
                return True

            def describe(self) -> str:
                return "wrapped fake client"

            def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
                raise AssertionError("This test should exercise generate_json only")

            def generate_json(
                self,
                *,
                instructions: str,
                input_text: str,
                schema_name: str,
                schema: dict,
                effort: str | None = None,
            ) -> dict:
                del instructions, input_text, schema_name, schema, effort
                return {"value": "wrapped"}

        with tempfile.TemporaryDirectory(prefix="codex-llm-wrapped-trace-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "run_dir": str(run_dir),
                "task_id": "task-2",
            }

            with bind_active_trace_context(state=state, graph_name="workflow_graph", node_name="review"):
                result = ensure_traced_llm_client(FakeClient()).generate_json(
                    instructions="Return a tiny structured review summary.",
                    input_text="Review the movement recovery request.",
                    schema_name="tiny",
                    schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                )
            prompt_trace = (run_dir / LLM_PROMPT_TRACE_FILE).read_text(encoding="utf-8")
            timeline_trace = (run_dir / GRAPH_TIMELINE_FILE).read_text(encoding="utf-8")

        self.assertEqual(result, {"value": "wrapped"})
        self.assertIn("wrapped fake client", prompt_trace)
        self.assertIn("### Final Prompt", prompt_trace)
        self.assertIn("### Model Response", prompt_trace)
        self.assertIn('"value": "wrapped"', prompt_trace)

        self.assertIn("`workflow_graph.review`", timeline_trace)
        self.assertIn("`PROMPT`", timeline_trace)
        self.assertIn("`RESPONSE`", timeline_trace)


class ClaudeCodeLLMClientTests(unittest.TestCase):
    def test_generate_text_extracts_result_from_json_envelope(self) -> None:
        client = ClaudeCodeLLMClient(
            ClaudeCodeConfig(
                command="claude",
                model="claude-sonnet-4-6",
                timeout_seconds=30,
            )
        )
        resolved_command = r"C:\Users\phuong.le\.claude\local\claude.cmd"
        captured: list[dict[str, object]] = []

        def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
            captured.append({"command": command, "input": kwargs.get("input")})
            import json
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "The movement bug is caused by a missing state reset.",
                    "session_id": "test-session",
                    "cost_usd": 0.001,
                    "duration_ms": 500,
                }),
                stderr="",
            )

        with mock.patch("core.llm.shutil.which", return_value=resolved_command):
            with mock.patch("core.llm.subprocess.run", side_effect=fake_run):
                result = client.generate_text(instructions="Analyze the bug", input_text="Player cannot move")

        self.assertEqual(result, "The movement bug is caused by a missing state reset.")
        command = captured[0]["command"]
        assert isinstance(command, list)
        self.assertEqual(command[0], resolved_command)
        self.assertIn("-p", command)
        self.assertIn("--output-format", command)
        self.assertEqual(command[command.index("--output-format") + 1], "json")
        self.assertIn("--model", command)
        self.assertEqual(command[command.index("--model") + 1], "claude-sonnet-4-6")
        self.assertIn("Player cannot move", str(captured[0]["input"]))

    def test_generate_json_parses_structured_output(self) -> None:
        client = ClaudeCodeLLMClient(
            ClaudeCodeConfig(
                command="claude",
                model="claude-sonnet-4-6",
                timeout_seconds=30,
            )
        )
        resolved_command = r"C:\Users\phuong.le\.claude\local\claude.cmd"
        captured: list[dict[str, object]] = []

        def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
            captured.append({"command": command, "input": kwargs.get("input")})
            import json
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": '{"task_type": "bugfix", "priority": "high"}',
                    "session_id": "test-session",
                }),
                stderr="",
            )

        with mock.patch("core.llm.shutil.which", return_value=resolved_command):
            with mock.patch("core.llm.subprocess.run", side_effect=fake_run):
                result = client.generate_json(
                    instructions="Classify the task",
                    input_text="Fix the dodge cancel",
                    schema_name="task_classification",
                    schema={
                        "type": "object",
                        "properties": {"task_type": {"type": "string"}, "priority": {"type": "string"}},
                        "required": ["task_type", "priority"],
                        "additionalProperties": False,
                    },
                )

        self.assertEqual(result, {"task_type": "bugfix", "priority": "high"})
        prompt = str(captured[0]["input"])
        self.assertIn("matching this schema", prompt)
        self.assertIn("task_type", prompt)

    def test_generate_text_raises_on_claude_error_response(self) -> None:
        client = ClaudeCodeLLMClient(
            ClaudeCodeConfig(
                command="claude",
                model="claude-sonnet-4-6",
                timeout_seconds=30,
            )
        )
        resolved_command = r"C:\Users\phuong.le\.claude\local\claude.cmd"

        def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
            import json
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({
                    "type": "result",
                    "subtype": "error",
                    "is_error": True,
                    "result": "Rate limited",
                }),
                stderr="",
            )

        with mock.patch("core.llm.shutil.which", return_value=resolved_command):
            with mock.patch("core.llm.subprocess.run", side_effect=fake_run):
                with self.assertRaises(Exception) as ctx:
                    client.generate_text(instructions="Test", input_text="Hello")
                self.assertIn("Rate limited", str(ctx.exception))

    def test_with_overrides_changes_max_turns(self) -> None:
        client = ClaudeCodeLLMClient(
            ClaudeCodeConfig(
                command="claude",
                model="claude-sonnet-4-6",
                timeout_seconds=30,
                max_turns=1,
            )
        )
        resolved_command = r"C:\Users\phuong.le\.claude\local\claude.cmd"
        captured: list[list[str]] = []

        def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
            captured.append(command)
            import json
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "ok"}),
                stderr="",
            )

        with mock.patch("core.llm.shutil.which", return_value=resolved_command):
            with mock.patch("core.llm.subprocess.run", side_effect=fake_run):
                client.with_overrides(max_turns=5).generate_text(instructions="Test", input_text="Hello")

        command = captured[0]
        self.assertEqual(command[command.index("--max-turns") + 1], "5")


if __name__ == "__main__":
    unittest.main()
