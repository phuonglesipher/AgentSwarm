from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from unittest import mock

from core.executor import (
    ClaudeCodeExecutorClient,
    ClaudeCodeExecutorConfig,
    ExecutionResult,
    build_executor_system_prompt,
    build_executor_task_prompt,
    _ANTI_RECURSION_GUARD,
)


class BuildExecutorSystemPromptTests(unittest.TestCase):
    def test_anti_recursion_guard_present(self) -> None:
        prompt = build_executor_system_prompt()
        self.assertIn("Do NOT run `python main.py`", prompt)
        self.assertIn("Do NOT trigger any workflow pipeline", prompt)

    def test_working_directory_included(self) -> None:
        prompt = build_executor_system_prompt(working_directory="/project/root")
        self.assertIn("/project/root", prompt)

    def test_scope_constraints_included(self) -> None:
        prompt = build_executor_system_prompt(
            scope_constraints=["Only modify src/", "Do not touch tests/"]
        )
        self.assertIn("Only modify src/", prompt)
        self.assertIn("Do not touch tests/", prompt)


class BuildExecutorTaskPromptTests(unittest.TestCase):
    def test_basic_task(self) -> None:
        prompt = build_executor_task_prompt(description="Fix the dodge cancel bug")
        self.assertIn("Fix the dodge cancel bug", prompt)
        self.assertIn("## Task", prompt)

    def test_with_feedback(self) -> None:
        prompt = build_executor_task_prompt(
            description="Fix bug",
            prior_feedback="Score 70/100. Missing root cause proof.",
        )
        self.assertIn("## Prior Review Feedback", prompt)
        self.assertIn("Missing root cause proof", prompt)

    def test_without_feedback(self) -> None:
        prompt = build_executor_task_prompt(description="Fix bug", prior_feedback=None)
        self.assertNotIn("Prior Review Feedback", prompt)

    def test_with_context(self) -> None:
        prompt = build_executor_task_prompt(
            description="Fix bug",
            context="The component is USipherCombatComponent.",
        )
        self.assertIn("## Context", prompt)
        self.assertIn("USipherCombatComponent", prompt)


class ClaudeCodeExecutorClientTests(unittest.TestCase):
    def _make_client(self, **overrides) -> ClaudeCodeExecutorClient:
        defaults = {
            "command": "claude",
            "model": "claude-sonnet-4-6",
            "timeout_seconds": 60,
            "max_turns": 5,
        }
        defaults.update(overrides)
        return ClaudeCodeExecutorClient(ClaudeCodeExecutorConfig(**defaults))

    def test_describe_when_command_not_found(self) -> None:
        client = self._make_client()
        with mock.patch("core.executor.shutil.which", return_value=None):
            self.assertIn("disabled", client.describe())
            self.assertFalse(client.is_enabled())

    def test_describe_when_available(self) -> None:
        client = self._make_client()
        with mock.patch("core.executor.shutil.which", return_value="/usr/bin/claude"):
            self.assertIn("available", client.describe())
            self.assertTrue(client.is_enabled())

    def test_execute_task_returns_result_on_success(self) -> None:
        client = self._make_client()
        json_output = json.dumps({"result": "Files updated successfully.", "is_error": False})

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout=json_output, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("core.executor.shutil.which", return_value="/usr/bin/claude"):
                with mock.patch("core.executor.subprocess.run", side_effect=fake_run):
                    result = client.execute_task(
                        task_prompt="Fix the bug",
                        system_prompt="You are an executor.",
                        working_directory=tmp,
                    )

        self.assertIsInstance(result, ExecutionResult)
        self.assertTrue(result.success)
        self.assertEqual(result.result_text, "Files updated successfully.")
        self.assertIsNone(result.error)

    def test_execute_task_passes_max_turns(self) -> None:
        client = self._make_client(max_turns=10)
        captured_commands: list[list[str]] = []

        def fake_run(command, **kwargs):
            captured_commands.append(command)
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps({"result": "done"}), stderr=""
            )

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("core.executor.shutil.which", return_value="/usr/bin/claude"):
                with mock.patch("core.executor.subprocess.run", side_effect=fake_run):
                    client.execute_task(task_prompt="test", working_directory=tmp)

        command = captured_commands[0]
        turns_idx = command.index("--max-turns")
        self.assertEqual(command[turns_idx + 1], "10")

    def test_execute_task_returns_error_on_failure(self) -> None:
        client = self._make_client()

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="Some error")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("core.executor.shutil.which", return_value="/usr/bin/claude"):
                with mock.patch("core.executor.subprocess.run", side_effect=fake_run):
                    result = client.execute_task(
                        task_prompt="Fix the bug",
                        working_directory=tmp,
                    )

        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    def test_execute_task_returns_error_when_disabled(self) -> None:
        client = self._make_client()
        with mock.patch("core.executor.shutil.which", return_value=None):
            result = client.execute_task(task_prompt="Fix the bug")

        self.assertFalse(result.success)
        self.assertIn("disabled", result.error)

    def test_execute_task_handles_timeout(self) -> None:
        client = self._make_client(timeout_seconds=1)

        def fake_run(command, **kwargs):
            raise subprocess.TimeoutExpired(command, 1)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("core.executor.shutil.which", return_value="/usr/bin/claude"):
                with mock.patch("core.executor.subprocess.run", side_effect=fake_run):
                    result = client.execute_task(
                        task_prompt="Fix the bug",
                        working_directory=tmp,
                    )

        self.assertFalse(result.success)
        self.assertIn("timed out", result.error)

    def test_execute_task_merges_system_prompt_into_input(self) -> None:
        client = self._make_client()
        captured_inputs: list[str] = []

        def fake_run(command, **kwargs):
            captured_inputs.append(kwargs.get("input", ""))
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps({"result": "ok"}), stderr=""
            )

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("core.executor.shutil.which", return_value="/usr/bin/claude"):
                with mock.patch("core.executor.subprocess.run", side_effect=fake_run):
                    client.execute_task(
                        task_prompt="Fix the bug",
                        system_prompt="System instructions here",
                        working_directory=tmp,
                    )

        self.assertIn("System instructions here", captured_inputs[0])
        self.assertIn("Fix the bug", captured_inputs[0])

    def test_execute_task_detects_auth_error(self) -> None:
        client = self._make_client()

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="401 Unauthorized")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("core.executor.shutil.which", return_value="/usr/bin/claude"):
                with mock.patch("core.executor.subprocess.run", side_effect=fake_run):
                    result = client.execute_task(
                        task_prompt="Fix the bug",
                        working_directory=tmp,
                    )

        self.assertFalse(result.success)
        # After auth error, client should be disabled
        self.assertFalse(client.is_enabled())


class AntiRecursionGuardTests(unittest.TestCase):
    def test_guard_mentions_main_py(self) -> None:
        self.assertIn("main.py", _ANTI_RECURSION_GUARD)

    def test_guard_mentions_claude_bridge(self) -> None:
        self.assertIn("claude_bridge.py", _ANTI_RECURSION_GUARD)

    def test_system_prompt_includes_guard(self) -> None:
        prompt = build_executor_system_prompt()
        self.assertIn(_ANTI_RECURSION_GUARD, prompt)


if __name__ == "__main__":
    unittest.main()
