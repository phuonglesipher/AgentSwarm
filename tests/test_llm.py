from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from core.llm import CodexCLIConfig, CodexCliLLMClient


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


if __name__ == "__main__":
    unittest.main()
