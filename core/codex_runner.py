from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import subprocess


@dataclass(frozen=True)
class CodexRunConfig:
    command: str = "codex"
    profile: str = "ai-gateway"
    model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    sandbox: str = "danger-full-access"
    approval_policy: str = "never"
    timeout_seconds: int = 1800
    working_directory: str = ""


@dataclass(frozen=True)
class CodexRunResult:
    success: bool
    exit_code: int
    output_text: str
    stdout: str
    stderr: str
    command: list[str]
    output_path: str
    stdout_path: str
    stderr_path: str


def build_codex_command(config: CodexRunConfig, output_path: Path) -> list[str]:
    resolved = shutil.which(config.command) or config.command
    command = [
        resolved,
        "--profile",
        config.profile,
        "--model",
        config.model,
        "--config",
        f'model_reasoning_effort="{config.reasoning_effort}"',
        "--sandbox",
        config.sandbox,
        "--ask-for-approval",
        config.approval_policy,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
        "--output-last-message",
        str(output_path),
    ]
    if config.working_directory:
        command.extend(["--cd", config.working_directory])
    command.append("-")
    return command


def run_codex_prompt(prompt: str, config: CodexRunConfig, artifact_dir: Path) -> CodexRunResult:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifact_dir / "codex_last_message.txt"
    stdout_path = artifact_dir / "codex_stdout.txt"
    stderr_path = artifact_dir / "codex_stderr.txt"
    command_path = artifact_dir / "codex_command.json"
    command = build_codex_command(config, output_path)
    command_path.write_text(json.dumps(command, indent=2), encoding="utf-8")

    completed = subprocess.run(
        command,
        cwd=config.working_directory or None,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=config.timeout_seconds,
        check=False,
    )
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    output_text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else ""

    return CodexRunResult(
        success=completed.returncode == 0,
        exit_code=completed.returncode,
        output_text=output_text,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        command=command,
        output_path=str(output_path),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )
