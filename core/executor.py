"""Multi-turn Claude Code executor for AgentSwarm task execution.

Unlike ClaudeCodeLLMClient (single-turn text generation), this client spawns
Claude Code with full tool access (Edit, Read, Bash, Grep, etc.) so it can
perform real file operations, investigations, and code changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from typing import Any

from core.graph_logging import log_llm_prompt_event, log_llm_response_event
from core.llm import LLMError, _retry_with_backoff


@dataclass(frozen=True)
class ClaudeCodeExecutorConfig:
    command: str
    model: str
    timeout_seconds: int
    working_directory: str | None = None
    permission_mode: str = "auto"
    max_turns: int = 25


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    result_text: str
    session_id: str | None = None
    cost_usd: float | None = None
    error: str | None = None


_ANTI_RECURSION_GUARD = (
    "CRITICAL CONSTRAINT: You are a task executor inside AgentSwarm. "
    "Do NOT run `python main.py`, `python claude_bridge.py`, or invoke any AgentSwarm orchestration. "
    "Do NOT trigger any workflow pipeline. Focus solely on the specific task below."
)


def build_executor_system_prompt(
    *,
    working_directory: str | None = None,
    scope_constraints: list[str] | None = None,
) -> str:
    """Build a system prompt that scopes the executor and prevents recursion."""
    sections = [_ANTI_RECURSION_GUARD]

    if working_directory:
        sections.append(f"Working directory: {working_directory}")

    if scope_constraints:
        sections.append("Scope constraints:")
        sections.extend(f"- {c}" for c in scope_constraints)

    sections.append(
        "After making changes, verify your work is correct (e.g. check for syntax errors, "
        "run relevant tests if available). Return a concise summary of what you did."
    )
    return "\n\n".join(sections)


def build_executor_task_prompt(
    *,
    description: str,
    prior_feedback: str | None = None,
    context: str | None = None,
) -> str:
    """Build the task prompt, optionally including reviewer feedback from a prior loop."""
    parts = [f"## Task\n\n{description}"]

    if context:
        parts.append(f"## Context\n\n{context}")

    if prior_feedback:
        parts.append(
            f"## Prior Review Feedback\n\n"
            f"Address the following feedback from the previous review round:\n\n{prior_feedback}"
        )

    return "\n\n".join(parts)


class ClaudeCodeExecutorClient:
    """Multi-turn Claude Code subprocess with full tool access."""

    def __init__(self, config: ClaudeCodeExecutorConfig) -> None:
        self.config = config
        self._disabled_reason: str | None = None

    def _resolve_command_path(self) -> str | None:
        resolved = shutil.which(self.config.command)
        if resolved:
            return resolved
        candidate = Path(self.config.command)
        if candidate.exists():
            return str(candidate)
        return None

    def is_enabled(self) -> bool:
        return self._disabled_reason is None and self._resolve_command_path() is not None

    def describe(self) -> str:
        if self._disabled_reason:
            return f"claude_code_executor/{self.config.model}: disabled ({self._disabled_reason})"
        if self._resolve_command_path() is None:
            return f"claude_code_executor/{self.config.model}: disabled (claude command not found)"
        return f"claude_code_executor/{self.config.model}: available"

    def execute_task(
        self,
        *,
        task_prompt: str,
        system_prompt: str | None = None,
        working_directory: str | None = None,
    ) -> ExecutionResult:
        """Execute a task using Claude Code with full tool access.

        Unlike generate_text(), this allows multi-turn tool use (Edit, Read, Bash, etc.).
        """
        resolved_command = self._resolve_command_path()
        if not resolved_command or not self.is_enabled():
            return ExecutionResult(
                success=False,
                result_text="",
                error=self.describe(),
            )

        effective_cwd = working_directory or self.config.working_directory or os.getcwd()
        effective_cwd = str(Path(effective_cwd).resolve())

        full_prompt = task_prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{full_prompt}"

        command = [
            resolved_command,
            "-p", "-",
            "--output-format", "json",
            "--model", self.config.model,
            "--max-turns", str(self.config.max_turns),
            "--verbose",
        ]

        client_label = f"claude_code_executor/{self.config.model}"
        request_event_id = log_llm_prompt_event(
            client_label=client_label,
            transport="claude_code_executor_subprocess",
            instructions=system_prompt or "",
            input_text=task_prompt,
            final_prompt=full_prompt,
            require_structured_output=False,
        )
        start_time = perf_counter()
        result_text = ""

        def _do_execute():
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    cwd=effective_cwd,
                    capture_output=True,
                    input=full_prompt,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.config.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMError(
                    f"Claude Code executor timed out after {self.config.timeout_seconds} seconds"
                ) from exc

            if completed.returncode != 0:
                combined = "\n".join(
                    part.strip()
                    for part in [completed.stdout, completed.stderr]
                    if part and part.strip()
                )
                normalized = combined.lower()
                if "unauthorized" in normalized or "not authenticated" in normalized or "api key" in normalized:
                    self._disabled_reason = "claude auth required"
                raise LLMError(
                    f"Claude Code executor failed with exit code {completed.returncode}: {combined}"
                )
            return completed

        try:
            completed = _retry_with_backoff(_do_execute)
            raw_output = completed.stdout.strip()
            if not raw_output:
                raise LLMError("Claude Code executor returned empty output")

            result_text = _extract_executor_result(raw_output)
            elapsed_ms = round((perf_counter() - start_time) * 1000, 2)
            log_llm_response_event(
                client_label=client_label,
                transport="claude_code_executor_subprocess",
                require_structured_output=False,
                response_text=result_text,
                elapsed_ms=elapsed_ms,
                request_event_id=request_event_id,
            )
            return ExecutionResult(success=True, result_text=result_text)

        except (LLMError, Exception) as exc:
            elapsed_ms = round((perf_counter() - start_time) * 1000, 2)
            log_llm_response_event(
                client_label=client_label,
                transport="claude_code_executor_subprocess",
                require_structured_output=False,
                response_text=result_text,
                elapsed_ms=elapsed_ms,
                request_event_id=request_event_id,
                error=str(exc),
            )
            return ExecutionResult(success=False, result_text=result_text, error=str(exc))


def _extract_executor_result(raw_output: str) -> str:
    """Parse Claude Code JSON output to extract result text."""
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output

    if isinstance(data, dict):
        if data.get("is_error"):
            raise LLMError(f"Claude Code executor returned error: {data.get('result', 'unknown error')}")
        result = data.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip()

    return raw_output
