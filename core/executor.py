"""Multi-turn Claude Code executor for AgentSwarm task execution.

Unlike ClaudeCodeLLMClient (single-turn text generation), this client spawns
Claude Code with full tool access (Edit, Read, Bash, Grep, etc.) so it can
perform real file operations, investigations, and code changes.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from time import perf_counter
from typing import Any

from core.graph_logging import log_llm_prompt_event, log_llm_response_event
from core.llm import LLMError, _extract_claude_result, _retry_with_backoff


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
        max_turns: int | None = None,
    ) -> ExecutionResult:
        """Execute a task using Claude Code with full tool access.

        Unlike generate_text(), this allows multi-turn tool use (Edit, Read, Bash, etc.).

        Args:
            max_turns: Override config.max_turns for this call. Useful for reducing
                       turn budget on later investigation rounds where context is
                       already established.
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

        effective_max_turns = max_turns if max_turns is not None else self.config.max_turns

        command = [
            resolved_command,
            "-p", "-",
            "--output-format", "json",
            "--model", self.config.model,
            "--max-turns", str(effective_max_turns),
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
            # On Windows, create a new process group so we can kill the entire
            # child tree on timeout instead of leaving orphaned claude.exe processes.
            popen_kwargs: dict[str, Any] = {}
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            proc = subprocess.Popen(
                command,
                cwd=effective_cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **popen_kwargs,
            )
            try:
                stdout, stderr = proc.communicate(
                    input=full_prompt,
                    timeout=self.config.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                # Kill the entire process tree, not just the parent.
                _kill_process_tree(proc)
                proc.wait(timeout=10)
                raise LLMError(
                    f"Claude Code executor timed out after {self.config.timeout_seconds} seconds"
                )

            completed = subprocess.CompletedProcess(
                args=command,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )

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

        def _is_retryable_executor_error(exc: Exception) -> bool:
            """Executor timeouts must NOT be retried — they mean the task is too
            complex for the time budget, not a transient network failure."""
            if not isinstance(exc, LLMError):
                return False
            msg = str(exc).lower()
            if "timed out" in msg or "timeout" in msg:
                return False
            retryable_signals = ("429", "500", "502", "503", "504", "connection")
            return any(s in msg for s in retryable_signals)

        try:
            completed = _retry_with_backoff(_do_execute, retryable_check=_is_retryable_executor_error)
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


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a subprocess and all its children.

    On Windows, ``taskkill /T /F`` kills the entire process tree.
    On POSIX, send SIGTERM to the process group.
    """
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=15,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        # Fallback: kill just the parent.
        try:
            proc.kill()
        except OSError:
            pass


def _extract_executor_result(raw_output: str) -> str:
    """Parse Claude Code JSON/JSONL output to extract the final result text.

    Delegates to ``_extract_claude_result`` which handles single JSON objects,
    JSON arrays, and JSONL streams (one JSON event per line).  The executor
    uses ``--output-format json --verbose`` which produces JSONL.
    """
    return _extract_claude_result(raw_output)
