from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from time import perf_counter
from typing import Any
from urllib import error, request

from core.graph_logging import log_llm_prompt_event, log_llm_response_event
from core.natural_language_prompts import build_llm_request


class LLMError(RuntimeError):
    """Raised when an LLM backend fails or returns unusable output."""


_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
_RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0"))
_RETRY_MAX_DELAY = 30.0


def _is_retryable_llm_error(exc: Exception) -> bool:
    if not isinstance(exc, LLMError):
        return False
    msg = str(exc).lower()
    retryable_signals = ("429", "500", "502", "503", "504", "timeout", "timed out", "connection")
    return any(signal in msg for signal in retryable_signals)


def _retry_with_backoff(fn, *, max_retries: int = _MAX_RETRIES, retryable_check=_is_retryable_llm_error):
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise
            if retryable_check and not retryable_check(exc):
                raise
            delay = min(_RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.5), _RETRY_MAX_DELAY)
            time.sleep(delay)
    raise last_exc  # unreachable


class LLMClient(ABC):
    @abstractmethod
    def is_enabled(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict[str, Any],
        effort: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class ResponsesLLMConfig:
    api_key: str | None
    model: str
    base_url: str
    timeout_seconds: int
    reasoning_effort: str


class ResponsesLLMClient(LLMClient):
    def __init__(self, config: ResponsesLLMConfig) -> None:
        self.config = config

    def is_enabled(self) -> bool:
        return bool(self.config.api_key)

    def describe(self) -> str:
        if not self.is_enabled():
            return f"responses_api/{self.config.model}: disabled (missing OPENAI_API_KEY)"
        return f"responses_api/{self.config.model}: enabled"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        resolved_effort = effort or self.config.reasoning_effort
        prompt_preview = _merge_prompt(instructions=instructions, input_text=input_text, require_json=False)
        request_event_id = log_llm_prompt_event(
            client_label=f"responses_api/{self.config.model}",
            transport="responses_api_instructions_input",
            instructions=instructions,
            input_text=input_text,
            final_prompt=prompt_preview,
            require_structured_output=False,
            effort=resolved_effort,
        )
        start_time = perf_counter()
        output_text = ""
        try:
            response = self._request(
                {
                    "instructions": instructions,
                    "input": input_text,
                    "reasoning": {"effort": resolved_effort},
                }
            )
            output_text = self._extract_output_text(response)
        except Exception as exc:
            log_llm_response_event(
                client_label=f"responses_api/{self.config.model}",
                transport="responses_api_instructions_input",
                require_structured_output=False,
                response_text=output_text,
                elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
                request_event_id=request_event_id,
                error=str(exc),
            )
            raise
        log_llm_response_event(
            client_label=f"responses_api/{self.config.model}",
            transport="responses_api_instructions_input",
            require_structured_output=False,
            response_text=output_text,
            elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
            request_event_id=request_event_id,
        )
        return output_text

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict[str, Any],
        effort: str | None = None,
    ) -> dict[str, Any]:
        resolved_effort = effort or self.config.reasoning_effort
        prompt_preview = _merge_prompt(instructions=instructions, input_text=input_text, require_json=True)
        request_event_id = log_llm_prompt_event(
            client_label=f"responses_api/{self.config.model}",
            transport="responses_api_instructions_input",
            instructions=instructions,
            input_text=input_text,
            final_prompt=prompt_preview,
            require_structured_output=True,
            schema_name=schema_name,
            effort=resolved_effort,
        )
        start_time = perf_counter()
        output_text = ""
        try:
            response = self._request(
                {
                    "instructions": instructions,
                    "input": input_text,
                    "reasoning": {"effort": resolved_effort},
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        }
                    },
                }
            )
            refusal = response.get("refusal")
            if refusal:
                raise LLMError(f"Model refused the request: {refusal}")
            output_text = self._extract_output_text(response)
            parsed = _parse_json_object(output_text)
        except Exception as exc:
            log_llm_response_event(
                client_label=f"responses_api/{self.config.model}",
                transport="responses_api_instructions_input",
                require_structured_output=True,
                response_text=output_text,
                elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
                schema_name=schema_name,
                request_event_id=request_event_id,
                error=str(exc),
            )
            raise
        log_llm_response_event(
            client_label=f"responses_api/{self.config.model}",
            transport="responses_api_instructions_input",
            require_structured_output=True,
            response_text=output_text,
            elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
            schema_name=schema_name,
            request_event_id=request_event_id,
        )
        return parsed

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.api_key:
            raise LLMError("OPENAI_API_KEY is not set")

        full_payload = {
            "model": self.config.model,
            "store": False,
            **payload,
        }
        encoded = json.dumps(full_payload).encode("utf-8")
        api_request = request.Request(
            url=f"{self.config.base_url.rstrip('/')}/responses",
            data=encoded,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        def _do_request():
            try:
                with request.urlopen(api_request, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise LLMError(f"OpenAI API returned {exc.code}: {body}") from exc
            except error.URLError as exc:
                raise LLMError(f"OpenAI API request failed: {exc}") from exc

        return _retry_with_backoff(_do_request)

    def _extract_output_text(self, response: dict[str, Any]) -> str:
        top_level = response.get("output_text")
        if isinstance(top_level, str) and top_level.strip():
            return top_level

        parts: list[str] = []
        for item in response.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        parts.append(str(content["text"]))
            elif item.get("type") == "output_text" and item.get("text"):
                parts.append(str(item["text"]))

        joined = "\n".join(part.strip() for part in parts if part.strip()).strip()
        if not joined:
            raise LLMError("Model response did not contain text output")
        return joined


@dataclass(frozen=True)
class CodexCLIConfig:
    command: str
    model: str
    timeout_seconds: int
    working_directory: str | None = None
    sandbox_mode: str = "read-only"


class CodexCliLLMClient(LLMClient):
    def __init__(self, config: CodexCLIConfig) -> None:
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
            return f"codex_cli/{self.config.model}: disabled ({self._disabled_reason})"
        if self._resolve_command_path() is None:
            return f"codex_cli/{self.config.model}: disabled (codex command not found)"
        return f"codex_cli/{self.config.model}: available"

    def with_overrides(
        self,
        *,
        sandbox_mode: str | None = None,
        working_directory: str | None = None,
        timeout_seconds: int | None = None,
    ) -> "CodexCliLLMClient":
        return CodexCliLLMClient(
            replace(
                self.config,
                sandbox_mode=sandbox_mode or self.config.sandbox_mode,
                working_directory=working_directory or self.config.working_directory,
                timeout_seconds=timeout_seconds or self.config.timeout_seconds,
            )
        )

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort
        prompt = _merge_prompt(instructions=instructions, input_text=input_text, require_json=False)
        request_event_id = log_llm_prompt_event(
            client_label=f"codex_cli/{self.config.model}",
            transport="codex_cli_stdin_prompt",
            instructions=instructions,
            input_text=input_text,
            final_prompt=prompt,
            require_structured_output=False,
        )
        start_time = perf_counter()
        output_text = ""
        try:
            output_text = self._run_codex(prompt=prompt, schema=None)
        except Exception as exc:
            log_llm_response_event(
                client_label=f"codex_cli/{self.config.model}",
                transport="codex_cli_stdin_prompt",
                require_structured_output=False,
                response_text=output_text,
                elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
                request_event_id=request_event_id,
                error=str(exc),
            )
            raise
        log_llm_response_event(
            client_label=f"codex_cli/{self.config.model}",
            transport="codex_cli_stdin_prompt",
            require_structured_output=False,
            response_text=output_text,
            elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
            request_event_id=request_event_id,
        )
        return output_text

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict[str, Any],
        effort: str | None = None,
    ) -> dict[str, Any]:
        del effort
        prompt = _merge_prompt(instructions=instructions, input_text=input_text, require_json=True)
        request_event_id = log_llm_prompt_event(
            client_label=f"codex_cli/{self.config.model}",
            transport="codex_cli_stdin_prompt",
            instructions=instructions,
            input_text=input_text,
            final_prompt=prompt,
            require_structured_output=True,
            schema_name=schema_name,
        )
        start_time = perf_counter()
        output = ""
        try:
            output = self._run_codex(prompt=prompt, schema={"name": schema_name, "schema": schema})
            parsed = _parse_json_object(output)
        except Exception as exc:
            log_llm_response_event(
                client_label=f"codex_cli/{self.config.model}",
                transport="codex_cli_stdin_prompt",
                require_structured_output=True,
                response_text=output,
                elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
                schema_name=schema_name,
                request_event_id=request_event_id,
                error=str(exc),
            )
            raise
        log_llm_response_event(
            client_label=f"codex_cli/{self.config.model}",
            transport="codex_cli_stdin_prompt",
            require_structured_output=True,
            response_text=output,
            elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
            schema_name=schema_name,
            request_event_id=request_event_id,
        )
        return parsed

    def _run_codex(self, *, prompt: str, schema: dict[str, Any] | None) -> str:
        resolved_command = self._resolve_command_path()
        if not resolved_command or not self.is_enabled():
            raise LLMError(self.describe())
        effective_cwd = (
            str(Path(self.config.working_directory).resolve()) if self.config.working_directory else os.getcwd()
        )

        with tempfile.TemporaryDirectory(prefix="codex-llm-") as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "last_message.txt"
            command = [
                resolved_command,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color",
                "never",
                "--sandbox",
                self.config.sandbox_mode,
                "-m",
                self.config.model,
                "-o",
                str(output_path),
                "--cd",
                effective_cwd,
            ]

            if schema is not None:
                schema_path = temp_path / "schema.json"
                schema_path.write_text(json.dumps(schema["schema"], indent=2), encoding="utf-8")
                command.extend(["--output-schema", str(schema_path)])

            command.append("-")

            def _do_codex_run():
                try:
                    completed = subprocess.run(
                        command,
                        check=False,
                        cwd=effective_cwd,
                        capture_output=True,
                        input=prompt,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=self.config.timeout_seconds,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise LLMError(f"Codex CLI timed out after {self.config.timeout_seconds} seconds") from exc

                if completed.returncode != 0:
                    combined = "\n".join(
                        part.strip() for part in [completed.stdout, completed.stderr] if part and part.strip()
                    )
                    if _looks_like_auth_error(combined):
                        self._disabled_reason = "codex login required"
                    raise LLMError(f"Codex CLI failed with exit code {completed.returncode}: {combined}")
                return completed

            completed = _retry_with_backoff(_do_codex_run)

            output = output_path.read_text(encoding="utf-8").strip()
            if not output:
                raise LLMError("Codex CLI returned an empty final message")
            return output


@dataclass(frozen=True)
class ClaudeCodeConfig:
    command: str
    model: str
    timeout_seconds: int
    working_directory: str | None = None
    max_turns: int = 1


class ClaudeCodeLLMClient(LLMClient):
    def __init__(self, config: ClaudeCodeConfig) -> None:
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
            return f"claude_code/{self.config.model}: disabled ({self._disabled_reason})"
        if self._resolve_command_path() is None:
            return f"claude_code/{self.config.model}: disabled (claude command not found)"
        return f"claude_code/{self.config.model}: available"

    def with_overrides(
        self,
        *,
        working_directory: str | None = None,
        timeout_seconds: int | None = None,
        max_turns: int | None = None,
    ) -> "ClaudeCodeLLMClient":
        return ClaudeCodeLLMClient(
            replace(
                self.config,
                working_directory=working_directory or self.config.working_directory,
                timeout_seconds=timeout_seconds or self.config.timeout_seconds,
                max_turns=max_turns if max_turns is not None else self.config.max_turns,
            )
        )

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort
        prompt = _merge_prompt(instructions=instructions, input_text=input_text, require_json=False)
        request_event_id = log_llm_prompt_event(
            client_label=f"claude_code/{self.config.model}",
            transport="claude_code_subprocess",
            instructions=instructions,
            input_text=input_text,
            final_prompt=prompt,
            require_structured_output=False,
        )
        start_time = perf_counter()
        output_text = ""
        try:
            output_text = self._run_claude(prompt=prompt, json_schema=None)
        except Exception as exc:
            log_llm_response_event(
                client_label=f"claude_code/{self.config.model}",
                transport="claude_code_subprocess",
                require_structured_output=False,
                response_text=output_text,
                elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
                request_event_id=request_event_id,
                error=str(exc),
            )
            raise
        log_llm_response_event(
            client_label=f"claude_code/{self.config.model}",
            transport="claude_code_subprocess",
            require_structured_output=False,
            response_text=output_text,
            elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
            request_event_id=request_event_id,
        )
        return output_text

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict[str, Any],
        effort: str | None = None,
    ) -> dict[str, Any]:
        del effort
        prompt = _merge_prompt(instructions=instructions, input_text=input_text, require_json=True)
        request_event_id = log_llm_prompt_event(
            client_label=f"claude_code/{self.config.model}",
            transport="claude_code_subprocess",
            instructions=instructions,
            input_text=input_text,
            final_prompt=prompt,
            require_structured_output=True,
            schema_name=schema_name,
        )
        start_time = perf_counter()
        output = ""
        try:
            output = self._run_claude(prompt=prompt, json_schema=schema)
            parsed = _parse_json_object(output)
        except Exception as exc:
            log_llm_response_event(
                client_label=f"claude_code/{self.config.model}",
                transport="claude_code_subprocess",
                require_structured_output=True,
                response_text=output,
                elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
                schema_name=schema_name,
                request_event_id=request_event_id,
                error=str(exc),
            )
            raise
        log_llm_response_event(
            client_label=f"claude_code/{self.config.model}",
            transport="claude_code_subprocess",
            require_structured_output=True,
            response_text=output,
            elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
            schema_name=schema_name,
            request_event_id=request_event_id,
        )
        return parsed

    def _run_claude(self, *, prompt: str, json_schema: dict[str, Any] | None) -> str:
        resolved_command = self._resolve_command_path()
        if not resolved_command or not self.is_enabled():
            raise LLMError(self.describe())

        command = [
            resolved_command,
            "-p", "-",
            "--output-format", "json",
            "--model", self.config.model,
            "--max-turns", str(self.config.max_turns),
            "--verbose",
        ]
        if json_schema is not None:
            schema_hint = (
                "\n\nYou MUST respond with ONLY a valid JSON object matching this schema "
                f"(no markdown fences, no explanation):\n{json.dumps(json_schema, indent=2)}"
            )
            prompt = prompt + schema_hint

        effective_cwd = (
            str(Path(self.config.working_directory).resolve()) if self.config.working_directory else os.getcwd()
        )

        def _do_claude_run():
            # Use Popen + communicate() instead of subprocess.run(capture_output=True)
            # to avoid pipe deadlock when --verbose produces large output.
            # communicate() reads stdout and stderr concurrently via threads.
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
                    input=prompt,
                    timeout=self.config.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                _kill_llm_process_tree(proc)
                proc.wait(timeout=10)
                raise LLMError(f"Claude Code timed out after {self.config.timeout_seconds} seconds")

            completed = subprocess.CompletedProcess(
                args=command,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )

            if completed.returncode != 0:
                combined = "\n".join(
                    part.strip() for part in [completed.stdout, completed.stderr] if part and part.strip()
                )
                if _looks_like_claude_auth_error(combined):
                    self._disabled_reason = "claude auth required"
                raise LLMError(f"Claude Code failed with exit code {completed.returncode}: {combined}")
            return completed

        completed = _retry_with_backoff(_do_claude_run)

        raw_output = completed.stdout.strip()
        if not raw_output:
            raise LLMError("Claude Code returned empty output")

        return _extract_claude_result(raw_output)


def _kill_llm_process_tree(proc: subprocess.Popen) -> None:
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
        try:
            proc.kill()
        except OSError:
            pass


def _looks_like_claude_auth_error(output: str) -> bool:
    normalized = output.lower()
    return "unauthorized" in normalized or "not authenticated" in normalized or "api key" in normalized


def _extract_claude_result(raw_output: str) -> str:
    # Try parsing as JSON first
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        data = None

    # Single JSON object (non-verbose output: {"result":"...", "is_error":false})
    if isinstance(data, dict):
        if data.get("is_error"):
            raise LLMError(f"Claude Code returned error: {data.get('result', 'unknown error')}")
        result = data.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip()
        return raw_output

    # JSON array of events (--output-format json --verbose produces
    # [{"type":"system",...}, {"type":"assistant","message":{...}}, ...])
    events = data if isinstance(data, list) else None

    # Also try JSONL (one JSON object per line)
    if events is None:
        events = []
        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not events:
        return raw_output

    # Extract the last assistant text and any result events
    last_assistant_text = ""
    for event in events:
        if not isinstance(event, dict):
            continue

        # Claude Code emits: {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}
        if event.get("type") == "assistant":
            message = event.get("message", {})
            content_blocks = message.get("content", [])
            text_parts = []
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        text_parts.append(text)
            if text_parts:
                last_assistant_text = "\n\n".join(text_parts)

        # {"type":"result","result":"..."} format
        if event.get("type") == "result":
            result = event.get("result")
            if isinstance(result, str) and result.strip():
                return result.strip()

    if last_assistant_text:
        return last_assistant_text

    return raw_output


def _merge_prompt(*, instructions: str, input_text: str, require_json: bool) -> str:
    return build_llm_request(
        instructions=instructions,
        input_text=input_text,
        require_structured_output=require_json,
    )


def _serialize_trace_json(value: dict[str, Any] | None) -> str:
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except TypeError:
        return repr(value)


def _parse_json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Structured response was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("Structured response must be a JSON object")
    return parsed


def _looks_like_auth_error(output: str) -> bool:
    normalized = output.lower()
    return "401 unauthorized" in normalized or "missing bearer" in normalized or "codex login" in normalized


class LLMManager:
    def __init__(self, profiles: dict[str, LLMClient], default_profile: str = "default") -> None:
        self._profiles = profiles
        self._default_profile = default_profile

    @classmethod
    def from_env(cls, *, working_directory: str | None = None) -> "LLMManager":
        provider = os.getenv("LLM_PROVIDER")
        if not provider:
            if shutil.which(os.getenv("CODEX_COMMAND", "codex")):
                provider = "codex_cli"
            elif shutil.which(os.getenv("CLAUDE_COMMAND", "claude")):
                provider = "claude_code"
            else:
                provider = "responses_api"
        provider = provider.lower()

        profiles: dict[str, LLMClient] = {}
        for profile in _discover_profiles(provider):
            profiles[profile] = _build_client_for_profile(
                provider=provider,
                profile=profile,
                working_directory=working_directory,
            )

        if "default" not in profiles:
            profiles["default"] = _build_client_for_profile(
                provider=provider,
                profile="default",
                working_directory=working_directory,
            )
        return cls(profiles=profiles)

    def resolve(self, profile: str | None = None) -> LLMClient:
        if not profile:
            return self._profiles[self._default_profile]
        return self._profiles.get(profile.lower(), self._profiles[self._default_profile])

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        selected_profile = profile.lower() if profile else self._default_profile
        client = self.resolve(profile)
        if selected_profile in self._profiles:
            return f"{selected_profile}: {client.describe()}"
        return f"{selected_profile} -> default: {client.describe()}"

    def available_profiles(self) -> list[str]:
        return sorted(self._profiles)


class TracedLLMClient(LLMClient):
    def __init__(self, client: Any, *, client_label: str | None = None) -> None:
        self._client = client
        self._client_label = client_label or self._describe_client(client)

    def is_enabled(self) -> bool:
        return bool(self._client.is_enabled())

    def describe(self) -> str:
        return str(self._client.describe())

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        prompt = _merge_prompt(instructions=instructions, input_text=input_text, require_json=False)
        request_event_id = log_llm_prompt_event(
            client_label=self._client_label,
            transport="wrapped_client_call",
            instructions=instructions,
            input_text=input_text,
            final_prompt=prompt,
            require_structured_output=False,
            effort=effort,
        )
        start_time = perf_counter()
        output_text = ""
        try:
            output_text = str(self._client.generate_text(instructions=instructions, input_text=input_text, effort=effort))
        except Exception as exc:
            log_llm_response_event(
                client_label=self._client_label,
                transport="wrapped_client_call",
                require_structured_output=False,
                response_text=output_text,
                elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
                request_event_id=request_event_id,
                error=str(exc),
            )
            raise
        log_llm_response_event(
            client_label=self._client_label,
            transport="wrapped_client_call",
            require_structured_output=False,
            response_text=output_text,
            elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
            request_event_id=request_event_id,
        )
        return output_text

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict[str, Any],
        effort: str | None = None,
    ) -> dict[str, Any]:
        prompt = _merge_prompt(instructions=instructions, input_text=input_text, require_json=True)
        request_event_id = log_llm_prompt_event(
            client_label=self._client_label,
            transport="wrapped_client_call",
            instructions=instructions,
            input_text=input_text,
            final_prompt=prompt,
            require_structured_output=True,
            schema_name=schema_name,
            effort=effort,
        )
        start_time = perf_counter()
        response_payload: dict[str, Any] | None = None
        try:
            response_payload = self._client.generate_json(
                instructions=instructions,
                input_text=input_text,
                schema_name=schema_name,
                schema=schema,
                effort=effort,
            )
        except Exception as exc:
            log_llm_response_event(
                client_label=self._client_label,
                transport="wrapped_client_call",
                require_structured_output=True,
                response_text=_serialize_trace_json(response_payload),
                elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
                schema_name=schema_name,
                request_event_id=request_event_id,
                error=str(exc),
            )
            raise
        log_llm_response_event(
            client_label=self._client_label,
            transport="wrapped_client_call",
            require_structured_output=True,
            response_text=_serialize_trace_json(response_payload),
            elapsed_ms=round((perf_counter() - start_time) * 1000, 2),
            schema_name=schema_name,
            request_event_id=request_event_id,
        )
        return response_payload

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    @staticmethod
    def _describe_client(client: Any) -> str:
        try:
            description = str(client.describe()).strip()
        except Exception:
            description = ""
        return description or client.__class__.__name__


def ensure_traced_llm_client(client: Any) -> Any:
    if isinstance(client, (ResponsesLLMClient, CodexCliLLMClient, ClaudeCodeLLMClient, TracedLLMClient)):
        return client
    # Pass through executor clients — they have their own tracing and a different interface.
    if hasattr(client, "execute_task"):
        return client
    if not hasattr(client, "generate_text") or not hasattr(client, "generate_json"):
        return client
    return TracedLLMClient(client)


def _discover_profiles(provider: str) -> list[str]:
    if provider == "claude_code":
        prefix = "CLAUDE_MODEL_"
    elif provider == "codex_cli":
        prefix = "CODEX_MODEL_"
    else:
        prefix = "OPENAI_MODEL_"
    profiles = {"default"}
    for env_key in os.environ:
        if not env_key.startswith(prefix):
            continue
        suffix = env_key.removeprefix(prefix).lower()
        if suffix:
            profiles.add(suffix)
    if provider == "claude_code":
        profiles.add("executor")
    return sorted(profiles)


def _build_client_for_profile(
    provider: str,
    profile: str,
    *,
    working_directory: str | None = None,
) -> LLMClient:
    if provider == "claude_code" and profile == "executor":
        from core.executor import ClaudeCodeExecutorClient, ClaudeCodeExecutorConfig

        return ClaudeCodeExecutorClient(
            ClaudeCodeExecutorConfig(
                command=os.getenv("CLAUDE_COMMAND", "claude"),
                model=os.getenv(
                    "CLAUDE_EXECUTOR_MODEL",
                    os.getenv("CLAUDE_MODEL", "claude-opus-4-6"),
                ),
                timeout_seconds=int(os.getenv("CLAUDE_EXECUTOR_TIMEOUT_SECONDS", "600")),
                working_directory=working_directory,
                permission_mode=os.getenv("CLAUDE_EXECUTOR_PERMISSION_MODE", "auto"),
                max_turns=int(os.getenv("CLAUDE_EXECUTOR_MAX_TURNS", "200")),
            )
        )

    if provider == "claude_code":
        return ClaudeCodeLLMClient(
            ClaudeCodeConfig(
                command=os.getenv("CLAUDE_COMMAND", "claude"),
                model=_get_profile_model(
                    profile=profile,
                    default_env="CLAUDE_MODEL",
                    profile_prefix="CLAUDE_MODEL_",
                    fallback_default="claude-opus-4-6",
                ),
                timeout_seconds=int(os.getenv("CLAUDE_TIMEOUT_SECONDS", "300")),
                working_directory=working_directory,
                max_turns=int(os.getenv("CLAUDE_MAX_TURNS", "1")),
            )
        )

    if provider == "codex_cli":
        return CodexCliLLMClient(
            CodexCLIConfig(
                command=os.getenv("CODEX_COMMAND", "codex"),
                model=_get_profile_model(
                    profile=profile,
                    default_env="CODEX_MODEL",
                    profile_prefix="CODEX_MODEL_",
                    fallback_default="gpt-5.3-codex",
                    legacy_default_env="OPENAI_MODEL",
                    legacy_profile_prefix="OPENAI_MODEL_",
                ),
                timeout_seconds=int(os.getenv("CODEX_TIMEOUT_SECONDS", "300")),
                working_directory=working_directory,
                sandbox_mode=os.getenv("CODEX_SANDBOX", "read-only"),
            )
        )

    return ResponsesLLMClient(
        ResponsesLLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=_get_profile_model(
                profile=profile,
                default_env="OPENAI_MODEL",
                profile_prefix="OPENAI_MODEL_",
                fallback_default="gpt-5.2-codex",
            ),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout_seconds=int(os.getenv("OPENAI_TIMEOUT_SECONDS", "180")),
            reasoning_effort=_get_reasoning_effort(profile),
        )
    )


def _get_profile_model(
    *,
    profile: str,
    default_env: str,
    profile_prefix: str,
    fallback_default: str,
    legacy_default_env: str | None = None,
    legacy_profile_prefix: str | None = None,
) -> str:
    if profile != "default":
        direct = os.getenv(f"{profile_prefix}{profile.upper()}")
        if direct:
            return direct
        if legacy_profile_prefix:
            legacy = os.getenv(f"{legacy_profile_prefix}{profile.upper()}")
            if legacy:
                return legacy
    return (
        os.getenv(default_env)
        or (os.getenv(legacy_default_env) if legacy_default_env else None)
        or fallback_default
    )


def _get_reasoning_effort(profile: str) -> str:
    if profile != "default":
        direct = os.getenv(f"OPENAI_REASONING_EFFORT_{profile.upper()}")
        if direct:
            return direct
    return os.getenv("OPENAI_REASONING_EFFORT", "medium")
