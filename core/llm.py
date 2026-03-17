from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
from urllib import error, request


class LLMError(RuntimeError):
    """Raised when an LLM backend fails or returns unusable output."""


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
        response = self._request(
            {
                "instructions": instructions,
                "input": input_text,
                "reasoning": {"effort": effort or self.config.reasoning_effort},
            }
        )
        return self._extract_output_text(response)

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict[str, Any],
        effort: str | None = None,
    ) -> dict[str, Any]:
        response = self._request(
            {
                "instructions": instructions,
                "input": input_text,
                "reasoning": {"effort": effort or self.config.reasoning_effort},
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
        return _parse_json_object(output_text)

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
        try:
            with request.urlopen(api_request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"OpenAI API returned {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise LLMError(f"OpenAI API request failed: {exc}") from exc

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

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort
        prompt = _merge_prompt(instructions=instructions, input_text=input_text, require_json=False)
        return self._run_codex(prompt=prompt, schema=None)

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
        output = self._run_codex(prompt=prompt, schema={"name": schema_name, "schema": schema})
        return _parse_json_object(output)

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
                "read-only",
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

            output = output_path.read_text(encoding="utf-8").strip()
            if not output:
                raise LLMError("Codex CLI returned an empty final message")
            return output


def _merge_prompt(*, instructions: str, input_text: str, require_json: bool) -> str:
    sections = [
        "System instructions:",
        instructions.strip(),
        "",
        "Task input:",
        input_text.strip(),
    ]
    if require_json:
        sections.extend(
            [
                "",
                "Output requirements:",
                "- Return only a JSON object matching the provided schema.",
                "- Do not include markdown fences or explanatory text.",
            ]
        )
    return "\n".join(sections).strip()


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
            provider = "codex_cli" if shutil.which(os.getenv("CODEX_COMMAND", "codex")) else "responses_api"
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


def _discover_profiles(provider: str) -> list[str]:
    if provider == "codex_cli":
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
    return sorted(profiles)


def _build_client_for_profile(
    provider: str,
    profile: str,
    *,
    working_directory: str | None = None,
) -> LLMClient:
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
