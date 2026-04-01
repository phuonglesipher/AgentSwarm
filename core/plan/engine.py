from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from core.llm import LLMError, ensure_traced_llm_client
from core.models import WorkflowContext, WorkflowMetadata
from core.text_utils import normalize_text, slugify

from .parsing import detect_missing_headings, extract_plan_summary
from .profile import PlanProfile
from .prompt_builder import build_plan_context, build_plan_fallback, build_plan_instructions


class PlanEngine:
    """Unified plan generation engine.

    Each plan type provides a ``PlanProfile`` that configures sections,
    strategies, prompts, and domain-specific extensions.  The engine handles
    LLM calls, fallback template generation, artifact writing, and state
    output.

    Wire ``generate_plan`` and ``revise_plan`` as LangGraph node functions.
    """

    def __init__(
        self,
        profile: PlanProfile,
        context: WorkflowContext,
        metadata: WorkflowMetadata,
    ) -> None:
        self._profile = profile
        self._context = context
        self._metadata = metadata

    # ------------------------------------------------------------------ #
    #  Public node functions
    # ------------------------------------------------------------------ #

    def generate_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        """Initial plan generation.  Pass to ``StateGraph.add_node()``."""
        return self._run(state, revise=False)

    def revise_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        """Revision pass — incorporates review feedback into next draft."""
        return self._run(state, revise=True)

    # ------------------------------------------------------------------ #
    #  Internal implementation
    # ------------------------------------------------------------------ #

    def _run(self, state: dict[str, Any], *, revise: bool) -> dict[str, Any]:
        p = self._profile
        plan_round = int(state.get(p.round_field, 0)) + 1
        artifact_path = self._artifact_dir(state)

        # Resolve active strategy
        mode_id = str(state.get(p.strategy_field, p.default_strategy)).strip().lower()
        strategy = p.get_strategy(mode_id)

        # Build LLM prompt
        instructions = build_plan_instructions(p, strategy, revise=revise)
        input_text = build_plan_context(p, strategy, state, revise=revise)

        # Generate plan document
        plan_doc = self._call_llm(instructions, input_text)
        if plan_doc is None:
            plan_doc = build_plan_fallback(p, strategy, state)

        # Write artifact
        (artifact_path / f"plan_round_{plan_round}.md").write_text(
            plan_doc, encoding="utf-8",
        )
        (artifact_path / "plan.md").write_text(plan_doc, encoding="utf-8")

        # Build summary
        summary = extract_plan_summary(plan_doc) or (
            f"{self._metadata.name} generated {p.display_name} (round {plan_round})."
        )

        # Build output
        output: dict[str, Any] = {
            p.plan_doc_field: plan_doc,
            p.round_field: plan_round,
            "artifact_dir": str(artifact_path),
            "summary": summary,
        }
        return _apply_field_aliases(output, p.state_field_aliases)

    def _call_llm(self, instructions: str, input_text: str) -> str | None:
        """Call the LLM, returning ``None`` on failure so the caller can
        fall back to a deterministic template."""
        llm = ensure_traced_llm_client(self._context.llm)
        if not llm.is_enabled():
            return None
        try:
            return llm.generate_text(
                instructions=instructions,
                input_text=input_text,
            )
        except (LLMError, Exception):
            return None

    def _artifact_dir(self, state: dict[str, Any]) -> Path:
        existing = str(state.get("artifact_dir", "")).strip()
        if existing:
            path = Path(existing)
            path.mkdir(parents=True, exist_ok=True)
            return path
        run_dir = str(state.get("run_dir", "")).strip()
        base_dir = Path(run_dir) if run_dir else Path(self._context.artifact_root) / "adhoc"
        task_id = str(state.get("task_id", "")).strip() or state.get("task_prompt", "plan")
        digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:6]
        task_dir = f"{_short_slug(task_id, fallback='plan')}-{digest}"
        path = base_dir / "tasks" / task_dir / self._metadata.name
        path.mkdir(parents=True, exist_ok=True)
        return path


# ------------------------------------------------------------------ #
#  Private helpers
# ------------------------------------------------------------------ #


def _apply_field_aliases(
    output: dict[str, Any],
    aliases: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    if not aliases:
        return output
    result = dict(output)
    for canonical_name, alias_name in aliases:
        if canonical_name in result:
            result[alias_name] = result[canonical_name]
    return result


def _short_slug(value: str, *, fallback: str, max_length: int = 18) -> str:
    slug = slugify(value, fallback=fallback)
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(normalize_text(value).encode("utf-8")).hexdigest()[:6]
    keep = max(1, max_length - len(digest) - 1)
    return f"{slug[:keep].rstrip('-') or fallback}-{digest}"
