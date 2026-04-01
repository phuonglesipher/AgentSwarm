from __future__ import annotations

from typing import Any, TYPE_CHECKING

from core.natural_language_prompts import build_prompt_brief

if TYPE_CHECKING:
    from .profile import PlanCriterion, PlanProfile, PlanStrategy


def build_plan_instructions(
    profile: PlanProfile,
    strategy: PlanStrategy,
    *,
    revise: bool = False,
) -> str:
    """Assemble the LLM instruction string for plan generation."""
    verb = "Rewrite the full markdown plan" if revise else "Produce a markdown plan"
    section_names = ", ".join(c.name for c in profile.criteria)

    parts: list[str] = []
    if profile.prompt_persona:
        parts.append(profile.prompt_persona)
    parts.append(
        f"{verb} using these exact sections: {section_names}. "
        f"Shape the plan for {strategy.display_name} work. {strategy.task_focus}"
    )
    if profile.prompt_domain_instructions:
        parts.append(profile.prompt_domain_instructions)
    return "\n\n".join(p for p in parts if p.strip())


def build_plan_context(
    profile: PlanProfile,
    strategy: PlanStrategy,
    state: dict[str, Any],
    *,
    revise: bool = False,
) -> str:
    """Assemble the LLM input text from state fields and strategy fragments."""
    sections: list[tuple[str, str]] = [
        ("Task request", str(state.get("task_prompt", "")).strip()),
        (
            "Planning frame",
            "\n".join([
                f"- Planning mode: {strategy.mode_id}",
                f"- Mode focus: {strategy.task_focus}",
            ]),
        ),
    ]

    # Inject context fields from state
    for field_name in profile.context_fields:
        value = state.get(field_name)
        if not value:
            continue
        if isinstance(value, list):
            text = "\n".join(f"- {item}" for item in value) if value else ""
        else:
            text = str(value).strip()
        if text:
            label = field_name.replace("_", " ").title()
            sections.append((label, text))

    # Strategy-level structural guidance
    guidance_lines: list[str] = [strategy.plan_overview]
    for step in strategy.plan_steps:
        guidance_lines.append(f"- {step}")
    sections.append(("Mode-specific guidance", "\n".join(guidance_lines)))

    # Revision feedback
    if revise:
        blocking = state.get("review_blocking_issues", [])
        improvements = state.get("review_improvement_actions", [])
        if blocking:
            sections.append((
                "Open blocking issues",
                "\n".join(f"- {item}" for item in blocking),
            ))
        if improvements:
            sections.append((
                "Requested improvements",
                "\n".join(f"- {item}" for item in improvements),
            ))
        prior_plan = str(state.get(profile.plan_doc_field, "")).strip()
        if prior_plan:
            sections.append(("Previous plan draft", prior_plan))

    return build_prompt_brief(
        opening="Draft the next plan for this workflow.",
        sections=sections,
        closing=(
            "Produce a plan that is implementation-ready and anchored on concrete evidence. "
            "Keep the plan technical and do not add review-round bookkeeping, sign-off "
            "workflow, or artifact naming requirements. "
            f"Stay aligned with the {strategy.display_name} planning mode."
        ),
    )


def build_plan_fallback(
    profile: PlanProfile,
    strategy: PlanStrategy,
    state: dict[str, Any],
) -> str:
    """Build a deterministic plan document when the LLM is unavailable."""
    lines = [f"# {profile.display_name}", ""]

    for criterion in profile.criteria:
        lines.append(f"## {criterion.name}")
        content = _fallback_section_content(criterion, strategy, state)
        lines.append(content)
        lines.append("")

    return "\n".join(lines).rstrip()


def _fallback_section_content(
    criterion: PlanCriterion,
    strategy: PlanStrategy,
    state: dict[str, Any],
) -> str:
    """Generate deterministic content for one plan section."""
    name_lower = criterion.name.lower()

    if "overview" in name_lower:
        return f"- {state.get('task_prompt', 'No task prompt.')}\n{strategy.plan_overview}"

    if "task type" in name_lower or "classification" in name_lower:
        task_type = state.get("task_type", "unknown")
        reason = state.get("classification_reason", f"Classified as {task_type}.")
        return (
            f"- {task_type}\n"
            f"- Planning mode: {strategy.mode_id}\n"
            f"- Classification reason: {reason}\n"
            f"{strategy.task_focus}"
        )

    if "step" in name_lower or "implementation" in name_lower:
        bullets: list[str] = []
        for step in strategy.plan_steps:
            bullets.append(f"- {step}")
        return "\n".join(bullets) or "- No steps defined."

    if "test" in name_lower or "validation" in name_lower:
        test_hits = state.get("test_hits", [])
        if test_hits:
            items = [f"- {t}" for t in test_hits]
            items.append(f"- {strategy.validation_focus}")
            return "\n".join(items)
        return "\n".join(f"- {t}" for t in strategy.default_tests)

    if "risk" in name_lower:
        return "\n".join(strategy.risks) or "- No risks identified."

    if "acceptance" in name_lower or "criteria" in name_lower:
        return "\n".join(strategy.acceptance) or "- No acceptance criteria defined."

    if "doc" in name_lower or "reference" in name_lower:
        doc_hits = state.get("doc_hits", [])
        if doc_hits:
            return "\n".join(f"- {d}" for d in doc_hits)
        return "- No grounded docs found."

    return f"- {criterion.description}"
