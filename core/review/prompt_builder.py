from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .profile import ReviewCriterion, ReviewProfile

# Centralized agent constraint block — shared by ALL review profiles.
# When a new impossible-request category is discovered, add it here once.
AGENT_CONSTRAINT_BLOCK = (
    "CRITICAL CONSTRAINT: The investigator is an AI agent that CANNOT launch the Unreal Editor, "
    "run the game, or capture any runtime data. The following are IMPOSSIBLE for the agent and "
    "must NEVER be required, requested as blocking issues, or scored against:\n"
    "- Runtime GPU captures (stat gpu, RenderDoc, PIX, GPU Insights)\n"
    "- Runtime profiling data (stat Nanite, frame captures, GPU timestamps)\n"
    "- Play-in-Editor testing, in-editor profiling, or any editor interaction\n"
    "- A/B test results that require running the game\n"
    "- Execution traces from a live session\n"
    "Accept source code evidence (file paths, line numbers, grep results, config values, "
    "code path analysis) as valid evidence. The agent may recommend runtime steps as follow-up "
    "actions for the human, but absence of runtime data must not block approval or reduce scores."
)


def build_review_instructions(
    profile: ReviewProfile,
    criteria: tuple[ReviewCriterion, ...],
    review_round: int,
    *,
    optimization_domain: str | None = None,
) -> str:
    """Assemble the full LLM instruction string from profile fields."""
    persona = profile.prompt_persona
    if optimization_domain and "{optimization_domain}" in persona:
        persona = persona.replace("{optimization_domain}", optimization_domain)

    parts: list[str] = [
        persona,
        (
            "Score it hard against the following criteria: "
            f"{', '.join(f'{c.name} ({c.weight} pts)' for c in criteria)}."
        ),
    ]
    if profile.prompt_domain_instructions:
        domain_instructions = profile.prompt_domain_instructions
        if optimization_domain and "{optimization_domain}" in domain_instructions:
            domain_instructions = domain_instructions.replace("{optimization_domain}", optimization_domain)
        parts.append(domain_instructions)

    parts.append("")
    parts.append(AGENT_CONSTRAINT_BLOCK)
    parts.append("")
    parts.append(
        f"Minimum final-approval depth is {profile.min_rounds} review rounds. "
        "If the current round is below that floor, require one more pass that independently "
        "re-validates the findings with fresh source code evidence or static analysis. "
        "Do not approve early just because the first brief sounds plausible. "
        "Only gate on technical quality. Do not require organizational ownership assignment, "
        "DRI naming, commit/PR provenance, or other process artifacts unless they are explicitly "
        "present in the provided evidence."
    )

    if profile.prompt_round_guidance:
        guidance = profile.prompt_round_guidance(review_round, profile.min_rounds)
        if guidance:
            parts.append(guidance)

    return "\n\n".join(p for p in parts if p.strip())


def build_review_input(
    profile: ReviewProfile,
    *,
    task_prompt: str,
    doc_text: str,
    review_round: int,
    optimization_domain: str | None = None,
    extra_context_lines: list[str] | None = None,
) -> str:
    """Assemble the LLM input text from the document and task context."""
    lines = [
        f"Task prompt:\n{task_prompt}",
        "",
    ]
    if optimization_domain:
        lines.append(f"Optimization domain: {optimization_domain}")
    if extra_context_lines:
        lines.extend(extra_context_lines)
    lines.extend([
        f"Review round: {review_round}/{profile.max_rounds}",
        "",
        f"Minimum rounds required before final approval can stick: {profile.min_rounds}",
        "",
        f"{profile.display_name} document:\n{doc_text}",
    ])
    return "\n".join(lines)


def build_markdown_fallback_instructions(profile: ReviewProfile) -> str:
    """Extra instructions appended when falling back to markdown output."""
    return (
        " Return markdown using this exact shape: "
        f"# {profile.review_doc_title}, Decision: APPROVE or REVISE, Overall Score: NN/100, "
        "## Criterion Scores, ## Blocking Issues, "
        f"## Improvement Checklist, ## {profile.notes_heading}. "
        "Use one bullet per criterion in the form `- Criterion: score/max - rationale`. "
        "If there are no blocking issues, write exactly `- None.` under Blocking Issues. "
        f"If there are no further changes requested, write exactly `- [x] {profile.no_action_text}` "
        "Keep Overall Score numerically consistent with the criterion bullets. Do not use JSON."
    )
