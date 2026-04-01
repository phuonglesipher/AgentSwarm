from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, NamedTuple


class ReviewCriterion(NamedTuple):
    """One scored criterion. expected_sections are doc headings that must be present."""

    name: str
    weight: int
    expected_sections: tuple[str, ...] = ()


class HardBlocker(NamedTuple):
    """A named hard-gate that gets auto-dismissed when all criteria pass."""

    label: str
    description: str


@dataclass(frozen=True)
class ReviewProfile:
    """Domain-specific configuration for one review type."""

    # --- Identity ---
    system_id: str
    display_name: str

    # --- Criteria ---
    criteria: tuple[ReviewCriterion, ...]

    # --- Thresholds & Rounds ---
    approval_threshold: int
    min_rounds: int
    max_rounds: int
    stagnation_limit: int = 2

    # --- Section enforcement ---
    require_missing_section_free: bool = True

    # --- Hard blockers (gameplay has 4; others have none) ---
    hard_blockers: tuple[HardBlocker, ...] = ()

    # --- Mandatory verification action ---
    # Either a static string or a callable(improvement_actions, blocking_issues) -> str
    mandatory_action: str | Callable[[list[str], list[str]], str] = ""

    # --- Prompt assembly ---
    prompt_persona: str = ""
    prompt_domain_instructions: str = ""
    prompt_round_guidance: Callable[[int, int], str] | None = None

    # --- Process-only filtering (domain extensions on top of shared base) ---
    extra_process_keywords: tuple[str, ...] = ()
    extra_process_patterns: tuple[re.Pattern[str], ...] = ()

    # --- LLM output format ---
    supports_markdown_fallback: bool = True

    # --- State I/O ---
    doc_field_name: str = "investigation_doc"
    review_doc_title: str = "Investigation Review"
    notes_heading: str = "Senior Engineer Notes"
    no_action_text: str = "No further investigation changes requested."
    domain_noun: str = "investigation"
    schema_name: str = "investigation_review"

    # --- Backward-compat field aliasing ---
    # Each tuple is (canonical_field, alias_field). Engine emits both.
    state_field_aliases: tuple[tuple[str, str], ...] = ()

    # --- Dynamic criteria override (optimization reads from state) ---
    dynamic_criteria_field: str | None = None
    dynamic_domain_field: str | None = None

    # --- Derived helpers ---

    @property
    def criteria_weight_map(self) -> dict[str, int]:
        return {c.name: c.weight for c in self.criteria}

    @property
    def expected_sections(self) -> tuple[str, ...]:
        return tuple(s for c in self.criteria for s in c.expected_sections)

    @property
    def loop_id(self) -> str:
        return self.system_id
