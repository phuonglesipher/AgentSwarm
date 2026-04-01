from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.review.profile import ReviewCriterion, ReviewProfile


class PlanCriterion(NamedTuple):
    """One required section in the plan document.

    ``expected_headings`` are markdown headings (## level) that the plan must
    contain.  They feed into the ReviewEngine when the plan is scored.
    """

    name: str
    weight: int
    description: str
    expected_headings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanStrategy:
    """Mode-specific prompt fragments.

    Each planning mode (bugfix, refactor, new_feature, ...) provides one
    frozen instance that the engine injects into the LLM prompt.  This
    replaces the ``_planning_mode_profile()`` dict pattern.
    """

    mode_id: str
    display_name: str

    # --- Plan-level fragments ---
    task_focus: str
    plan_overview: str
    plan_steps: tuple[str, ...]
    validation_focus: str
    default_tests: tuple[str, ...]
    risks: tuple[str, ...]
    acceptance: tuple[str, ...]

    # --- Design-doc fragments (optional, empty if not used) ---
    design_overview: str = ""
    design_behavior: str = ""
    design_technical_note: str = ""
    design_risk: str = ""
    design_focus: str = ""


@dataclass(frozen=True)
class PlanProfile:
    """Domain-specific configuration for one plan type.

    Mirrors ``ReviewProfile`` in structure: the engine reads this at
    construction time and uses it to drive all prompt assembly, artifact
    writing, and state I/O.
    """

    # --- Identity ---
    system_id: str
    display_name: str

    # --- Plan sections ---
    criteria: tuple[PlanCriterion, ...]

    # --- Strategy registry (replaces planning_mode conditionals) ---
    strategies: dict[str, PlanStrategy]
    default_strategy: str

    # --- LLM prompt ---
    prompt_persona: str = ""
    prompt_domain_instructions: str = ""

    # --- Quality loop integration ---
    approval_threshold: int = 90
    min_rounds: int = 1
    max_rounds: int = 3
    stagnation_limit: int = 2

    # --- State I/O field names ---
    plan_doc_field: str = "plan_doc"
    strategy_field: str = "planning_mode"
    context_fields: tuple[str, ...] = ()
    round_field: str = "plan_round"

    # --- Backward-compat field aliasing ---
    state_field_aliases: tuple[tuple[str, str], ...] = ()

    # --- Derived helpers ---

    def get_strategy(self, mode_id: str) -> PlanStrategy:
        """Return the strategy for *mode_id*, falling back to the default."""
        return self.strategies.get(mode_id, self.strategies[self.default_strategy])

    @property
    def criteria_weight_map(self) -> dict[str, int]:
        return {c.name: c.weight for c in self.criteria}

    @property
    def expected_headings(self) -> tuple[str, ...]:
        return tuple(h for c in self.criteria for h in c.expected_headings)

    def to_review_profile(self, **overrides: object) -> ReviewProfile:
        """Generate a ``ReviewProfile`` that can score plans from this profile.

        The criteria names and weights are carried over so review scoring stays
        in sync with the plan structure automatically.
        """
        from core.review.profile import ReviewCriterion as RC, ReviewProfile as RP

        defaults: dict[str, object] = {
            "system_id": f"{self.system_id}-review",
            "display_name": f"{self.display_name} Review",
            "criteria": tuple(
                RC(name=c.name, weight=c.weight, expected_sections=c.expected_headings)
                for c in self.criteria
            ),
            "approval_threshold": self.approval_threshold,
            "min_rounds": self.min_rounds,
            "max_rounds": self.max_rounds,
            "stagnation_limit": self.stagnation_limit,
            "doc_field_name": self.plan_doc_field,
            "review_doc_title": f"{self.display_name} Review",
            "domain_noun": "plan",
            "schema_name": self.system_id.replace("-", "_") + "_review",
        }
        defaults.update(overrides)
        return RP(**defaults)  # type: ignore[arg-type]
