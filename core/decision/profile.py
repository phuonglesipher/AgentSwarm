from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple


class Branch(NamedTuple):
    """One routing target for a decision point.

    ``target`` overrides the actual node name returned by ``decide()`` and used
    in the edge map, while ``name`` remains the human-readable label the LLM
    sees in the JSON schema.  Useful when the real node name is a sentinel
    (e.g. LangGraph ``END = "__end__"``).
    """

    name: str
    description: str
    is_default: bool = False
    target: str = ""


@dataclass(frozen=True)
class DecisionProfile:
    """Configuration for one LLM-powered routing decision."""

    system_id: str
    branches: tuple[Branch, ...]
    context_instruction: str
    state_summarizer: Callable[[dict[str, Any]], str] | None = None
    llm_profile: str | None = None
    effort: str | None = None

    def __post_init__(self) -> None:
        if len(self.branches) < 2:
            raise ValueError(f"DecisionProfile '{self.system_id}' requires at least 2 branches")
        defaults = [b for b in self.branches if b.is_default]
        if len(defaults) != 1:
            raise ValueError(
                f"DecisionProfile '{self.system_id}' requires exactly 1 default branch, "
                f"found {len(defaults)}"
            )
        names = [b.name for b in self.branches]
        if len(names) != len(set(names)):
            raise ValueError(f"DecisionProfile '{self.system_id}' has duplicate branch names")

    @property
    def default_branch(self) -> str:
        b = next(b for b in self.branches if b.is_default)
        return b.target or b.name

    @property
    def branch_names(self) -> frozenset[str]:
        return frozenset(b.name for b in self.branches)
