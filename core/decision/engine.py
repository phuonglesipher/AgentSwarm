from __future__ import annotations

import logging
from typing import Any

from core.graph_logging import log_graph_payload_event, trace_route_decision
from core.llm import LLMError
from core.models import WorkflowContext, WorkflowMetadata
from core.natural_language_prompts import build_prompt_brief

from .profile import Branch, DecisionProfile

logger = logging.getLogger(__name__)

_MAX_STR_LEN = 500
_MAX_LIST_ITEMS = 10


def _default_state_summarizer(state: dict[str, Any]) -> str:
    """Serialize non-internal state keys into a compact text summary."""
    lines: list[str] = []
    for key, value in state.items():
        if key.startswith("_"):
            continue
        if isinstance(value, str):
            display = value[:_MAX_STR_LEN] + "..." if len(value) > _MAX_STR_LEN else value
        elif isinstance(value, list):
            if len(value) > _MAX_LIST_ITEMS:
                display = f"[{len(value)} items, first {_MAX_LIST_ITEMS}: {value[:_MAX_LIST_ITEMS]!r}]"
            else:
                display = repr(value)
        else:
            display = repr(value)
        lines.append(f"{key}: {display}")
    return "\n".join(lines)


def _build_schema(branches: tuple[Branch, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": [b.name for b in branches]},
            "reasoning": {"type": "string"},
        },
        "required": ["choice", "reasoning"],
        "additionalProperties": False,
    }


def _build_prompt(profile: DecisionProfile, state_summary: str) -> str:
    branch_descriptions = "\n".join(
        f"- {b.name}: {b.description}" for b in profile.branches
    )
    return build_prompt_brief(
        opening=f"Decide the next step for: {profile.system_id}",
        sections=[
            ("Available branches", branch_descriptions),
            ("Current state", state_summary),
        ],
        closing="Pick exactly one branch. Explain your reasoning briefly.",
    )


class DecisionEngine:
    """LLM-powered routing for LangGraph conditional edges.

    Configured via a ``DecisionProfile`` that describes available branches
    and provides domain context. Falls back to the default branch when
    the LLM is unavailable or returns an invalid choice.
    """

    def __init__(
        self,
        profile: DecisionProfile,
        context: WorkflowContext,
        metadata: WorkflowMetadata,
    ) -> None:
        self._profile = profile
        self._context = context
        self._metadata = metadata
        self._llm = context.get_llm(profile.llm_profile)
        self._schema = _build_schema(profile.branches)

    def decide(self, state: dict[str, Any]) -> str:
        """Route function: state -> next node name.

        Compatible with ``trace_route_decision`` and ``add_conditional_edges``.
        """
        p = self._profile
        summarizer = p.state_summarizer or _default_state_summarizer
        state_summary = summarizer(state)

        if not self._llm.is_enabled():
            logger.info("DecisionEngine[%s]: LLM disabled, using default '%s'", p.system_id, p.default_branch)
            return p.default_branch

        try:
            result = self._llm.generate_json(
                instructions=(
                    f"You are a workflow routing decision maker. {p.context_instruction} "
                    "Choose the single best branch based on the current state."
                ),
                input_text=_build_prompt(p, state_summary),
                schema_name=f"decision_{p.system_id}",
                schema=self._schema,
                effort=p.effort,
            )
        except (LLMError, Exception) as exc:
            logger.warning("DecisionEngine[%s]: LLM error, falling back to '%s': %s", p.system_id, p.default_branch, exc)
            return p.default_branch

        choice = result.get("choice", "")
        reasoning = result.get("reasoning", "")

        if choice not in p.branch_names:
            logger.warning(
                "DecisionEngine[%s]: invalid choice '%s', falling back to '%s'",
                p.system_id, choice, p.default_branch,
            )
            return p.default_branch

        log_graph_payload_event(
            state=state,
            graph_name=self._metadata.name,
            node_name=p.system_id,
            phase="ROUTE_LLM_REASONING",
            payload_label="decision",
            payload={"choice": choice, "reasoning": reasoning},
            message=f"choice={choice}",
        )

        branch = next(b for b in p.branches if b.name == choice)
        return branch.target or branch.name

    def wire_edges(
        self,
        graph: Any,
        source_node: str,
        *,
        graph_name: str,
    ) -> None:
        """Convenience: wires ``decide()`` into graph with tracing and edge map."""
        route_fn = trace_route_decision(
            graph_name=graph_name,
            router_name=self._profile.system_id,
            route_fn=self.decide,
        )
        graph.add_conditional_edges(
            source_node,
            route_fn,
            {(b.target or b.name): (b.target or b.name) for b in self._profile.branches},
        )
