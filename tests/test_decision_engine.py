from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from core.decision.engine import DecisionEngine, _build_schema, _default_state_summarizer
from core.decision.profile import Branch, DecisionProfile
from core.llm import LLMError


def _make_profile(**overrides: Any) -> DecisionProfile:
    defaults = dict(
        system_id="test_gate",
        branches=(
            Branch("branch_a", "Choose when condition A holds"),
            Branch("branch_b", "Choose when condition B holds", is_default=True),
        ),
        context_instruction="Decide based on the test state.",
    )
    defaults.update(overrides)
    return DecisionProfile(**defaults)


def _make_engine(
    profile: DecisionProfile | None = None,
    llm_enabled: bool = True,
    generate_json_return: dict[str, Any] | None = None,
    generate_json_side_effect: Exception | None = None,
) -> DecisionEngine:
    profile = profile or _make_profile()
    llm = MagicMock()
    llm.is_enabled.return_value = llm_enabled
    if generate_json_side_effect:
        llm.generate_json.side_effect = generate_json_side_effect
    elif generate_json_return is not None:
        llm.generate_json.return_value = generate_json_return

    context = MagicMock()
    context.get_llm.return_value = llm
    metadata = MagicMock()
    metadata.name = "test_workflow"

    return DecisionEngine(profile=profile, context=context, metadata=metadata)


class DecisionProfileTests(unittest.TestCase):
    def test_valid_profile(self) -> None:
        profile = _make_profile()
        self.assertEqual(profile.default_branch, "branch_b")
        self.assertEqual(profile.branch_names, frozenset({"branch_a", "branch_b"}))

    def test_fewer_than_two_branches_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DecisionProfile(
                system_id="bad",
                branches=(Branch("only_one", "solo", is_default=True),),
                context_instruction="nope",
            )

    def test_no_default_branch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DecisionProfile(
                system_id="bad",
                branches=(
                    Branch("a", "desc a"),
                    Branch("b", "desc b"),
                ),
                context_instruction="nope",
            )

    def test_multiple_defaults_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DecisionProfile(
                system_id="bad",
                branches=(
                    Branch("a", "desc a", is_default=True),
                    Branch("b", "desc b", is_default=True),
                ),
                context_instruction="nope",
            )

    def test_duplicate_branch_names_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DecisionProfile(
                system_id="bad",
                branches=(
                    Branch("same", "desc a", is_default=True),
                    Branch("same", "desc b"),
                ),
                context_instruction="nope",
            )


class SchemaTests(unittest.TestCase):
    def test_schema_has_branch_names_as_enum(self) -> None:
        branches = (
            Branch("alpha", "first"),
            Branch("beta", "second", is_default=True),
        )
        schema = _build_schema(branches)
        self.assertEqual(schema["properties"]["choice"]["enum"], ["alpha", "beta"])
        self.assertIn("reasoning", schema["properties"])
        self.assertFalse(schema["additionalProperties"])


class DefaultStateSummarizerTests(unittest.TestCase):
    def test_skips_internal_keys(self) -> None:
        state = {"_internal": "hidden", "visible": "shown"}
        result = _default_state_summarizer(state)
        self.assertNotIn("_internal", result)
        self.assertIn("visible: shown", result)

    def test_truncates_long_strings(self) -> None:
        state = {"long_field": "x" * 600}
        result = _default_state_summarizer(state)
        self.assertIn("...", result)
        self.assertTrue(len(result) < 600)

    def test_truncates_long_lists(self) -> None:
        state = {"big_list": list(range(20))}
        result = _default_state_summarizer(state)
        self.assertIn("20 items", result)


class DecisionEngineTests(unittest.TestCase):
    def test_successful_llm_routing(self) -> None:
        engine = _make_engine(
            generate_json_return={"choice": "branch_a", "reasoning": "condition A is met"},
        )
        result = engine.decide({"some_key": "some_value"})
        self.assertEqual(result, "branch_a")

    def test_fallback_when_llm_disabled(self) -> None:
        engine = _make_engine(llm_enabled=False)
        result = engine.decide({"some_key": "some_value"})
        self.assertEqual(result, "branch_b")

    def test_fallback_on_llm_error(self) -> None:
        engine = _make_engine(generate_json_side_effect=LLMError("API down"))
        result = engine.decide({"some_key": "some_value"})
        self.assertEqual(result, "branch_b")

    def test_fallback_on_invalid_choice(self) -> None:
        engine = _make_engine(
            generate_json_return={"choice": "nonexistent", "reasoning": "oops"},
        )
        result = engine.decide({"some_key": "some_value"})
        self.assertEqual(result, "branch_b")

    def test_fallback_on_missing_choice_key(self) -> None:
        engine = _make_engine(
            generate_json_return={"reasoning": "no choice field"},
        )
        result = engine.decide({"some_key": "some_value"})
        self.assertEqual(result, "branch_b")

    def test_custom_state_summarizer(self) -> None:
        profile = _make_profile(
            state_summarizer=lambda s: f"score={s.get('score', 0)}",
        )
        engine = _make_engine(
            profile=profile,
            generate_json_return={"choice": "branch_a", "reasoning": "score is high"},
        )
        result = engine.decide({"score": 95})
        self.assertEqual(result, "branch_a")
        # Verify the summarizer output was passed to the LLM
        call_args = engine._llm.generate_json.call_args
        self.assertIn("score=95", call_args.kwargs["input_text"])

    def test_wire_edges_adds_conditional_edges(self) -> None:
        engine = _make_engine()
        graph = MagicMock()
        engine.wire_edges(graph, "source_node", graph_name="test_graph")
        graph.add_conditional_edges.assert_called_once()
        args = graph.add_conditional_edges.call_args
        self.assertEqual(args[0][0], "source_node")
        edge_map = args[0][2]
        self.assertEqual(edge_map, {"branch_a": "branch_a", "branch_b": "branch_b"})


class BranchTargetTests(unittest.TestCase):
    """Tests for Branch.target field — allows display name to differ from wire target."""

    def test_decide_returns_target_when_set(self) -> None:
        profile = DecisionProfile(
            system_id="end_test",
            branches=(
                Branch("continue", "Keep going"),
                Branch("finish", "Done", is_default=True, target="__end__"),
            ),
            context_instruction="Test END handling.",
        )
        engine = _make_engine(
            profile=profile,
            generate_json_return={"choice": "finish", "reasoning": "done"},
        )
        result = engine.decide({"x": 1})
        self.assertEqual(result, "__end__")

    def test_decide_returns_name_when_no_target(self) -> None:
        profile = DecisionProfile(
            system_id="no_target",
            branches=(
                Branch("alpha", "first"),
                Branch("beta", "second", is_default=True),
            ),
            context_instruction="No targets set.",
        )
        engine = _make_engine(
            profile=profile,
            generate_json_return={"choice": "alpha", "reasoning": "pick alpha"},
        )
        result = engine.decide({"x": 1})
        self.assertEqual(result, "alpha")

    def test_default_branch_returns_target(self) -> None:
        profile = DecisionProfile(
            system_id="default_target",
            branches=(
                Branch("loop", "loop back"),
                Branch("done", "exit", is_default=True, target="__end__"),
            ),
            context_instruction="Test.",
        )
        self.assertEqual(profile.default_branch, "__end__")

    def test_fallback_uses_target(self) -> None:
        profile = DecisionProfile(
            system_id="fallback_target",
            branches=(
                Branch("loop", "loop back"),
                Branch("done", "exit", is_default=True, target="__end__"),
            ),
            context_instruction="Test.",
        )
        engine = _make_engine(profile=profile, llm_enabled=False)
        result = engine.decide({"x": 1})
        self.assertEqual(result, "__end__")

    def test_wire_edges_uses_target_in_map(self) -> None:
        profile = DecisionProfile(
            system_id="wire_target",
            branches=(
                Branch("investigate", "keep investigating"),
                Branch("finish", "done", is_default=True, target="__end__"),
            ),
            context_instruction="Test.",
        )
        engine = _make_engine(profile=profile)
        graph = MagicMock()
        engine.wire_edges(graph, "review", graph_name="test")
        edge_map = graph.add_conditional_edges.call_args[0][2]
        self.assertEqual(edge_map, {"investigate": "investigate", "__end__": "__end__"})

    def test_schema_uses_name_not_target(self) -> None:
        branches = (
            Branch("loop", "keep going"),
            Branch("finish", "done", is_default=True, target="__end__"),
        )
        schema = _build_schema(branches)
        self.assertEqual(schema["properties"]["choice"]["enum"], ["loop", "finish"])
        self.assertNotIn("__end__", schema["properties"]["choice"]["enum"])


if __name__ == "__main__":
    unittest.main()
