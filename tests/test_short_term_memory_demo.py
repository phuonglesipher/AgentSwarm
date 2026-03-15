from __future__ import annotations

from operator import add
from typing import Annotated, Any
import unittest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict


class SharedMemoryState(TypedDict):
    prompt: str
    messages: Annotated[list[str], add]
    seen_by_a: list[str]
    seen_by_b: list[str]


class WrapperState(TypedDict):
    prompt: str
    outputs: Annotated[list[str], add]


class PrivateMemoryState(TypedDict):
    prompt: str
    private_messages: Annotated[list[str], add]


class RepoStyleWorkflowState(TypedDict):
    task_prompt: str
    plan_doc: str
    review_round: int
    review_notes: Annotated[list[str], add]
    reviewer_seen_notes: list[str]
    implementer_seen_notes: list[str]
    final_report: dict[str, Any]


def _shared_llm_a(state: SharedMemoryState) -> dict[str, Any]:
    seen = list(state["messages"])
    return {
        "seen_by_a": seen,
        "messages": [f"a<{state['prompt']}> saw={len(seen)}"],
    }


def _shared_llm_b(state: SharedMemoryState) -> dict[str, Any]:
    seen = list(state["messages"])
    return {
        "seen_by_b": seen,
        "messages": [f"b<{state['prompt']}> saw={len(seen)}"],
    }


def _private_llm_a(state: PrivateMemoryState) -> dict[str, Any]:
    seen = list(state["private_messages"])
    return {"private_messages": [f"private-a<{state['prompt']}> saw={len(seen)}"]}


def _private_llm_b(state: PrivateMemoryState) -> dict[str, Any]:
    seen = list(state["private_messages"])
    return {"private_messages": [f"private-b<{state['prompt']}> saw={len(seen)}"]}


def _repo_plan_work(state: RepoStyleWorkflowState) -> dict[str, Any]:
    return {
        "plan_doc": f"Plan for {state['task_prompt']}",
        "review_round": state["review_round"] + 1,
        "review_notes": [f"plan<{state['task_prompt']}>"],
    }


def _repo_review_plan(state: RepoStyleWorkflowState) -> dict[str, Any]:
    seen = list(state["review_notes"])
    return {
        "reviewer_seen_notes": seen,
        "review_notes": [f"review<{state['task_prompt']}> saw={len(seen)}"],
    }


def _repo_implement_code(state: RepoStyleWorkflowState) -> dict[str, Any]:
    seen = list(state["review_notes"])
    return {
        "implementer_seen_notes": seen,
        "final_report": {
            "seen_count": len(seen),
            "last_note": seen[-1] if seen else "",
        },
    }


def _build_shared_parent_graph():
    child = StateGraph(SharedMemoryState)
    child.add_node("llm_a", _shared_llm_a)
    child.add_node("llm_b", _shared_llm_b)
    child.add_edge(START, "llm_a")
    child.add_edge("llm_a", "llm_b")
    child_graph = child.compile()

    parent = StateGraph(SharedMemoryState)
    parent.add_node("memory_subgraph", child_graph)
    parent.add_edge(START, "memory_subgraph")
    return parent.compile(checkpointer=InMemorySaver())


def _build_wrapper_parent_graph(*, persist_private_subgraph: bool):
    child = StateGraph(PrivateMemoryState)
    child.add_node("private_llm_a", _private_llm_a)
    child.add_node("private_llm_b", _private_llm_b)
    child.add_edge(START, "private_llm_a")
    child.add_edge("private_llm_a", "private_llm_b")
    child_graph = child.compile(checkpointer=True if persist_private_subgraph else None)

    def call_private_subgraph(state: WrapperState, config) -> dict[str, Any]:
        result = child_graph.invoke(
            {
                "prompt": state["prompt"],
                "private_messages": [],
            },
            config,
        )
        return {"outputs": [result["private_messages"][-1]]}

    parent = StateGraph(WrapperState)
    parent.add_node("call_private_subgraph", call_private_subgraph)
    parent.add_edge(START, "call_private_subgraph")
    return parent.compile(checkpointer=InMemorySaver())


def _build_repo_style_engineer_graph():
    reviewer = StateGraph(RepoStyleWorkflowState)
    reviewer.add_node("review_plan", _repo_review_plan)
    reviewer.add_edge(START, "review_plan")
    reviewer_graph = reviewer.compile()

    engineer = StateGraph(RepoStyleWorkflowState)
    engineer.add_node("plan_work", _repo_plan_work)
    engineer.add_node("gameplay-reviewer-workflow", reviewer_graph)
    engineer.add_node("implement_code", _repo_implement_code)
    engineer.add_edge(START, "plan_work")
    engineer.add_edge("plan_work", "gameplay-reviewer-workflow")
    engineer.add_edge("gameplay-reviewer-workflow", "implement_code")
    return engineer.compile(checkpointer=InMemorySaver())


class ShortTermMemoryDemoTests(unittest.TestCase):
    def test_subgraph_nodes_share_state_within_one_invoke(self) -> None:
        graph = _build_shared_parent_graph()
        config = {"configurable": {"thread_id": "shared-one-shot"}}

        result = graph.invoke(
            {
                "prompt": "turn1",
                "messages": [],
                "seen_by_a": [],
                "seen_by_b": [],
            },
            config,
        )

        self.assertEqual(result["seen_by_a"], [])
        self.assertEqual(result["seen_by_b"], ["a<turn1> saw=0"])
        self.assertEqual(result["messages"], ["a<turn1> saw=0", "b<turn1> saw=1"])

    def test_same_thread_reuses_short_term_memory_across_invokes(self) -> None:
        graph = _build_shared_parent_graph()
        config = {"configurable": {"thread_id": "shared-across-invokes"}}

        graph.invoke(
            {
                "prompt": "turn1",
                "messages": [],
                "seen_by_a": [],
                "seen_by_b": [],
            },
            config,
        )
        result = graph.invoke(
            {
                "prompt": "turn2",
                "messages": [],
                "seen_by_a": [],
                "seen_by_b": [],
            },
            config,
        )

        self.assertEqual(result["seen_by_a"], ["a<turn1> saw=0", "b<turn1> saw=1"])
        self.assertEqual(
            result["seen_by_b"],
            ["a<turn1> saw=0", "b<turn1> saw=1", "a<turn2> saw=2"],
        )
        snapshot = graph.get_state(config)
        self.assertEqual(snapshot.values["seen_by_a"], ["a<turn1> saw=0", "b<turn1> saw=1"])
        self.assertEqual(
            snapshot.values["seen_by_b"],
            ["a<turn1> saw=0", "b<turn1> saw=1", "a<turn2> saw=2"],
        )

    def test_private_subgraph_state_resets_without_subgraph_checkpointer(self) -> None:
        graph = _build_wrapper_parent_graph(persist_private_subgraph=False)
        config = {"configurable": {"thread_id": "private-resets"}}

        first = graph.invoke({"prompt": "turn1", "outputs": []}, config)
        second = graph.invoke({"prompt": "turn2", "outputs": []}, config)

        self.assertEqual(first["outputs"][-1], "private-b<turn1> saw=1")
        self.assertEqual(second["outputs"][-1], "private-b<turn2> saw=1")

    def test_private_subgraph_state_persists_with_checkpointer_true(self) -> None:
        graph = _build_wrapper_parent_graph(persist_private_subgraph=True)
        config = {"configurable": {"thread_id": "private-persists"}}

        first = graph.invoke({"prompt": "turn1", "outputs": []}, config)
        second = graph.invoke({"prompt": "turn2", "outputs": []}, config)

        self.assertEqual(first["outputs"][-1], "private-b<turn1> saw=1")
        self.assertEqual(second["outputs"][-1], "private-b<turn2> saw=3")
        self.assertEqual(
            graph.get_state(config).values["outputs"],
            [
                "private-b<turn1> saw=1",
                "private-b<turn2> saw=3",
            ],
        )

    def test_repo_style_subgraph_shares_review_notes_with_implement_node(self) -> None:
        graph = _build_repo_style_engineer_graph()
        config = {"configurable": {"thread_id": "repo-style-one-shot"}}

        result = graph.invoke(
            {
                "task_prompt": "fix dodge cancel timing",
                "plan_doc": "",
                "review_round": 0,
                "review_notes": [],
                "reviewer_seen_notes": [],
                "implementer_seen_notes": [],
                "final_report": {},
            },
            config,
        )

        self.assertEqual(result["reviewer_seen_notes"], ["plan<fix dodge cancel timing>"])
        self.assertEqual(
            result["implementer_seen_notes"],
            [
                "plan<fix dodge cancel timing>",
                "plan<fix dodge cancel timing>",
                "review<fix dodge cancel timing> saw=1",
            ],
        )
        self.assertEqual(result["final_report"]["last_note"], "review<fix dodge cancel timing> saw=1")

    def test_repo_style_same_thread_reuses_review_memory_across_invokes(self) -> None:
        graph = _build_repo_style_engineer_graph()
        config = {"configurable": {"thread_id": "repo-style-across-invokes"}}

        graph.invoke(
            {
                "task_prompt": "fix dodge cancel timing",
                "plan_doc": "",
                "review_round": 0,
                "review_notes": [],
                "reviewer_seen_notes": [],
                "implementer_seen_notes": [],
                "final_report": {},
            },
            config,
        )
        result = graph.invoke(
            {
                "task_prompt": "tighten melee recovery windows",
                "plan_doc": "",
                "review_round": 0,
                "review_notes": [],
                "reviewer_seen_notes": [],
                "implementer_seen_notes": [],
                "final_report": {},
            },
            config,
        )

        self.assertEqual(
            result["reviewer_seen_notes"],
            [
                "plan<fix dodge cancel timing>",
                "plan<fix dodge cancel timing>",
                "review<fix dodge cancel timing> saw=1",
                "plan<tighten melee recovery windows>",
            ],
        )
        self.assertEqual(result["final_report"]["last_note"], "review<tighten melee recovery windows> saw=4")
        self.assertEqual(result["final_report"]["seen_count"], 9)


if __name__ == "__main__":
    unittest.main()
