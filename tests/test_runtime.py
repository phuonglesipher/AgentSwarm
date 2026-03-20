from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver

from core.config_loader import load_agentswarm_config, load_project_manifest
from core.host_setup import initialize_host_project
from core.runtime_paths import resolve_runtime_paths
from core.tool_loader import load_tools
from core.workflow_loader import load_workflows
from core.graph_logging import GRAPH_DEBUG_TRACE_FILE
from core.main_graph import build_initial_state, build_main_graph, build_runtime_config


def _investigation_payload(**overrides) -> dict:
    payload = {
        "doc_hits": [],
        "doc_context": "",
        "config_hits": [],
        "source_hits": [],
        "test_hits": [],
        "blueprint_hits": [],
        "blueprint_text_hits": [],
        "current_runtime_paths": [],
        "legacy_runtime_paths": [],
        "runtime_path_hypotheses": [],
        "ownership_summary": "",
        "investigation_summary": "",
        "code_context": "",
        "blueprint_context": "",
        "implementation_medium": "cpp",
        "implementation_medium_reason": "The strongest surviving evidence points to code-owned gameplay logic.",
        "bug_context_doc": "",
    }
    payload.update(overrides)
    return payload


def _plan_review_payload(
    *,
    section_rows: list[tuple[str, int, str, str, list[str]]],
    feedback: str,
    blocking_issues: list[str],
    improvement_actions: list[str],
    approved: bool,
) -> dict:
    return {
        "score": sum(score for _, score, _, _, _ in section_rows),
        "feedback": feedback,
        "missing_sections": [section for section, _, status, _, _ in section_rows if status == "missing"],
        "section_reviews": [
            {
                "section": section,
                "score": score,
                "status": status,
                "rationale": rationale,
                "action_items": action_items,
            }
            for section, score, status, rationale, action_items in section_rows
        ],
        "blocking_issues": blocking_issues,
        "improvement_actions": improvement_actions,
        "approved": approved,
    }


def _investigation_review_payload(
    *,
    section_rows: list[tuple[str, int, str, str, list[str]]],
    feedback: str,
    blocking_issues: list[str],
    improvement_actions: list[str],
    approved: bool,
) -> dict:
    return {
        "feedback": feedback,
        "section_reviews": [
            {
                "section": section,
                "score": score,
                "status": status,
                "rationale": rationale,
                "action_items": action_items,
            }
            for section, score, status, rationale, action_items in section_rows
        ],
        "blocking_issues": blocking_issues,
        "improvement_actions": improvement_actions,
        "approved": approved,
    }


def _approved_investigation_review_payload(feedback: str = "The investigation is technically ready.") -> dict:
    return _investigation_review_payload(
        section_rows=[
            ("Supporting References", 10, "pass", "Grounded references support the investigation.", []),
            ("Runtime Owner Precision", 25, "pass", "The live runtime owner is identified precisely.", []),
            ("Current vs Legacy Split", 10, "pass", "Current ownership is separated from stale references.", []),
            ("Ownership Summary", 10, "pass", "Ownership is summarized clearly enough for handoff.", []),
            ("Root Cause Hypothesis", 15, "pass", "The causal hypothesis is concrete and technically grounded.", []),
            ("Investigation Summary", 10, "pass", "The investigation summary is handoff-ready.", []),
            ("Implementation Medium", 5, "pass", "The implementation medium is justified.", []),
            ("Validation Plan", 10, "pass", "The validation path is concrete.", []),
            ("Noise Control", 5, "pass", "The evidence stays focused on the live owner.", []),
        ],
        feedback=feedback,
        blocking_issues=[],
        improvement_actions=[],
        approved=True,
    )


class DisabledLLMClient:
    def is_enabled(self) -> bool:
        return False

    def describe(self) -> str:
        return "disabled test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        raise AssertionError("LLM should not be called in deterministic tests")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        raise AssertionError("LLM should not be called in deterministic tests")


class FakeLLMManager:
    def __init__(self) -> None:
        self._client = DisabledLLMClient()

    def resolve(self, profile: str | None = None) -> DisabledLLMClient:
        return self._client

    def is_enabled(self, profile: str | None = None) -> bool:
        return False

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: disabled test client"

    def available_profiles(self) -> list[str]:
        return ["default"]


class AlwaysBadLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "always bad test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del instructions, input_text, effort
        return "Need more information before I can produce the required plan."

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, input_text, schema, effort
        if schema_name == "gameplay_task_classification":
            return {
                "task_type": "feature",
                "reason": "Intentionally returning a non-actionable feature classification for review-cap testing.",
            }
        if schema_name == "gameplay_engineering_context":
            return _investigation_payload(
                doc_hits=["Workflows/gameplay-engineer-workflow/Workflow.md"],
                doc_context="The workflow metadata is the only reliable document in this fixture.",
                source_hits=["Workflows/gameplay-engineer-workflow/entry.py"],
                current_runtime_paths=["Workflows/gameplay-engineer-workflow/entry.py"],
                runtime_path_hypotheses=["The workflow entry file is the only concrete runtime owner available in this test fixture."],
                ownership_summary="The review-cap fixture anchors on the workflow entry file because no richer host project exists.",
                investigation_summary="Investigation is intentionally minimal so the test can focus on the review loop cap.",
                code_context="Workflow behavior under test lives in Workflows/gameplay-engineer-workflow/entry.py.",
                implementation_medium="cpp",
                implementation_medium_reason="Return code-side execution so the test can focus on the feature review loop.",
            )
        if schema_name == "gameplay_investigation_review":
            return _approved_investigation_review_payload(
                feedback="Investigation quality is sufficient, so the test can focus on the plan-review failure path."
            )
        if schema_name == "gameplay_plan_review":
            return _plan_review_payload(
                section_rows=[
                    ("Overview", 4, "needs-work", "The overview is too vague to trust.", ["Clarify the player-visible gameplay goal."]),
                    ("Task Type", 4, "needs-work", "The task framing is still too thin.", ["Explain why this work belongs on the chosen gameplay track."]),
                    ("Existing Docs", 2, "needs-work", "Grounding evidence is still too weak.", ["Cite the live design, runtime, or test references."]),
                    ("Implementation Steps", 6, "needs-work", "Implementation steps are not grounded in the runtime owner.", ["Name the owning runtime file and the exact change hook."]),
                    ("Unit Tests", 2, "needs-work", "Regression coverage is still undefined.", ["Specify the exact automated regression checks."]),
                    ("Risks", 2, "needs-work", "Risks and mitigations are not actionable yet.", ["Name the main gameplay regression risk and mitigation."]),
                    ("Acceptance Criteria", 2, "needs-work", "Acceptance criteria are not player-visible enough.", ["Write player-visible acceptance criteria."]),
                ],
                feedback="The plan is still incomplete and never becomes implementation-ready in this test.",
                blocking_issues=[
                    "Implementation Steps: Name the owning runtime file and the exact change hook.",
                    "Unit Tests: Specify the exact automated regression checks.",
                    "Acceptance Criteria: Write player-visible acceptance criteria.",
                ],
                improvement_actions=[
                    "Clarify the player-visible gameplay goal.",
                    "Name the owning runtime file and the exact change hook.",
                    "Specify the exact automated regression checks.",
                    "Write player-visible acceptance criteria.",
                ],
                approved=False,
            )
        return {
            "score": 0,
            "feedback": "The plan is incomplete and still missing required sections.",
            "missing_sections": [],
            "approved": False,
        }


class AlwaysBadLLMManager:
    def __init__(self) -> None:
        self._client = AlwaysBadLLMClient()

    def resolve(self, profile: str | None = None) -> AlwaysBadLLMClient:
        return self._client

    def is_enabled(self, profile: str | None = None) -> bool:
        return True

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: always bad test client"

    def available_profiles(self) -> list[str]:
        return ["default"]


class AlmostApprovedLLMClient:
    def __init__(self) -> None:
        self._review_calls = 0

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "almost approved test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort
        feature_prompt = "add " in input_text.lower() or "new feature" in input_text.lower()
        if "Write a concise markdown design context document" in instructions:
            return "\n".join(
                [
                    "# Gameplay Design Context",
                    "",
                    "## Overview",
                    (
                        "- Player-facing feature goal is clearly identified."
                        if feature_prompt
                        else "- Player-facing issue is clearly identified."
                    ),
                    "",
                    "## Existing References",
                    "- docs/designer/combat_design_template.md",
                ]
            )
        if "Write a concise markdown bug investigation brief" in instructions:
            return "\n".join(
                [
                    "# Gameplay Bug Context",
                    "",
                    "## Bug Summary",
                    "- Repro and likely failing gameplay state are identified.",
                    "",
                    "## Current Signals",
                    "- src/combat_dodge.py",
                ]
            )
        if (
            "Produce a markdown implementation plan" in instructions
            or "Produce a markdown architecture and implementation plan" in instructions
            or "Rewrite the full markdown implementation plan" in instructions
        ):
            task_type = "feature" if feature_prompt else "bugfix"
            task_reason = (
                "- Classification reason: this is a feature because it adds new gameplay behavior that needs plan approval."
                if feature_prompt
                else "- Classification reason: this is a bugfix because it restores intended gameplay behavior."
            )
            return "\n".join(
                [
                    "# Gameplay Implementation Plan",
                    "",
                    "## Overview",
                    (
                        "- Add a new melee combo extension that preserves existing responsiveness."
                        if feature_prompt
                        else "- Restore the intended dodge cancel timing for melee gameplay."
                    ),
                    (
                        "- Keep nearby responsiveness and state transitions stable for players."
                        if not feature_prompt
                        else "- Keep adjacent combat states readable for design review and implementation."
                    ),
                    "",
                    "## Task Type",
                    f"- {task_type}",
                    task_reason,
                    "",
                    "## Existing Docs",
                    "- docs/designer/combat_design_template.md",
                    "- docs/programming/gameplay_runtime.md",
                    "",
                    "## Implementation Steps",
                    (
                        "- Inspect src/combat_combo.py for the melee combo state machine and the extension points for the new branch."
                        if feature_prompt
                        else "- Inspect src/combat_dodge.py for the dodge cancel state machine and the melee timing gates."
                    ),
                    (
                        "- Add the new branch in src/combat_combo.py while preserving valid transition order."
                        if feature_prompt
                        else "- Update the timing logic in src/combat_dodge.py while preserving valid transition order."
                    ),
                    "- Add debug breadcrumbs so regressions are easy to spot.",
                    "",
                    "## Unit Tests",
                    (
                        "- Add an automated regression test for the new melee combo branch."
                        if feature_prompt
                        else "- Add an automated regression test for melee dodge cancel timing."
                    ),
                    (
                        "- Assert the new branch unlocks only from the expected combat state."
                        if feature_prompt
                        else "- Assert the expected state transition and responsiveness window."
                    ),
                    "",
                    "## Risks",
                    "- Risk: adjacent cancel windows may drift if timing hooks change.",
                    "- Mitigation: keep validation and debug logging around the transition gates.",
                    "",
                    "## Acceptance Criteria",
                    (
                        "- Players should be able to enter the new melee combo branch from the approved combat window."
                        if feature_prompt
                        else "- Players should be able to dodge cancel melee actions at the intended timing window."
                    ),
                    "- Regression checks for neighboring melee states should still pass.",
                ]
            )
        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, schema, effort
        feature_prompt = "add " in input_text.lower() or "new feature" in input_text.lower()
        if schema_name == "gameplay_task_classification":
            return {
                "task_type": "feature" if feature_prompt else "bugfix",
                "reason": (
                    "The prompt adds new gameplay behavior and should go through the feature track."
                    if feature_prompt
                    else "The prompt is clearly about fixing unintended gameplay behavior."
                ),
            }
        if schema_name == "gameplay_engineering_context":
            return _investigation_payload(
                doc_hits=["Workflows/gameplay-engineer-workflow/Workflow.md"],
                doc_context="The workflow covers gameplay implementation, bug fixing, and planning.",
                source_hits=["Workflows/gameplay-engineer-workflow/entry.py"],
                test_hits=["tests/test_runtime.py"],
                current_runtime_paths=["Workflows/gameplay-engineer-workflow/entry.py"],
                runtime_path_hypotheses=["The workflow entry graph is the current owned runtime path for this test fixture."],
                ownership_summary="The workflow behavior under test is owned by Workflows/gameplay-engineer-workflow/entry.py.",
                investigation_summary="The test fixture converged on the workflow entry file and runtime tests.",
                code_context="Gameplay workflow logic is implemented in Workflows/gameplay-engineer-workflow/entry.py.",
                implementation_medium="cpp",
                implementation_medium_reason="The test fixture is source-owned and validated by Python runtime tests.",
            )
        if schema_name == "gameplay_investigation_review":
            return _approved_investigation_review_payload(
                feedback="Investigation quality is strong enough to move into the plan-review loop."
            )
        if schema_name == "gameplay_plan_review":
            self._review_calls += 1
            task_type_score = 5 if self._review_calls == 1 else 10
            task_type_status = "needs-work" if self._review_calls == 1 else "pass"
            task_type_rationale = (
                "The plan names the task type but does not fully justify it."
                if self._review_calls == 1
                else "The task type rationale is now explicit enough for approval."
            )
            task_type_actions = (
                [
                    (
                        "Add one sentence justifying why this work is classified as a feature."
                        if feature_prompt
                        else "Add one sentence justifying why this work is classified as a bugfix."
                    )
                ]
                if self._review_calls == 1
                else []
            )
            risks_score = 10 if self._review_calls == 1 else 5
            risks_status = "pass" if self._review_calls == 1 else "needs-work"
            risks_rationale = (
                "Risks and mitigations are documented."
                if self._review_calls == 1
                else "A small residual implementation risk remains, but it is not blocking."
            )
            risks_actions = [] if self._review_calls == 1 else ["Monitor adjacent cancel windows during implementation review."]
            return {
                "score": 95,
                "feedback": (
                    "The plan is implementation-ready. Only the task type rationale could be more explicit."
                    if self._review_calls == 1
                    else "The plan is implementation-ready and the remaining risk is non-blocking."
                ),
                "missing_sections": [],
                "section_reviews": [
                    {
                        "section": "Overview",
                        "score": 10,
                        "status": "pass",
                        "rationale": "The gameplay goal and scope are clear.",
                        "action_items": [],
                    },
                    {
                        "section": "Task Type",
                        "score": task_type_score,
                        "status": task_type_status,
                        "rationale": task_type_rationale,
                        "action_items": task_type_actions,
                    },
                    {
                        "section": "Existing Docs",
                        "score": 10,
                        "status": "pass",
                        "rationale": "Relevant gameplay references are listed.",
                        "action_items": [],
                    },
                    {
                        "section": "Implementation Steps",
                        "score": 25,
                        "status": "pass",
                        "rationale": "Implementation sequencing is concrete and actionable.",
                        "action_items": [],
                    },
                    {
                        "section": "Unit Tests",
                        "score": 20,
                        "status": "pass",
                        "rationale": "Automated regression coverage is specified.",
                        "action_items": [],
                    },
                    {
                        "section": "Risks",
                        "score": risks_score,
                        "status": risks_status,
                        "rationale": risks_rationale,
                        "action_items": risks_actions,
                    },
                    {
                        "section": "Acceptance Criteria",
                        "score": 15,
                        "status": "pass",
                        "rationale": "Player-visible outcomes and regression checks are explicit.",
                        "action_items": [],
                    },
                ],
                "blocking_issues": [],
                "improvement_actions": [*task_type_actions, *risks_actions],
                "approved": self._review_calls >= 2,
            }
        raise AssertionError(f"Unexpected schema_name: {schema_name}")


class MixedLLMManager:
    def __init__(self) -> None:
        self._default_client = AlmostApprovedLLMClient()
        self._codegen_client = DisabledLLMClient()

    def resolve(self, profile: str | None = None):
        if profile == "codegen":
            return self._codegen_client
        return self._default_client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default", "reviewer", "codegen"]


class RepairAwareCodegenLLMClient:
    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.calls = 0

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "repair-aware codegen test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        raise AssertionError("Repair-aware codegen should not call generate_text")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, schema, effort
        if schema_name != "gameplay_code_bundle":
            raise AssertionError(f"Unexpected schema_name: {schema_name}")
        self.calls += 1
        self.inputs.append(input_text)
        if self.calls == 1:
            return {
                "source_code": "\n".join(
                    [
                        "class PlayerCharacter:",
                        "    def __init__(self):",
                        "        self.can_move = False",
                        "",
                        "    def spawn(self):",
                        "        return self",
                        "",
                        "    def move(self):",
                        "        return self.can_move",
                    ]
                ),
                "test_code": "\n".join(
                    [
                        "from src.player_movement import PlayerCharacter",
                        "",
                        "def test_player_can_move_after_spawn():",
                        "    player = PlayerCharacter().spawn()",
                        "    assert player.move() is True, 'repair me'",
                    ]
                ),
                "implementation_notes": "First attempt keeps the failing behavior so the repair loop has concrete feedback.",
            }
        return {
            "source_code": "\n".join(
                [
                    "class PlayerCharacter:",
                    "    def __init__(self):",
                    "        self.can_move = False",
                    "",
                    "    def spawn(self):",
                    "        self.can_move = True",
                    "        return self",
                    "",
                    "    def move(self):",
                    "        return self.can_move",
                ]
            ),
            "test_code": "\n".join(
                [
                    "from src.player_movement import PlayerCharacter",
                    "",
                    "def test_player_can_move_after_spawn():",
                    "    player = PlayerCharacter().spawn()",
                    "    assert player.move() is True",
                ]
            ),
            "implementation_notes": "Second attempt fixes the spawn path after reading the failing self-test output.",
        }


class RepairAwareCodegenLLMManager:
    def __init__(self) -> None:
        self._default_client = RepairDefaultLLMClient()
        self._codegen_client = RepairAwareCodegenLLMClient()

    @property
    def client(self) -> RepairAwareCodegenLLMClient:
        return self._codegen_client

    def resolve(self, profile: str | None = None):
        if profile == "codegen":
            return self._codegen_client
        return self._default_client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default", "codegen"]


class ContextOnlyInvestigationLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "context-only investigation test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort, input_text
        if "Write a concise markdown bug investigation brief" in instructions:
            return "\n".join(
                [
                    "# Gameplay Bug Context",
                    "",
                    "## Bug Summary",
                    "- Existing code context suggests the gameplay issue lives in combat dodge state handling.",
                ]
            )
        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, input_text, schema, effort
        if schema_name == "gameplay_task_classification":
            return {
                "task_type": "bugfix",
                "reason": "The prompt fixes unintended gameplay behavior.",
            }
        if schema_name == "gameplay_engineering_context":
            return _investigation_payload(
                code_context="Potential ownership is around src/combat_dodge.py and its regression tests.",
                ownership_summary="The issue likely belongs to src/combat_dodge.py once local fallback resolves the exact path.",
                investigation_summary="Code ownership is clearer than Blueprint ownership in this fixture.",
            )
        if schema_name == "gameplay_investigation_review":
            return _approved_investigation_review_payload(
                feedback="The investigation is grounded enough to continue past the review gate."
            )
        raise AssertionError(f"Unexpected schema_name: {schema_name}")


class ContextOnlyInvestigationLLMManager:
    def __init__(self) -> None:
        self._default_client = ContextOnlyInvestigationLLMClient()
        self._codegen_client = DisabledLLMClient()

    def resolve(self, profile: str | None = None):
        if profile == "codegen":
            return self._codegen_client
        return self._default_client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default", "codegen"]


class InvestigationRetryLLMClient:
    def __init__(self) -> None:
        self._investigation_calls = 0
        self._review_calls = 0

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "investigation retry test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort, input_text
        if "Write a concise markdown bug investigation brief" in instructions:
            return "\n".join(
                [
                    "# Gameplay Bug Context",
                    "",
                    "## Bug Summary",
                    "- Investigation converged on the runtime module after one retry.",
                ]
            )
        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, input_text, schema, effort
        if schema_name == "gameplay_task_classification":
            return {
                "task_type": "bugfix",
                "reason": "The request is about fixing unintended movement behavior.",
            }
        if schema_name == "gameplay_engineering_context":
            self._investigation_calls += 1
            if self._investigation_calls == 1:
                return _investigation_payload(
                    source_hits=["Scripts/runtime.py"],
                    test_hits=["Validation/test_runtime.py"],
                    legacy_runtime_paths=["Scripts/runtime.py"],
                    runtime_path_hypotheses=["The only surviving path still looks legacy or lower-confidence and needs a cleaner current owner."],
                    ownership_summary="Scripts/runtime.py is a candidate, but the first pass has not separated current ownership from stale evidence yet.",
                    investigation_summary="The first investigation pass still needs a clearer current-vs-legacy split before implementation.",
                    code_context="A runtime module exists, but the ownership split is still ambiguous on the first pass.",
                    implementation_medium="cpp",
                    implementation_medium_reason="The available evidence is code-side, but the current owner is still ambiguous.",
                )
            return _investigation_payload(
                source_hits=["Scripts/runtime.py"],
                test_hits=["Validation/test_runtime.py"],
                current_runtime_paths=["Scripts/runtime.py"],
                runtime_path_hypotheses=["Movement ownership likely lives in Scripts/runtime.py, with Validation/test_runtime.py validating the path."],
                ownership_summary="Movement ownership appears to live in Scripts/runtime.py and its regression tests.",
                investigation_summary="The second investigation round converged on the runtime module and its tests.",
                code_context="Movement ownership appears to live in Scripts/runtime.py and its regression tests.",
                implementation_medium="cpp",
                implementation_medium_reason="The strongest surviving evidence points to code-side movement logic.",
            )
        if schema_name == "gameplay_investigation_review":
            self._review_calls += 1
            if self._review_calls == 1:
                return _investigation_review_payload(
                    section_rows=[
                        ("Supporting References", 10, "pass", "Grounded references support the investigation.", []),
                        ("Runtime Owner Precision", 12, "needs-work", "The live runtime owner still needs a cleaner current path.", ["Isolate the current runtime owner."]),
                        ("Current vs Legacy Split", 4, "needs-work", "Legacy references still blur current ownership.", ["Separate current ownership from stale evidence."]),
                        ("Ownership Summary", 10, "pass", "Ownership direction is plausible.", []),
                        ("Root Cause Hypothesis", 10, "needs-work", "The causal chain still needs a clearer live owner.", ["Tie the hypothesis to the live owner."]),
                        ("Investigation Summary", 10, "pass", "The summary captures the remaining gap.", []),
                        ("Implementation Medium", 5, "pass", "Code ownership is plausible.", []),
                        ("Validation Plan", 4, "needs-work", "Validation is still too vague.", ["Name the exact validation path."]),
                        ("Noise Control", 5, "pass", "The investigation stayed focused.", []),
                    ],
                    feedback="Round 1 still needs a stronger current-vs-legacy split and a concrete validation path.",
                    blocking_issues=[
                        "Runtime Owner Precision: Isolate the current runtime owner.",
                        "Current vs Legacy Split: Separate current ownership from stale evidence.",
                        "Validation Plan: Name the exact validation path.",
                    ],
                    improvement_actions=[
                        "Isolate the current runtime owner.",
                        "Separate current ownership from stale evidence.",
                        "Name the exact validation path.",
                    ],
                    approved=False,
                )
            return _approved_investigation_review_payload(
                feedback="The second investigation round is grounded enough to move forward."
            )
        raise AssertionError(f"Unexpected schema_name: {schema_name}")


class InvestigationRetryLLMManager:
    def __init__(self) -> None:
        self._default_client = InvestigationRetryLLMClient()
        self._codegen_client = DisabledLLMClient()

    def resolve(self, profile: str | None = None):
        if profile == "codegen":
            return self._codegen_client
        return self._default_client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default", "codegen"]


class InvestigationLearningCarryForwardLLMClient:
    def __init__(self) -> None:
        self.strategy_inputs: list[str] = []
        self._strategy_calls = 0
        self._investigation_calls = 0
        self._review_calls = 0

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "investigation learning carry-forward test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort, input_text
        if "Write a concise markdown bug investigation brief" in instructions:
            return "\n".join(
                [
                    "# Gameplay Bug Context",
                    "",
                    "## Bug Summary",
                    "- The investigation notebook carried forward the final runtime owner and validation path.",
                ]
            )
        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, schema, effort
        if schema_name == "gameplay_task_classification":
            return {
                "task_type": "bugfix",
                "reason": "The prompt is a bug investigation and should stay on the bugfix track.",
            }
        if schema_name == "gameplay_investigation_strategy":
            self._strategy_calls += 1
            self.strategy_inputs.append(input_text)
            if self._strategy_calls == 1:
                return {
                    "focus_terms": ["player", "movement", "owner"],
                    "avoid_terms": ["archive"],
                    "search_notes": ["Reject stale docs and isolate the live runtime owner."],
                    "implementation_medium_hint": "cpp",
                    "implementation_medium_reason": "The gameplay behavior still looks code-owned.",
                    "investigation_root_cause": "",
                    "investigation_validation_plan": "",
                }
            return {
                "focus_terms": ["player", "movement", "runtime"],
                "avoid_terms": ["archive", "legacy"],
                "search_notes": [
                    "Keep the surviving movement signal from the previous round.",
                    "Avoid the rejected archive note and isolate the live runtime owner.",
                ],
                "implementation_medium_hint": "cpp",
                "implementation_medium_reason": "The first round narrowed the task to code-owned movement logic.",
                "investigation_root_cause": "The player movement lock likely persists in Gameplay/player_movement.py.",
                "investigation_validation_plan": "Validate with Checks/test_player_movement.py.",
            }
        if schema_name == "gameplay_engineering_context":
            self._investigation_calls += 1
            if self._investigation_calls == 1:
                return _investigation_payload(
                    doc_hits=["docs/archive/player_movement_notes.md"],
                    doc_context="Archived movement notes describe a past fix but do not name the current runtime owner.",
                    legacy_runtime_paths=["docs/archive/player_movement_notes.md"],
                    runtime_path_hypotheses=["The archive note is stale and the live runtime owner still needs to be isolated."],
                    ownership_summary="The first pass only found an archive movement note, so current ownership is still ambiguous.",
                    investigation_summary="The first pass surfaced a stale archive note and needs a second round.",
                    code_context="No trusted runtime file was isolated on the first pass.",
                    implementation_medium="cpp",
                    implementation_medium_reason="Movement behavior still looks code-owned once the live file is identified.",
                )
            return _investigation_payload(
                source_hits=["Gameplay/player_movement.py"],
                test_hits=["Checks/test_player_movement.py"],
                current_runtime_paths=["Gameplay/player_movement.py"],
                runtime_path_hypotheses=[
                    "Gameplay/player_movement.py owns the runtime movement lock and Checks/test_player_movement.py validates the path."
                ],
                ownership_summary="Gameplay/player_movement.py now looks like the live movement owner, with Checks/test_player_movement.py covering the regression path.",
                investigation_summary="The second round carried forward the useful movement signal, discarded the archive note, and isolated the live owner.",
                code_context="Gameplay/player_movement.py is the live movement owner and Checks/test_player_movement.py validates the regression path.",
                implementation_medium="cpp",
                implementation_medium_reason="The strongest surviving evidence points to code-owned movement logic.",
            )
        if schema_name == "gameplay_investigation_review":
            self._review_calls += 1
            if self._review_calls == 1:
                return _investigation_review_payload(
                    section_rows=[
                        ("Supporting References", 6, "needs-work", "The first pass leaned on an archive note instead of current references.", ["Replace the archive note with live runtime references."]),
                        ("Runtime Owner Precision", 10, "needs-work", "The live movement owner is still too ambiguous.", ["Identify the live runtime owner."]),
                        ("Current vs Legacy Split", 3, "needs-work", "Archive evidence still dominates the brief.", ["Discard the archive note and keep only live evidence."]),
                        ("Ownership Summary", 8, "needs-work", "Ownership summary still needs a stronger runtime owner.", ["Name the live owner and why it owns the bug."]),
                        ("Root Cause Hypothesis", 10, "needs-work", "The hypothesis exists but is not tied tightly enough to the live owner.", ["Tie the hypothesis to the live owner."]),
                        ("Investigation Summary", 8, "needs-work", "The summary still needs a stronger runtime owner.", ["Tighten the investigation summary around the live owner."]),
                        ("Implementation Medium", 5, "pass", "Code ownership is still plausible.", []),
                        ("Validation Plan", 5, "needs-work", "The validation path still needs the concrete regression test.", ["Name the concrete validation path."]),
                        ("Noise Control", 2, "needs-work", "Rejected archive evidence is still polluting the handoff.", ["Keep rejected archive evidence out of the next pass."]),
                    ],
                    feedback="Round 1 found useful signal, but the next pass must carry forward only the live runtime owner and discard the archive note.",
                    blocking_issues=[
                        "Runtime Owner Precision: Identify the live runtime owner.",
                        "Current vs Legacy Split: Discard the archive note and keep only live evidence.",
                        "Validation Plan: Name the concrete validation path.",
                    ],
                    improvement_actions=[
                        "Identify the live runtime owner.",
                        "Discard the archive note and keep only live evidence.",
                        "Name the concrete validation path.",
                    ],
                    approved=False,
                )
            return _approved_investigation_review_payload(
                feedback="The second investigation round carried forward the right evidence and is ready for handoff."
            )
        raise AssertionError(f"Unexpected schema_name: {schema_name}")


class InvestigationLearningCarryForwardLLMManager:
    def __init__(self) -> None:
        self._default_client = InvestigationLearningCarryForwardLLMClient()
        self._codegen_client = DisabledLLMClient()

    @property
    def client(self) -> InvestigationLearningCarryForwardLLMClient:
        return self._default_client

    def resolve(self, profile: str | None = None):
        if profile == "codegen":
            return self._codegen_client
        return self._default_client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default", "codegen"]


class StrictLoopingFeatureLLMClient:
    def __init__(self) -> None:
        self.strategy_inputs: list[str] = []
        self.plan_inputs: list[str] = []
        self.review_inputs: list[str] = []
        self._strategy_calls = 0
        self._investigation_calls = 0
        self._plan_calls = 0
        self._review_calls = 0

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "strict looping feature test client"

    def _plan_doc(self, stage: int) -> str:
        if stage <= 1:
            return "\n".join(
                [
                    "# Gameplay Implementation Plan",
                    "",
                    "## Overview",
                    "- Add wall-jump air dash recharge.",
                    "",
                    "## Task Type",
                    "- feature",
                    "",
                    "## Existing Docs",
                    "- docs/designer/air_dash_recharge.md",
                    "",
                    "## Implementation Steps",
                    "- Update air dash runtime.",
                    "",
                    "## Unit Tests",
                    "- Add one test.",
                    "",
                    "## Risks",
                    "- Risk: state bugs.",
                    "",
                    "## Acceptance Criteria",
                    "- Player can air dash more.",
                ]
            )
        if stage == 2:
            return "\n".join(
                [
                    "# Gameplay Implementation Plan",
                    "",
                    "## Overview",
                    "- Add a wall-jump recharge rule so the player regains one air dash charge after a successful wall jump.",
                    "- Keep nearby aerial movement states readable so recharge only happens on the intended transition.",
                    "",
                    "## Task Type",
                    "- feature",
                    "- This adds new player-facing movement behavior that must be approved before implementation.",
                    "",
                    "## Existing Docs",
                    "- docs/designer/air_dash_recharge.md",
                    "- docs/programming/air_dash_runtime.md",
                    "",
                    "## Implementation Steps",
                    "- Trace the wall jump transition and the existing air dash charge bookkeeping in src/traversal_runtime.py.",
                    "- Add a dedicated recharge hook for successful wall jumps and keep the charge clamped to the configured maximum.",
                    "- Add state logging around wall-jump recharge so regressions in aerial transitions are easy to debug.",
                    "",
                    "## Unit Tests",
                    "- Add a regression test proving a wall jump restores one air dash charge after the player has spent one in air.",
                    "- Add a regression test proving repeated wall jumps cannot overfill the charge cap.",
                    "",
                    "## Risks",
                    "- Risk: recharge may trigger on unrelated collision states if wall-jump ownership is too broad.",
                    "- Mitigation: gate the recharge through the confirmed wall-jump path and log the transition during validation.",
                    "",
                    "## Acceptance Criteria",
                    "- Players regain exactly one charge after a valid wall jump.",
                    "- The change does not exceed the max air dash charge cap.",
                ]
            )
        return "\n".join(
            [
                "# Gameplay Implementation Plan",
                "",
                "## Overview",
                "- Add a wall-jump recharge rule so the player regains exactly one air dash charge after a successful wall jump.",
                "- Keep aerial traversal readable by limiting recharge to the confirmed wall-jump transition and preserving the existing charge cap.",
                "",
                "## Task Type",
                "- feature",
                "- This adds a new player-facing traversal rule, so it requires architecture approval before implementation.",
                "",
                "## Existing Docs",
                "- docs/designer/air_dash_recharge.md",
                "- docs/programming/air_dash_runtime.md",
                "",
                "## Implementation Steps",
                "- Confirm `src/traversal_runtime.py` owns both charge spending and the wall-jump transition that can safely grant a recharge.",
                "- Add a narrow recharge path that fires only after a valid wall jump and restores one charge without exceeding `max_charges`.",
                "- Preserve adjacent aerial states by keeping the recharge logic out of generic landing or collision handlers and add lightweight state breadcrumbs for recharge grants.",
                "",
                "## Unit Tests",
                "- Add a regression test proving a player who has spent one charge regains exactly one charge after a valid wall jump.",
                "- Add a regression test proving repeated wall jumps cannot push `air_dash_charges` above `max_charges`.",
                "- Add a regression test proving unrelated aerial updates do not restore charges when no wall jump occurred.",
                "",
                "## Risks",
                "- Risk: the recharge hook could fire from the wrong aerial transition and silently over-grant mobility.",
                "- Mitigation: bind the recharge to the explicit wall-jump transition and cover the neighboring non-wall-jump path with tests.",
                "",
                "## Acceptance Criteria",
                "- After spending one charge, the player regains exactly one air dash charge when a valid wall jump succeeds.",
                "- Repeating the wall jump flow never exceeds the configured max charge cap.",
                "- Nearby aerial traversal paths that are not wall jumps continue to leave charges unchanged.",
            ]
        )

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort
        if "Write a concise markdown design context document" in instructions:
            return "\n".join(
                [
                    "# Gameplay Design Context",
                    "",
                    "## Overview",
                    "- Players should regain one air dash charge after a valid wall jump.",
                    "",
                    "## Existing References",
                    "- docs/designer/air_dash_recharge.md",
                    "",
                    "## Player-Facing Behavior",
                    "- The recharge should feel deterministic and never exceed the configured cap.",
                    "",
                    "## Technical Notes",
                    "- Runtime ownership lives in src/traversal_runtime.py and validation should stay in tests/test_traversal_runtime.py.",
                    "",
                    "## Risks",
                    "- Nearby aerial states could accidentally receive recharge if the transition hook is too broad.",
                ]
            )
        if (
            "Produce a markdown architecture and implementation plan" in instructions
            or "Rewrite the full markdown implementation plan" in instructions
        ):
            self._plan_calls += 1
            self.plan_inputs.append(input_text)
            return self._plan_doc(min(self._plan_calls, 3))
        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, schema, effort
        if schema_name == "gameplay_task_classification":
            return {
                "task_type": "feature",
                "reason": "The prompt adds new traversal behavior, so it belongs on the feature track.",
            }
        if schema_name == "gameplay_investigation_strategy":
            self._strategy_calls += 1
            self.strategy_inputs.append(input_text)
            if self._strategy_calls == 1:
                return {
                    "focus_terms": ["air dash", "wall jump", "charge"],
                    "avoid_terms": ["archive", "demo"],
                    "search_notes": ["Isolate the live charge owner before discussing implementation."],
                    "implementation_medium_hint": "cpp",
                    "implementation_medium_reason": "The feature appears code-owned, but the live owner still needs a current-vs-legacy split.",
                    "investigation_root_cause": "",
                    "investigation_validation_plan": "",
                }
            if self._strategy_calls == 2:
                return {
                    "focus_terms": ["air dash", "wall jump", "charge cap"],
                    "avoid_terms": ["archive", "demo", "legacy"],
                    "search_notes": [
                        "Keep the live runtime owner from the previous round and stop anchoring on older notes.",
                        "Promote the runtime owner, then identify the missing validation path.",
                    ],
                    "implementation_medium_hint": "cpp",
                    "implementation_medium_reason": "The task is code-owned once the live wall-jump recharge owner is isolated.",
                    "investigation_root_cause": "Wall-jump recharge likely belongs in the live air dash runtime owner instead of generic movement helpers.",
                    "investigation_validation_plan": "",
                }
            return {
                "focus_terms": ["air dash", "wall jump", "recharge", "charge cap"],
                "avoid_terms": ["archive", "demo", "legacy"],
                "search_notes": [
                    "Carry forward the live runtime owner and finalize the automated validation path before implementation.",
                ],
                "implementation_medium_hint": "cpp",
                "implementation_medium_reason": "The strongest surviving evidence points to code-owned traversal logic.",
                "investigation_root_cause": "Recharge should happen in the live wall-jump transition after a spent charge, with the result clamped to max_charges.",
                "investigation_validation_plan": "Validate with tests/test_traversal_runtime.py, covering recharge and cap behavior.",
            }
        if schema_name == "gameplay_engineering_context":
            self._investigation_calls += 1
            if self._investigation_calls == 1:
                return _investigation_payload(
                    doc_hits=["docs/designer/air_dash_recharge.md"],
                    doc_context="Wall jumps should restore one air dash charge, but the live runtime owner is still ambiguous on the first pass.",
                    runtime_path_hypotheses=[
                        "The live recharge owner still needs to be isolated from the wall-jump design intent before planning."
                    ],
                    ownership_summary="The first pass only confirmed the design intent, so the live runtime owner still needs another investigation pass.",
                    investigation_summary="The first investigation pass found a plausible owner but still lacks enough confidence to hand off implementation.",
                    code_context="The first pass still lacks a trustworthy runtime owner snippet.",
                    implementation_medium="cpp",
                    implementation_medium_reason="The available evidence points to code-owned traversal logic, but the current owner is still under-evidenced.",
                )
            if self._investigation_calls == 2:
                return _investigation_payload(
                    doc_hits=["docs/designer/air_dash_recharge.md"],
                    doc_context="Wall jumps should restore one charge, and the charge should remain capped.",
                    source_hits=["src/traversal_runtime.py"],
                    test_hits=["tests/test_traversal_runtime.py"],
                    current_runtime_paths=["src/traversal_runtime.py"],
                    legacy_runtime_paths=["docs/designer/air_dash_recharge.md"],
                    runtime_path_hypotheses=[
                        "src/traversal_runtime.py owns the live air dash charge bookkeeping, but the final validation path still needs to be made explicit."
                    ],
                    ownership_summary="src/traversal_runtime.py is the live owner for wall-jump recharge, but the validation handoff is still incomplete.",
                    investigation_summary="The second pass isolated the live runtime owner and eliminated the main ownership ambiguity, but validation is still too vague.",
                    investigation_root_cause="The wall-jump transition in src/traversal_runtime.py likely never restores a spent charge after the player re-enters the airborne state.",
                    code_context="src/traversal_runtime.py owns the live air dash charge bookkeeping and the wall-jump transition that can restore one charge.",
                    implementation_medium="cpp",
                    implementation_medium_reason="The strongest surviving evidence points to code-owned traversal logic.",
                )
            return _investigation_payload(
                doc_hits=["docs/designer/air_dash_recharge.md"],
                doc_context="Wall jumps should restore one charge and the feature must stay capped at max_charges.",
                source_hits=["src/traversal_runtime.py"],
                test_hits=["tests/test_traversal_runtime.py"],
                current_runtime_paths=["src/traversal_runtime.py"],
                runtime_path_hypotheses=[
                    "src/traversal_runtime.py owns the live wall-jump recharge path and tests/test_traversal_runtime.py should validate both recharge and cap behavior."
                ],
                ownership_summary="src/traversal_runtime.py is the live wall-jump recharge owner and tests/test_traversal_runtime.py is the validation path for the feature.",
                investigation_summary="The third pass isolated the live owner, a specific root cause, and the exact validation path needed before implementation.",
                investigation_root_cause="The wall-jump transition in src/traversal_runtime.py needs to restore one spent charge and clamp the result to max_charges.",
                investigation_validation_plan="Update tests/test_traversal_runtime.py to prove a wall jump restores one charge and repeated wall jumps cannot overfill the cap.",
                code_context="src/traversal_runtime.py owns the wall-jump recharge path and tests/test_traversal_runtime.py should validate recharge and cap behavior.",
                implementation_medium="cpp",
                implementation_medium_reason="The strongest surviving evidence points to code-owned traversal logic.",
            )
        if schema_name == "gameplay_investigation_review":
            if self._investigation_calls == 1:
                return _investigation_review_payload(
                    section_rows=[
                        ("Supporting References", 10, "pass", "Design intent is grounded in the relevant references.", []),
                        ("Runtime Owner Precision", 12, "needs-work", "The live runtime owner still needs to be isolated.", ["Identify the current code runtime owner before drafting the feature plan."]),
                        ("Current vs Legacy Split", 4, "needs-work", "The brief still carries stale ownership ambiguity.", ["Separate the current runtime path from legacy references."]),
                        ("Ownership Summary", 10, "pass", "Ownership direction is plausible.", []),
                        ("Root Cause Hypothesis", 9, "needs-work", "The hypothesis still needs a clearer owner and transition.", ["State the likely failing transition and tie it to the selected runtime owner."]),
                        ("Investigation Summary", 10, "pass", "The summary explains what survived round 1.", []),
                        ("Implementation Medium", 5, "pass", "The work is clearly code-owned.", []),
                        ("Validation Plan", 4, "needs-work", "The validation path is still incomplete.", ["Name the exact automated validation path before implementation."]),
                        ("Noise Control", 5, "pass", "The evidence set stayed focused.", []),
                    ],
                    feedback="Round 1 still needs the live owner, a cleaner current-vs-legacy split, and an exact validation path.",
                    blocking_issues=[
                        "Runtime Owner Precision: Identify the current code runtime owner before drafting the feature plan.",
                        "Current vs Legacy Split: Separate the current runtime path from legacy references.",
                        "Validation Plan: Name the exact automated validation path before implementation.",
                    ],
                    improvement_actions=[
                        "Identify the current code runtime owner before drafting the feature plan.",
                        "Separate the current runtime path from legacy references.",
                        "Name the exact automated validation path before implementation.",
                    ],
                    approved=False,
                )
            if self._investigation_calls == 2:
                return _investigation_review_payload(
                    section_rows=[
                        ("Supporting References", 10, "pass", "Grounded references support the investigation.", []),
                        ("Runtime Owner Precision", 25, "pass", "The live runtime owner is isolated clearly.", []),
                        ("Current vs Legacy Split", 10, "pass", "Current ownership is separated from stale references.", []),
                        ("Ownership Summary", 10, "pass", "Ownership is summarized clearly enough for handoff.", []),
                        ("Root Cause Hypothesis", 15, "pass", "The causal hypothesis is concrete and grounded.", []),
                        ("Investigation Summary", 10, "pass", "The summary is handoff-ready.", []),
                        ("Implementation Medium", 5, "pass", "The implementation medium is justified.", []),
                        ("Validation Plan", 4, "needs-work", "The final validation path still needs the exact automated test handoff.", ["Name the exact automated validation path before implementation."]),
                        ("Noise Control", 5, "pass", "The evidence set stayed focused.", []),
                    ],
                    feedback="Round 2 isolated the live owner, but it still needs the exact automated validation path.",
                    blocking_issues=[
                        "Validation Plan: Name the exact automated validation path before implementation.",
                    ],
                    improvement_actions=[
                        "Name the exact automated validation path before implementation.",
                    ],
                    approved=False,
                )
            return _approved_investigation_review_payload(
                feedback="The investigation is finally strong enough to move into planning."
            )
        if schema_name == "gameplay_plan_review":
            self._review_calls += 1
            self.review_inputs.append(input_text)
            if self._review_calls == 1:
                return _plan_review_payload(
                    section_rows=[
                        ("Overview", 8, "needs-work", "The plan names the feature but not the nearby aerial states it could affect.", ["Explain the neighboring aerial states that must remain stable."]),
                        ("Task Type", 6, "needs-work", "The task type is named, but the approval rationale is too thin.", ["Justify why this traversal change is a feature that needs approval."]),
                        ("Existing Docs", 6, "needs-work", "The plan cites one doc but does not explain how it constrains the implementation.", ["Explain how the design doc constrains the recharge behavior and charge cap."]),
                        ("Implementation Steps", 10, "needs-work", "The implementation steps are too generic to trust for a gameplay feature.", ["Name the owning runtime file and the exact wall-jump recharge hook."]),
                        ("Unit Tests", 6, "needs-work", "The plan asks for a test, but it does not describe the assertions needed for recharge and cap behavior.", ["List the exact automated tests and assertions for recharge and cap behavior."]),
                        ("Risks", 5, "needs-work", "Risks are vague and do not show how regressions will be contained.", ["Describe the main traversal regression risk and the mitigation plan."]),
                        ("Acceptance Criteria", 4, "needs-work", "The acceptance criteria are too vague for approval.", ["Write player-visible acceptance criteria for recharge amount, cap behavior, and neighboring aerial states."]),
                    ],
                    feedback="Rejecting round 1. The plan is still too generic, especially around ownership, test assertions, and player-visible acceptance criteria.",
                    blocking_issues=[
                        "Implementation Steps: Name the owning runtime file and the exact wall-jump recharge hook.",
                        "Unit Tests: List the exact automated tests and assertions for recharge and cap behavior.",
                        "Acceptance Criteria: Write player-visible acceptance criteria for recharge amount, cap behavior, and neighboring aerial states.",
                    ],
                    improvement_actions=[
                        "Anchor the plan on src/traversal_runtime.py and the wall-jump recharge transition.",
                        "Specify automated regression coverage for recharge amount and charge-cap behavior.",
                        "Tighten acceptance criteria so the player-visible outcome is testable.",
                    ],
                    approved=False,
                )
            if self._review_calls == 2:
                return _plan_review_payload(
                    section_rows=[
                        ("Overview", 10, "pass", "The overview now names the gameplay goal and nearby traversal states.", []),
                        ("Task Type", 10, "pass", "The plan clearly explains why the work is a feature.", []),
                        ("Existing Docs", 10, "pass", "The plan cites the relevant design and runtime references.", []),
                        ("Implementation Steps", 20, "needs-work", "The plan is closer, but it still does not clearly protect the non-wall-jump aerial path.", ["Add an explicit safeguard for the neighboring non-wall-jump aerial transition."]),
                        ("Unit Tests", 14, "needs-work", "The tests cover recharge and cap behavior, but the non-wall-jump regression path is still missing.", ["Add a regression test proving unrelated aerial updates do not restore charges."]),
                        ("Risks", 8, "needs-work", "The risks are real, but the mitigation should name the specific neighboring transition being protected.", ["Name the neighboring non-wall-jump transition in the mitigation plan."]),
                        ("Acceptance Criteria", 11, "needs-work", "The acceptance criteria improved, but they still omit the non-wall-jump edge case.", ["Add a player-visible pass condition for the non-wall-jump aerial path."]),
                    ],
                    feedback="Round 2 is better, but it is still not strict enough for approval. The plan must protect the nearby non-wall-jump aerial path explicitly.",
                    blocking_issues=[
                        "Implementation Steps: Add an explicit safeguard for the neighboring non-wall-jump aerial transition.",
                        "Unit Tests: Add a regression test proving unrelated aerial updates do not restore charges.",
                        "Acceptance Criteria: Add a player-visible pass condition for the non-wall-jump aerial path.",
                    ],
                    improvement_actions=[
                        "Protect the neighboring non-wall-jump aerial transition explicitly in the implementation steps.",
                        "Add regression coverage for the non-wall-jump path.",
                        "State the non-wall-jump pass condition in acceptance criteria.",
                    ],
                    approved=False,
                )
            return _plan_review_payload(
                section_rows=[
                    ("Overview", 10, "pass", "The overview is scoped clearly enough for implementation.", []),
                    ("Task Type", 10, "pass", "The task is correctly framed as a feature that needs approval.", []),
                    ("Existing Docs", 10, "pass", "Relevant design and runtime docs are referenced clearly.", []),
                    ("Implementation Steps", 25, "pass", "The implementation steps name the owner, the exact recharge hook, and the neighboring transition safeguards.", []),
                    ("Unit Tests", 20, "pass", "The plan now covers recharge amount, cap behavior, and the non-wall-jump regression path.", []),
                    ("Risks", 10, "pass", "Risks and mitigations are concrete and actionable.", []),
                    ("Acceptance Criteria", 15, "pass", "Acceptance criteria now capture the player-visible result and the neighboring edge case.", []),
                ],
                feedback="Approve round 3. The plan is specific enough to implement safely.",
                blocking_issues=[],
                improvement_actions=[],
                approved=True,
            )
        raise AssertionError(f"Unexpected schema_name: {schema_name}")


class StrictLoopingCodegenLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "strict looping codegen test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        raise AssertionError("Codegen profile should not call generate_text in this test")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, input_text, schema, effort
        if schema_name != "gameplay_code_bundle":
            raise AssertionError(f"Unexpected schema_name: {schema_name}")
        return {
            "source_code": "\n".join(
                [
                    "class AirDashState:",
                    "    def __init__(self, max_charges=2, air_dash_charges=1):",
                    "        self.max_charges = max_charges",
                    "        self.air_dash_charges = air_dash_charges",
                    "",
                    "    def consume_air_dash(self):",
                    "        if self.air_dash_charges <= 0:",
                    "            return False",
                    "        self.air_dash_charges -= 1",
                    "        return True",
                    "",
                    "    def on_wall_jump(self):",
                    "        self.air_dash_charges = min(self.max_charges, self.air_dash_charges + 1)",
                    "        return self",
                ]
            ),
            "test_code": "\n".join(
                [
                    "from src.traversal_runtime import AirDashState",
                    "",
                    "def test_wall_jump_restores_one_spent_air_dash_charge():",
                    "    state = AirDashState(max_charges=2, air_dash_charges=1)",
                    "    assert state.consume_air_dash() is True",
                    "    state.on_wall_jump()",
                    "    assert state.air_dash_charges == 1",
                    "",
                    "def test_wall_jump_does_not_exceed_max_charge_cap():",
                    "    state = AirDashState(max_charges=2, air_dash_charges=2)",
                    "    state.on_wall_jump()",
                    "    assert state.air_dash_charges == 2",
                    "",
                    "def test_non_wall_jump_path_leaves_charges_unchanged():",
                    "    state = AirDashState(max_charges=2, air_dash_charges=1)",
                    "    assert state.air_dash_charges == 1",
                ]
            ),
            "implementation_notes": "Added wall-jump recharge logic in src/traversal_runtime.py and covered recharge, cap, and neighboring aerial-path regression behavior.",
        }


class StrictLoopingFeatureLLMManager:
    def __init__(self) -> None:
        self._default_client = StrictLoopingFeatureLLMClient()
        self._codegen_client = StrictLoopingCodegenLLMClient()

    @property
    def client(self) -> StrictLoopingFeatureLLMClient:
        return self._default_client

    def resolve(self, profile: str | None = None):
        if profile == "codegen":
            return self._codegen_client
        return self._default_client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default", "codegen"]


class ProcessDriftReviewerLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "process-drift reviewer test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        raise AssertionError("Process drift reviewer test client should not call generate_text")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, input_text, schema, effort
        if schema_name != "gameplay_plan_review":
            raise AssertionError(f"Unexpected schema_name: {schema_name}")
        return {
            "score": 72,
            "feedback": "The plan is ready for implementation, but the review drifted into approval bookkeeping.",
            "missing_sections": [],
            "section_reviews": [
                {"section": "Overview", "score": 10, "status": "pass", "rationale": "The gameplay goal and scope are clear.", "action_items": []},
                {"section": "Task Type", "score": 6, "status": "needs-work", "rationale": "The section says round metadata is stale for the current review context.", "action_items": ["Update round metadata to match the active review round."]},
                {"section": "Existing Docs", "score": 10, "status": "pass", "rationale": "The docs and owner paths are grounded.", "action_items": []},
                {"section": "Implementation Steps", "score": 15, "status": "needs-work", "rationale": "The implementation is concrete, but the verification artifact naming still references the prior round.", "action_items": ["Rename the verification artifacts for the active review round."]},
                {"section": "Unit Tests", "score": 20, "status": "pass", "rationale": "Recharge, cap, and adjacent-path tests are covered.", "action_items": []},
                {"section": "Risks", "score": 10, "status": "pass", "rationale": "Risks and mitigations are concrete.", "action_items": []},
                {"section": "Acceptance Criteria", "score": 9, "status": "needs-work", "rationale": "Acceptance criteria still require an independent verifier sign-off artifact.", "action_items": ["Add a sign-off artifact for the active review round."]},
            ],
            "blocking_issues": [
                "Task Type: Update round metadata to match the active review round.",
                "Implementation Steps: Rename the verification artifacts for the active review round.",
                "Acceptance Criteria: Add a sign-off artifact for the active review round.",
            ],
            "improvement_actions": [
                "Update round metadata to match the active review round.",
                "Rename the verification artifacts for the active review round.",
                "Add a sign-off artifact for the active review round.",
            ],
            "approved": False,
        }


class ProcessDriftReviewerLLMManager:
    def __init__(self) -> None:
        self._client = ProcessDriftReviewerLLMClient()

    def resolve(self, profile: str | None = None):
        del profile
        return self._client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default"]


class ContradictoryReadyReviewerLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "contradictory-ready reviewer test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        raise AssertionError("Contradictory-ready reviewer test client should not call generate_text")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, input_text, schema, effort
        if schema_name != "gameplay_plan_review":
            raise AssertionError(f"Unexpected schema_name: {schema_name}")
        return {
            "score": 100,
            "feedback": "The plan is ready for implementation.",
            "missing_sections": [],
            "section_reviews": [
                {"section": "Overview", "score": 10, "status": "pass", "rationale": "Player-visible outcome is explicit.", "action_items": []},
                {"section": "Task Type", "score": 10, "status": "pass", "rationale": "Task framing and owner are clear.", "action_items": []},
                {"section": "Existing Docs", "score": 10, "status": "pass", "rationale": "Docs and owner paths are grounded.", "action_items": []},
                {"section": "Implementation Steps", "score": 25, "status": "pass", "rationale": "Implementation steps are concrete and safe.", "action_items": []},
                {"section": "Unit Tests", "score": 20, "status": "pass", "rationale": "Regression coverage is explicit.", "action_items": []},
                {"section": "Risks", "score": 10, "status": "pass", "rationale": "Risks and mitigations are concrete.", "action_items": []},
                {"section": "Acceptance Criteria", "score": 15, "status": "pass", "rationale": "Acceptance criteria are player-visible and verifiable.", "action_items": []},
            ],
            "blocking_issues": [
                "Player Outcome: Name the player-visible result and the nearby gameplay boundary that must remain stable.",
            ],
            "improvement_actions": [
                "Clarify the player-visible outcome and the nearby gameplay boundary.",
            ],
            "approved": False,
        }


class ContradictoryReadyReviewerLLMManager:
    def __init__(self) -> None:
        self._client = ContradictoryReadyReviewerLLMClient()

    def resolve(self, profile: str | None = None):
        del profile
        return self._client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default"]


class CodeOnlyMixedNoiseLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "code-only mixed-noise test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort, input_text
        if "Write a concise markdown bug investigation brief" in instructions:
            return "\n".join(
                [
                    "# Gameplay Bug Context",
                    "",
                    "## Bug Summary",
                    "- Spawn should restore movement immediately and remain blocked only while stunned or rooted.",
                ]
            )
        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, input_text, schema, effort
        if schema_name == "gameplay_task_classification":
            return {
                "task_type": "bugfix",
                "reason": "The prompt fixes unintended spawn movement behavior.",
            }
        if schema_name == "gameplay_investigation_strategy":
            return {
                "focus_terms": ["spawn", "movement", "stun", "root"],
                "avoid_terms": ["archive"],
                "search_notes": ["Inspect the live movement owner and the validating tests before editing."],
                "implementation_medium_hint": "mixed",
                "implementation_medium_reason": "Spawn and movement gating sometimes span code and Blueprint systems.",
                "investigation_root_cause": "Movement probably never re-enables in the spawn path.",
                "investigation_validation_plan": "Validate with the existing player movement tests.",
            }
        if schema_name == "gameplay_engineering_context":
            return _investigation_payload(
                doc_hits=["docs/designer/player_movement.md"],
                doc_context="Players should regain movement immediately after spawning and remain blocked only while stunned or rooted.",
                source_hits=["src/player_movement.py"],
                test_hits=["tests/test_player_movement.py"],
                current_runtime_paths=["src/player_movement.py"],
                runtime_path_hypotheses=[
                    "src/player_movement.py owns the movement gate and tests/test_player_movement.py validates the spawn path."
                ],
                ownership_summary="src/player_movement.py is the current runtime owner and tests/test_player_movement.py covers the spawn regression path.",
                investigation_summary="The grounded evidence points to code-only movement ownership in src/player_movement.py.",
                code_context="src/player_movement.py owns the spawn movement gate and tests/test_player_movement.py validates the path.",
                implementation_medium="mixed",
                implementation_medium_reason="Spawn and movement gating sometimes span code and Blueprint systems.",
            )
        if schema_name == "gameplay_investigation_review":
            return _approved_investigation_review_payload(
                feedback="Investigation quality is strong enough to continue into implementation."
            )
        raise AssertionError(f"Unexpected schema_name: {schema_name}")


class CodeOnlyNoiseCodegenLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "code-only noise codegen test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        raise AssertionError("Codegen profile should not call generate_text in this test")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, input_text, schema, effort
        if schema_name != "gameplay_code_bundle":
            raise AssertionError(f"Unexpected schema_name: {schema_name}")
        return {
            "source_code": "\n".join(
                [
                    "class PlayerCharacter:",
                    "    def __init__(self):",
                    "        self.can_move = False",
                    "        self.is_stunned = False",
                    "",
                    "    def spawn(self):",
                    "        self.is_stunned = False",
                    "        self.can_move = True",
                    "        return self",
                    "",
                    "    def apply_stun(self):",
                    "        self.is_stunned = True",
                    "",
                    "    def clear_stun(self):",
                    "        self.is_stunned = False",
                    "",
                    "    def move(self):",
                    "        return self.can_move and not self.is_stunned",
                ]
            ),
            "test_code": "\n".join(
                [
                    "from src.player_movement import PlayerCharacter",
                    "",
                    "def test_player_can_move_after_spawn():",
                    "    player = PlayerCharacter().spawn()",
                    "    assert player.move() is True",
                    "",
                    "def test_player_cannot_move_while_stunned():",
                    "    player = PlayerCharacter().spawn()",
                    "    player.apply_stun()",
                    "    assert player.move() is False",
                    "",
                    "def test_player_can_move_again_after_stun_clears():",
                    "    player = PlayerCharacter().spawn()",
                    "    player.apply_stun()",
                    "    player.clear_stun()",
                    "    assert player.move() is True",
                ]
            ),
            "implementation_notes": "Updated the spawn path to restore movement and added a regression for stun recovery.",
        }


class CodeOnlyMixedNoiseLLMManager:
    def __init__(self) -> None:
        self._default_client = CodeOnlyMixedNoiseLLMClient()
        self._codegen_client = CodeOnlyNoiseCodegenLLMClient()

    def resolve(self, profile: str | None = None):
        if profile == "codegen":
            return self._codegen_client
        return self._default_client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default", "codegen"]


class RepairDefaultLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "repair default test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort, input_text
        if "Write a concise markdown bug investigation brief" in instructions:
            return "\n".join(
                [
                    "# Gameplay Bug Context",
                    "",
                    "## Bug Summary",
                    "- Runtime ownership and validation path are clear enough to attempt a fix.",
                ]
            )
        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, input_text, schema, effort
        if schema_name == "gameplay_task_classification":
            return {
                "task_type": "bugfix",
                "reason": "The prompt fixes broken gameplay behavior.",
            }
        if schema_name == "gameplay_engineering_context":
            return _investigation_payload(
                source_hits=["src/runtime.py"],
                test_hits=["tests/test_runtime.py"],
                current_runtime_paths=["src/runtime.py"],
                runtime_path_hypotheses=["The code path in src/runtime.py should be validated by tests/test_runtime.py."],
                ownership_summary="The bug is likely owned by src/runtime.py and should be validated by tests/test_runtime.py.",
                investigation_summary="Runtime ownership and validation path are clear enough to attempt a fix.",
                code_context="The bug is likely in src/runtime.py and should be validated by tests/test_runtime.py.",
                implementation_medium="cpp",
                implementation_medium_reason="The task is fully code-owned in this fixture.",
            )
        if schema_name == "gameplay_investigation_review":
            return _approved_investigation_review_payload(
                feedback="Investigation quality is sufficient, so the repair-loop test can focus on self-test failures."
            )
        raise AssertionError(f"Unexpected schema_name: {schema_name}")


class AlwaysFailingCodegenLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "always failing codegen test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        raise AssertionError("Codegen profile should not call generate_text in this test")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, input_text, schema, effort
        if schema_name != "gameplay_code_bundle":
            raise AssertionError(f"Unexpected schema_name: {schema_name}")
        return {
            "source_code": "\n".join(
                [
                    "from __future__ import annotations",
                    "",
                    "def build_gameplay_change_summary() -> dict:",
                    '    return {"implementation_status": "broken"}',
                ]
            ),
            "test_code": "\n".join(
                [
                    "from gameplay_change import build_gameplay_change_summary",
                    "",
                    "def test_build_gameplay_change_summary():",
                    "    summary = build_gameplay_change_summary()",
                    '    assert summary["implementation_status"] == "ready-for-review"',
                ]
            ),
            "implementation_notes": "Intentional failing bundle for repair loop regression coverage.",
        }


class StuckRepairLLMManager:
    def __init__(self) -> None:
        self._default_client = RepairDefaultLLMClient()
        self._codegen_client = AlwaysFailingCodegenLLMClient()

    def resolve(self, profile: str | None = None):
        if profile == "codegen":
            return self._codegen_client
        return self._default_client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default", "codegen"]


class PlannerLoopLLMClient:
    def __init__(self) -> None:
        self.research_inputs: list[str] = []
        self.plan_inputs: list[str] = []
        self._research_calls = 0
        self._plan_calls = 0

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "planner loop test client"

    def _research_report(self, stage: int) -> str:
        if stage == 1:
            return "\n".join(
                [
                    "# Gameplay Planner Research Report",
                    "",
                    "## Goal",
                    "- Plan a wall-jump recharge solution before implementation.",
                    "",
                    "## Grounded Evidence",
                    "- docs/designer/wall_jump_recharge.md",
                    "- src/traversal_runtime.py",
                    "",
                    "## Constraints",
                    "- The first pass still needs sharper validation detail.",
                    "",
                    "## Proposed Direction",
                    "- Route the solution through the traversal runtime rather than a generic movement helper.",
                    "",
                    "## Open Questions",
                    "- Which non-wall-jump aerial path is most likely to regress?",
                ]
            )
        if stage == 2:
            return "\n".join(
                [
                    "# Gameplay Planner Research Report",
                    "",
                    "## Goal",
                    "- Tighten the wall-jump recharge solution around the live traversal owner.",
                    "",
                    "## Grounded Evidence",
                    "- docs/designer/wall_jump_recharge.md",
                    "- src/traversal_runtime.py",
                    "- tests/test_traversal_runtime.py",
                    "",
                    "## Constraints",
                    "- The next plan revision must add the non-wall-jump regression path explicitly.",
                    "",
                    "## Proposed Direction",
                    "- Keep recharge in the wall-jump transition and validate the neighboring aerial path.",
                    "",
                    "## Open Questions",
                    "- None beyond the remaining regression detail.",
                ]
            )
        return "\n".join(
            [
                "# Gameplay Planner Research Report",
                "",
                "## Goal",
                "- Finalize an implementation-ready wall-jump recharge plan.",
                "",
                "## Grounded Evidence",
                "- docs/designer/wall_jump_recharge.md",
                "- src/traversal_runtime.py",
                "- tests/test_traversal_runtime.py",
                "",
                "## Constraints",
                "- Recharge must stay scoped to the valid wall-jump transition and remain capped.",
                "",
                "## Proposed Direction",
                "- Implement one-charge restore on the wall-jump transition and guard the adjacent aerial path with tests.",
                "",
                "## Open Questions",
                "- None.",
            ]
        )

    def _plan_doc(self, stage: int) -> str:
        if stage == 1:
            return "\n".join(
                [
                    "# Gameplay Solution Plan",
                    "",
                    "## Problem Framing",
                    "- Players should regain one air dash charge after a valid wall jump.",
                    "- The plan should stay reviewable before code changes begin.",
                    "",
                    "## Current Context",
                    "- docs/designer/wall_jump_recharge.md",
                    "- src/traversal_runtime.py",
                    "",
                    "## Proposed Solution",
                    "- Update traversal so wall jump can restore a charge.",
                    "- Keep the change narrow.",
                    "",
                    "## Execution Plan",
                    "- Inspect the traversal runtime.",
                    "- Draft the change sequence.",
                    "",
                    "## Validation Plan",
                    "- Add tests for recharge behavior.",
                    "",
                    "## Risks and Open Questions",
                    "- Risk: nearby movement states could regress.",
                    "- Open question: which aerial path needs the strongest safeguard?",
                ]
            )
        if stage == 2:
            return "\n".join(
                [
                    "# Gameplay Solution Plan",
                    "",
                    "## Problem Framing",
                    "- Players should regain exactly one spent air dash charge after a valid wall jump.",
                    "- The solution must preserve nearby aerial traversal behavior and respect the existing charge cap.",
                    "",
                    "## Current Context",
                    "- docs/designer/wall_jump_recharge.md defines the one-charge restore and cap requirement.",
                    "- src/traversal_runtime.py is the likely owner for air dash spending and wall-jump traversal state.",
                    "- tests/test_traversal_runtime.py is the obvious regression anchor for recharge coverage.",
                    "",
                    "## Proposed Solution",
                    "- Keep the recharge logic inside src/traversal_runtime.py at the valid wall-jump transition instead of a generic airborne update path.",
                    "- Restore one spent charge and clamp the result to the configured max so repeated wall jumps cannot overfill charges.",
                    "- Preserve neighboring aerial states by limiting recharge to the explicit wall-jump success path.",
                    "",
                    "## Execution Plan",
                    "- Confirm the exact wall-jump transition in src/traversal_runtime.py that can safely grant the recharge.",
                    "- Define the implementation sequence around charge spend, wall-jump success, and cap clamping before coding begins.",
                    "- Review the adjacent aerial path that must not restore charges and call it out in the handoff.",
                    "",
                    "## Validation Plan",
                    "- Add a regression check proving a valid wall jump restores exactly one spent charge.",
                    "- Add a regression check proving repeated wall jumps never exceed the configured charge cap.",
                    "",
                    "## Risks and Open Questions",
                    "- Risk: recharge could leak into a generic airborne path and over-grant mobility.",
                    "- Mitigation: keep the solution tied to the explicit wall-jump success transition in src/traversal_runtime.py.",
                    "- Open question: name the non-wall-jump aerial path that must stay unchanged in the final plan.",
                ]
            )
        return "\n".join(
            [
                "# Gameplay Solution Plan",
                "",
                "## Problem Framing",
                "- Players should regain exactly one spent air dash charge after a valid wall jump.",
                "- The final plan must preserve nearby aerial traversal behavior and never exceed the existing charge cap.",
                "",
                "## Current Context",
                "- docs/designer/wall_jump_recharge.md defines the one-charge restore and cap requirement.",
                "- src/traversal_runtime.py owns air dash charge spending and the wall-jump traversal transition that can safely grant recharge.",
                "- tests/test_traversal_runtime.py is the validation anchor for recharge amount, cap behavior, and adjacent traversal regressions.",
                "- Current behavior today is still incomplete: the traversal runtime owns the path, but recharge-specific assertions have not been added yet.",
                "- The plan treats the runtime owner and test anchor as grounded facts, while any caller-side validity contract remains an explicit follow-up assumption.",
                "",
                "## Proposed Solution",
                "- Add the recharge at the explicit wall-jump success transition in src/traversal_runtime.py so the proposal stays grounded in the live traversal owner.",
                "- Restore one spent charge and clamp the result to the configured cap instead of touching generic airborne recovery code.",
                "- Protect the neighboring non-wall-jump aerial update path by keeping recharge out of unrelated airborne transitions.",
                "",
                "## Execution Plan",
                "- Confirm the exact wall-jump success transition in src/traversal_runtime.py and document where recharge should happen.",
                "- Sequence the future implementation around spent-charge detection, wall-jump success, one-charge restore, and cap clamping.",
                "- Hand off the non-wall-jump aerial safeguard explicitly so implementation and review stay aligned on the nearby regression path.",
                "",
                "## Validation Plan",
                "- Validate with tests/test_traversal_runtime.py that a valid wall jump restores exactly one spent charge.",
                "- Validate with tests/test_traversal_runtime.py that repeated wall jumps never exceed the configured charge cap.",
                "- Validate with tests/test_traversal_runtime.py that the non-wall-jump aerial path leaves charges unchanged.",
                "- If the test runner is unavailable in the current environment, record that execution block explicitly and still preserve the exact assertions for the next implementation pass.",
                "",
                "## Risks and Open Questions",
                "- Risk: recharge could leak into a generic airborne path and over-grant mobility.",
                "- Mitigation: bind the solution to the explicit wall-jump success transition and review the neighboring aerial branch separately.",
                "- Operational concern: the caller-side valid wall-jump contract must stay single-fire so repeated event dispatch cannot double-grant recharge.",
                "- Open question: none; the remaining edge case is already named in the validation plan.",
            ]
        )

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort
        if "write a concise markdown gameplay planning research report" in instructions.lower():
            self._research_calls += 1
            self.research_inputs.append(input_text)
            return self._research_report(min(self._research_calls, 3))
        if "rewrite the full markdown gameplay solution plan" in instructions.lower():
            self._plan_calls += 1
            self.plan_inputs.append(input_text)
            return self._plan_doc(min(self._plan_calls, 3))
        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        raise AssertionError(f"Unexpected generate_json request on planner client: {schema_name}")


class PlannerLoopReviewerLLMClient:
    def __init__(self) -> None:
        self.review_inputs: list[str] = []
        self._review_calls = 0

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "planner reviewer loop test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        raise AssertionError("Reviewer profile should not call generate_text in this test")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        del instructions, schema, effort
        if schema_name != "gameplay_solution_plan_review":
            raise AssertionError(f"Unexpected schema_name: {schema_name}")
        self._review_calls += 1
        self.review_inputs.append(input_text)
        if self._review_calls == 1:
            return {
                "score": 74,
                "feedback": (
                    "Round 1 is still too generic. The proposed solution and execution plan do not name the exact "
                    "wall-jump owner strongly enough, and the validation path is under-specified."
                ),
                "missing_sections": [],
                "section_reviews": [
                    {"section": "Problem Framing", "score": 15, "status": "pass", "rationale": "The goal is clear.", "action_items": []},
                    {"section": "Current Context", "score": 15, "status": "pass", "rationale": "The current references are grounded.", "action_items": []},
                    {"section": "Proposed Solution", "score": 12, "status": "needs-work", "rationale": "Name the live traversal owner and the exact recharge path.", "action_items": ["Anchor the solution on src/traversal_runtime.py and the wall-jump success transition."]},
                    {"section": "Execution Plan", "score": 12, "status": "needs-work", "rationale": "The steps are too short to trust as a handoff.", "action_items": ["Expand the execution plan into a clearer ordered sequence."]},
                    {"section": "Validation Plan", "score": 8, "status": "needs-work", "rationale": "The validation path is too vague.", "action_items": ["List the exact regression checks for recharge amount and cap behavior."]},
                    {"section": "Risks and Open Questions", "score": 12, "status": "needs-work", "rationale": "The risk exists, but the mitigation is still thin.", "action_items": ["Explain how the adjacent aerial path will be protected."]},
                ],
                "blocking_issues": [
                    "Proposed Solution: Anchor the solution on src/traversal_runtime.py and the wall-jump success transition.",
                    "Execution Plan: Expand the execution plan into a clearer ordered sequence.",
                    "Validation Plan: List the exact regression checks for recharge amount and cap behavior.",
                ],
                "improvement_actions": [
                    "Name the exact traversal owner in the proposal.",
                    "Expand the ordered execution steps.",
                    "Spell out the recharge and cap validation checks.",
                ],
                "approved": False,
            }
        if self._review_calls == 2:
            return {
                "score": 86,
                "feedback": (
                    "Round 2 is much better, but the review still needs the neighboring non-wall-jump aerial path "
                    "called out explicitly in validation before approval."
                ),
                "missing_sections": [],
                "section_reviews": [
                    {"section": "Problem Framing", "score": 15, "status": "pass", "rationale": "The framing is clear.", "action_items": []},
                    {"section": "Current Context", "score": 15, "status": "pass", "rationale": "The context is grounded.", "action_items": []},
                    {"section": "Proposed Solution", "score": 20, "status": "pass", "rationale": "The solution is now concrete.", "action_items": []},
                    {"section": "Execution Plan", "score": 20, "status": "pass", "rationale": "The execution sequence is actionable.", "action_items": []},
                    {"section": "Validation Plan", "score": 10, "status": "needs-work", "rationale": "The non-wall-jump aerial regression path is still missing.", "action_items": ["Add a regression check proving the non-wall-jump aerial path leaves charges unchanged."]},
                    {"section": "Risks and Open Questions", "score": 6, "status": "needs-work", "rationale": "The remaining open question should now be answered directly.", "action_items": ["Close the remaining question by naming the neighboring aerial path in validation and handoff notes."]},
                ],
                "blocking_issues": [
                    "Validation Plan: Add a regression check proving the non-wall-jump aerial path leaves charges unchanged.",
                ],
                "improvement_actions": [
                    "Add the non-wall-jump aerial regression check.",
                    "Close the remaining open question directly in the final plan.",
                ],
                "approved": False,
            }
        return {
            "score": 95,
            "feedback": "Round 3 is specific, grounded, and safe enough to approve.",
            "missing_sections": [],
            "section_reviews": [
                {"section": "Problem Framing", "score": 15, "status": "pass", "rationale": "The framing is clear.", "action_items": []},
                {"section": "Current Context", "score": 15, "status": "pass", "rationale": "The context is grounded.", "action_items": []},
                {"section": "Proposed Solution", "score": 20, "status": "pass", "rationale": "The solution is implementation-ready.", "action_items": []},
                {"section": "Execution Plan", "score": 20, "status": "pass", "rationale": "The execution plan is concrete.", "action_items": []},
                {"section": "Validation Plan", "score": 15, "status": "pass", "rationale": "The validation plan covers recharge, cap, and neighboring path behavior.", "action_items": []},
                {"section": "Risks and Open Questions", "score": 10, "status": "needs-work", "rationale": "Small residual risk remains, but it is not blocking.", "action_items": ["Monitor the neighboring aerial branch during implementation review."]},
            ],
            "blocking_issues": [],
            "improvement_actions": [],
            "approved": True,
        }


class PlannerLoopLLMManager:
    def __init__(self) -> None:
        self._planner_client = PlannerLoopLLMClient()
        self._reviewer_client = PlannerLoopReviewerLLMClient()

    @property
    def planner_client(self) -> PlannerLoopLLMClient:
        return self._planner_client

    @property
    def reviewer_client(self) -> PlannerLoopReviewerLLMClient:
        return self._reviewer_client

    def resolve(self, profile: str | None = None):
        if profile == "reviewer":
            return self._reviewer_client
        return self._planner_client

    def is_enabled(self, profile: str | None = None) -> bool:
        return self.resolve(profile).is_enabled()

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: {self.resolve(profile).describe()}"

    def available_profiles(self) -> list[str]:
        return ["default", "planner", "reviewer"]


class WorkflowDrivenRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.workflows_root = cls.project_root / "Workflows"
        cls.tools_root = cls.project_root / "Tools"
        cls.llm_manager = FakeLLMManager()
        cls.tool_registry = load_tools(
            project_root=cls.project_root,
            tools_root=cls.tools_root,
            llm_manager=cls.llm_manager,
        )
        cls.registry = load_workflows(
            project_root=cls.project_root,
            workflows_root=cls.workflows_root,
            llm_manager=cls.llm_manager,
        )

    @staticmethod
    def _single_workflow_artifact_dir(run_dir: Path, workflow_name: str) -> Path:
        matches = list(run_dir.glob(f"tasks/*/{workflow_name}"))
        if len(matches) != 1:
            raise AssertionError(f"Expected exactly one artifact dir for {workflow_name}, found {len(matches)}")
        return matches[0]

    def test_load_tools_and_workflows_for_gameplay_task(self) -> None:
        tool_names = [item.name for item in self.tool_registry.list_metadata()]
        self.assertEqual(
            tool_names,
            [
                "find-gameplay-blueprints",
                "find-gameplay-code",
                "load-blueprint-context",
                "load-source-context",
            ],
        )
        qualified_tool_names = [item.qualified_name for item in self.tool_registry.list_metadata()]
        self.assertEqual(
            qualified_tool_names,
            [
                "agentswarm::find-gameplay-blueprints",
                "agentswarm::find-gameplay-code",
                "agentswarm::load-blueprint-context",
                "agentswarm::load-source-context",
            ],
        )

        metadata = self.registry.list_metadata()
        names = [item.name for item in metadata]
        self.assertEqual(
            names,
            [
                "gameplay-engineer-planner",
                "gameplay-engineer-workflow",
                "gameplay-reviewer-workflow",
                "template-investigation-reviewer-workflow",
                "template-investigation-workflow",
            ],
        )

        exposed_names = [item.name for item in self.registry.list_metadata(exposed_only=True)]
        self.assertEqual(
            exposed_names,
            [
                "gameplay-engineer-planner",
                "gameplay-engineer-workflow",
                "template-investigation-workflow",
            ],
        )

        routed = self.registry.route("Fix a combat gameplay bug affecting melee dodge timing")
        self.assertIsNotNone(routed)
        self.assertEqual(routed.name, "gameplay-engineer-workflow")

        planner_routed = self.registry.route(
            "Research and plan a gameplay solution for wall jump recharge before implementation"
        )
        self.assertIsNotNone(planner_routed)
        self.assertEqual(planner_routed.name, "gameplay-engineer-planner")

    def test_reviewer_workflow_flags_missing_sections(self) -> None:
        result = self.registry.invoke(
            "gameplay-reviewer-workflow",
            {
                "task_prompt": "Fix combat dodge cancel bug",
                "plan_doc": "# Gameplay Implementation Plan\n\n## Overview\n- A short plan.",
                "review_round": 1,
            },
        )

        self.assertEqual(result["score"], 0)
        self.assertEqual(result["missing_sections"], [])
        self.assertFalse(result["approved"])
        self.assertTrue(result["blocking_issues"])
        self.assertTrue(result["improvement_actions"])
        self.assertEqual(result["section_reviews"], [])
        self.assertEqual(result["loop_status"], "llm-unavailable")
        self.assertFalse(result["loop_should_continue"])
        self.assertIn("## Section Scores", result["feedback"])
        self.assertIn("Reviewer LLM is unavailable", result["feedback"])

    def test_reviewer_workflow_ignores_process_only_drift_when_plan_is_technically_ready(self) -> None:
        drift_registry = load_workflows(
            project_root=self.project_root,
            workflows_root=self.workflows_root,
            llm_manager=ProcessDriftReviewerLLMManager(),
        )
        reviewer_graph = drift_registry.get("gameplay-reviewer-workflow").graph
        self.assertIsNotNone(reviewer_graph)

        plan_doc = "\n".join(
            [
                "# Gameplay Implementation Plan",
                "",
                "## Overview",
                "- A valid wall jump should restore exactly one spent air dash charge.",
                "- Nearby non-wall-jump aerial paths must remain unchanged and the cap must stay enforced.",
                "",
                "## Task Type",
                "- feature",
                "- Classification reason: this adds a new player-facing traversal rule that needs approval before implementation.",
                "",
                "## Existing Docs",
                "- docs/designer/air_dash_recharge.md",
                "- docs/programming/air_dash_runtime.md",
                "",
                "## Implementation Steps",
                "- Update src/traversal_runtime.py at the explicit wall-jump success hook.",
                "- Restore one spent charge and clamp it to the configured maximum.",
                "- Keep recharge out of unrelated aerial transitions so the neighboring path stays unchanged.",
                "",
                "## Unit Tests",
                "- Add a test proving a valid wall jump restores exactly one spent charge.",
                "- Add a test proving repeated valid wall jumps never exceed the cap.",
                "- Add a test proving non-wall-jump aerial updates leave charges unchanged.",
                "",
                "## Risks",
                "- Risk: recharge could leak into a generic aerial path.",
                "- Mitigation: keep the hook tied to the explicit wall-jump transition and guard the neighboring path with tests.",
                "",
                "## Acceptance Criteria",
                "- After spending one charge, the player regains exactly one charge when a valid wall jump succeeds.",
                "- Repeated wall jumps never exceed the configured charge cap.",
                "- Nearby aerial traversal paths that are not wall jumps continue to leave charges unchanged.",
            ]
        )

        result = reviewer_graph.invoke(
            {
                "task_prompt": "Add gameplay feature: a valid wall jump should restore exactly one air dash charge without exceeding the cap.",
                "plan_doc": plan_doc,
                "review_round": 1,
            }
        )

        self.assertEqual(result["review_round"], 2)
        self.assertTrue(result["approved"])
        self.assertTrue(result["review_approved"])
        self.assertEqual(result["loop_status"], "passed")
        self.assertEqual(result["review_loop_status"], "passed")
        self.assertEqual(result["score"], 100)
        self.assertFalse(result["blocking_issues"])
        self.assertFalse(result["improvement_actions"])
        self.assertNotIn("round metadata", result["feedback"].lower())
        self.assertNotIn("sign-off artifact", result["feedback"].lower())

    def test_reviewer_workflow_drops_contradictory_generic_hard_blockers_when_all_sections_pass(self) -> None:
        drift_registry = load_workflows(
            project_root=self.project_root,
            workflows_root=self.workflows_root,
            llm_manager=ContradictoryReadyReviewerLLMManager(),
        )
        reviewer_graph = drift_registry.get("gameplay-reviewer-workflow").graph
        self.assertIsNotNone(reviewer_graph)

        plan_doc = "\n".join(
            [
                "# Gameplay Implementation Plan",
                "",
                "## Overview",
                "- A valid wall jump should restore exactly one spent air dash charge.",
                "- Nearby aerial traversal that is not a wall jump must remain unchanged.",
                "",
                "## Task Type",
                "- feature",
                "- Classification reason: this adds a new player-facing traversal rule that needs approval before implementation.",
                "",
                "## Existing Docs",
                "- docs/designer/air_dash_recharge.md",
                "- docs/programming/air_dash_runtime.md",
                "",
                "## Implementation Steps",
                "- Update src/traversal_runtime.py at the explicit wall-jump success hook.",
                "- Restore one spent charge and clamp it to the configured maximum.",
                "- Keep recharge out of unrelated aerial transitions so adjacent traversal remains unchanged.",
                "",
                "## Unit Tests",
                "- Add a test proving a valid wall jump restores exactly one spent charge.",
                "- Add a test proving repeated valid wall jumps never exceed the cap.",
                "- Add a test proving non-wall-jump aerial updates leave charges unchanged.",
                "",
                "## Risks",
                "- Risk: recharge could leak into a generic aerial path.",
                "- Mitigation: keep the hook tied to the explicit wall-jump transition and guard the neighboring path with tests.",
                "",
                "## Acceptance Criteria",
                "- After spending one charge, the player regains exactly one charge when a valid wall jump succeeds.",
                "- Repeated wall jumps never exceed the configured charge cap.",
                "- Nearby aerial traversal paths that are not wall jumps continue to leave charges unchanged.",
            ]
        )

        result = reviewer_graph.invoke(
            {
                "task_prompt": "Add gameplay feature: a valid wall jump should restore exactly one air dash charge without exceeding the cap.",
                "plan_doc": plan_doc,
                "review_round": 1,
            }
        )

        self.assertEqual(result["review_round"], 2)
        self.assertTrue(result["approved"])
        self.assertEqual(result["loop_status"], "passed")
        self.assertEqual(result["score"], 100)
        self.assertFalse(result["blocking_issues"])
        self.assertIn("The plan is ready for implementation.", result["feedback"])

    def test_planner_workflow_loops_research_and_review_until_solution_plan_is_approved(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-planner-loop-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            (host_root / "docs" / "designer").mkdir(parents=True, exist_ok=True)
            (host_root / "src").mkdir(parents=True, exist_ok=True)
            (host_root / "tests").mkdir(parents=True, exist_ok=True)
            (host_root / "docs" / "designer" / "wall_jump_recharge.md").write_text(
                (
                    "# Wall Jump Recharge\n\n"
                    "A valid wall jump should restore exactly one spent air dash charge.\n"
                    "The recharge must not exceed the configured charge cap.\n"
                ),
                encoding="utf-8",
            )
            (host_root / "src" / "traversal_runtime.py").write_text(
                "\n".join(
                    [
                        "class TraversalRuntime:",
                        "    def grant_wall_jump_recharge(self):",
                        "        return 1",
                    ]
                ),
                encoding="utf-8",
            )
            (host_root / "tests" / "test_traversal_runtime.py").write_text(
                "\n".join(
                    [
                        "from src.traversal_runtime import TraversalRuntime",
                        "",
                        "def test_runtime_smoke():",
                        "    assert TraversalRuntime().grant_wall_jump_recharge() == 1",
                    ]
                ),
                encoding="utf-8",
            )

            planner_manager = PlannerLoopLLMManager()
            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=planner_manager,
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            planner_graph = registry.get("gameplay-engineer-planner").graph
            self.assertIsNotNone(planner_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "planner-loop"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = planner_graph.invoke(
                {
                    "prompt": "Research and plan a gameplay solution for wall jump recharge in traversal runtime before implementation.",
                    "task_prompt": "Research and plan a gameplay solution for wall jump recharge in traversal runtime before implementation.",
                    "task_id": "task-1-plan-wall-jump-recharge",
                    "run_dir": str(run_dir),
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            review_round_1 = (artifact_dir / "planner_review_round_1.md").read_text(encoding="utf-8")
            review_round_2 = (artifact_dir / "planner_review_round_2.md").read_text(encoding="utf-8")
            review_round_3 = (artifact_dir / "planner_review_round_3.md").read_text(encoding="utf-8")
            final_plan = (artifact_dir / "solution_plan.md").read_text(encoding="utf-8")

            self.assertEqual(result["planning_round"], 3)
            self.assertEqual(result["score"], 100)
            self.assertTrue(result["approved"])
            self.assertEqual(result["loop_status"], "passed")
            self.assertEqual(result["final_report"]["status"], "completed")
            self.assertEqual(result["final_report"]["planning_rounds"], 3)
            self.assertIn("docs/designer/wall_jump_recharge.md", result["doc_hits"])
            self.assertIn("src/traversal_runtime.py", result["source_hits"])
            self.assertIn("tests/test_traversal_runtime.py", final_plan)
            self.assertIn("non-wall-jump aerial path leaves charges unchanged", final_plan)
            self.assertTrue((artifact_dir / "planner_research_round_1.md").exists())
            self.assertTrue((artifact_dir / "planner_research_round_2.md").exists())
            self.assertTrue((artifact_dir / "planner_research_round_3.md").exists())
            self.assertTrue((artifact_dir / "planner_plan_round_1.md").exists())
            self.assertTrue((artifact_dir / "planner_plan_round_2.md").exists())
            self.assertTrue((artifact_dir / "planner_plan_round_3.md").exists())
            self.assertTrue((artifact_dir / "planner_review_round_1.md").exists())
            self.assertTrue((artifact_dir / "planner_review_round_2.md").exists())
            self.assertTrue((artifact_dir / "planner_review_round_3.md").exists())
            self.assertTrue((artifact_dir / "research_report.md").exists())
            self.assertTrue((artifact_dir / "solution_plan.md").exists())
            self.assertTrue((artifact_dir / "final_report.md").exists())
            self.assertIn("- Approved: False", review_round_1)
            self.assertIn("Approval bar: >= 90/100 and zero blocking issues", review_round_1)
            self.assertIn("Player Outcome:", review_round_1)
            self.assertIn("Current Behavior Evidence:", review_round_1)
            self.assertIn("Speculation Control:", review_round_1)
            self.assertIn("Validation Plan: List the exact regression checks for recharge amount and cap behavior.", review_round_1)
            self.assertIn("- Approved: False", review_round_2)
            self.assertIn("Edge and Regression Coverage:", review_round_2)
            self.assertIn("non-wall-jump aerial path leaves charges unchanged", review_round_2)
            self.assertIn("- Approved: True", review_round_3)
            self.assertEqual(len(planner_manager.planner_client.research_inputs), 3)
            self.assertEqual(len(planner_manager.planner_client.plan_inputs), 3)
            self.assertRegex(planner_manager.planner_client.plan_inputs[1], r"Previous score: \d+/100")
            self.assertIn(
                "Anchor the solution on src/traversal_runtime.py and the wall-jump success transition.",
                planner_manager.planner_client.plan_inputs[1],
            )
            self.assertIn(
                "non-wall-jump aerial path leaves charges unchanged",
                planner_manager.planner_client.plan_inputs[2],
            )
            self.assertEqual(len(planner_manager.reviewer_client.review_inputs), 3)
            self.assertIn("Hard blocker and scoring rules:", planner_manager.reviewer_client.review_inputs[0])
            self.assertIn("[hard blocker]", planner_manager.reviewer_client.review_inputs[0])
            self.assertIn("Planning round: 2", planner_manager.reviewer_client.review_inputs[1])
            self.assertIn("Planning round: 3", planner_manager.reviewer_client.review_inputs[2])

    def test_main_graph_runs_end_to_end_without_llm(self) -> None:
        graph = build_main_graph(registry=self.registry, llm_manager=self.llm_manager)
        with tempfile.TemporaryDirectory(prefix="langgraph-tests-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = graph.invoke(
                build_initial_state(
                    prompt="Add a melee combo extension feature and keep 3C responsiveness stable",
                    run_dir=str(run_dir),
                )
            )

            self.assertIn("gameplay-engineer-workflow", result["final_response"])
            self.assertTrue((run_dir / "summary.md").exists())
            trace_log = run_dir / "graph_traversal.log"
            self.assertTrue(trace_log.exists())
            self.assertTrue((run_dir / GRAPH_DEBUG_TRACE_FILE).exists())

            artifact_dir = self._single_workflow_artifact_dir(run_dir, "gameplay-engineer-workflow")
            self.assertTrue((artifact_dir / "engineer_investigation.md").exists())
            self.assertTrue((artifact_dir / "investigation_review_round_1.md").exists())
            self.assertTrue((artifact_dir / "investigation_abort.md").exists())
            self.assertTrue((artifact_dir / "final_report.md").exists())
            self.assertFalse((artifact_dir / "plan_doc.md").exists())
            self.assertFalse((artifact_dir / "architecture_plan.md").exists())
            self.assertFalse((artifact_dir / "pull_request.md").exists())
            self.assertFalse((artifact_dir / "self_test.txt").exists())
            self.assertIn("gameplay investigation never grounded a safe handoff", result["final_response"])

            trace_output = trace_log.read_text(encoding="utf-8")
            self.assertIn("[main_graph] [analyze_prompt] ENTER", trace_output)
            self.assertIn("[main_graph] [dispatch_active_task] ROUTE", trace_output)
            self.assertIn("input_keys=", trace_output)
            self.assertIn("output_keys=", trace_output)
            self.assertIn(f"details={GRAPH_DEBUG_TRACE_FILE}#", trace_output)
            self.assertIn("next=agentswarm__gameplay-engineer-workflow", trace_output)
            self.assertIn("[gameplay-engineer-workflow] [evaluate_investigation] ENTER", trace_output)
            self.assertIn("[gameplay-engineer-workflow] [prepare_investigation_blocked_delivery] ENTER", trace_output)
            self.assertIn("[main_graph] [finalize] EXIT", trace_output)

    def test_main_graph_registers_workflow_subgraphs(self) -> None:
        graph = build_main_graph(registry=self.registry, llm_manager=self.llm_manager)
        subgraphs = dict(graph.get_subgraphs())

        self.assertIn("agentswarm__gameplay-engineer-workflow", subgraphs)
        self.assertIn("agentswarm__gameplay-reviewer-workflow", subgraphs)

    def test_engineer_workflow_registers_reviewer_subgraph(self) -> None:
        engineer_graph = self.registry.get("gameplay-engineer-workflow").graph
        self.assertIsNotNone(engineer_graph)

        subgraphs = dict(engineer_graph.get_subgraphs())
        self.assertIn("gameplay-reviewer-workflow", subgraphs)
        self.assertNotIn("agentswarm__find-gameplay-blueprints", subgraphs)
        self.assertNotIn("agentswarm__find-gameplay-code", subgraphs)
        self.assertNotIn("agentswarm__load-blueprint-context", subgraphs)
        self.assertNotIn("agentswarm__load-source-context", subgraphs)

    def test_engineer_workflow_handles_doc_search_inside_codex_investigation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="langgraph-engineer-investigation-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            (host_root / "docs" / "designer").mkdir(parents=True, exist_ok=True)
            (host_root / "src").mkdir(parents=True, exist_ok=True)
            (host_root / "tests").mkdir(parents=True, exist_ok=True)
            (host_root / "Content" / "Combat").mkdir(parents=True, exist_ok=True)
            (host_root / "docs" / "designer" / "combat_design_template.md").write_text(
                "# Combat Design Template\n\nDodge cancel and melee responsiveness notes.\n",
                encoding="utf-8",
            )
            (host_root / "src" / "combat_dodge.py").write_text(
                "\n".join(
                    [
                        "def get_dodge_window():",
                        "    return 3",
                    ]
                ),
                encoding="utf-8",
            )
            (host_root / "tests" / "test_combat_dodge.py").write_text(
                "\n".join(
                    [
                        "from src.combat_dodge import get_dodge_window",
                        "",
                        "def test_get_dodge_window():",
                        "    assert get_dodge_window() == 3",
                    ]
                ),
                encoding="utf-8",
            )
            (host_root / "Content" / "Combat" / "BP_DodgeCancel.uasset").write_bytes(b"binary-blueprint-placeholder")
            (host_root / "Content" / "Combat" / "BP_DodgeCancel.copy").write_text(
                "\n".join(
                    [
                        "EventGraph:",
                        "  State: DodgeCancel",
                        "  Bug: cancel window closes too early during melee recovery",
                    ]
                ),
                encoding="utf-8",
            )

            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=ContextOnlyInvestigationLLMManager(),
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": "Fix combat dodge cancel bug in melee gameplay",
                    "task_prompt": "Fix combat dodge cancel bug in melee gameplay",
                    "task_id": "task-1-fix-combat-dodge-cancel-bug",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])

            self.assertIn("docs/designer/combat_design_template.md", result["doc_hits"])
            self.assertIn("src/combat_dodge.py", result["source_hits"])
            self.assertIn("tests/test_combat_dodge.py", result["test_hits"])
            self.assertIn("Content/Combat/BP_DodgeCancel.uasset", result["blueprint_hits"])
            self.assertIn("Content/Combat/BP_DodgeCancel.copy", result["blueprint_text_hits"])
            self.assertIn("src/combat_dodge.py", result["current_runtime_paths"])
            self.assertTrue(result["ownership_summary"])
            self.assertTrue(result["investigation_summary"])
            self.assertTrue((artifact_dir / "engineer_investigation.md").exists())
            investigation_doc = (artifact_dir / "engineer_investigation.md").read_text(encoding="utf-8")
            self.assertIn("## Doc Hits", investigation_doc)
            self.assertIn("## Current Runtime Paths", investigation_doc)
            self.assertNotIn("find-gameplay-docs", investigation_doc)

    def test_engineer_workflow_stops_after_read_only_investigation_handoff(self) -> None:
        with tempfile.TemporaryDirectory(prefix="langgraph-engineer-read-only-") as temp_dir:
            host_root = Path(temp_dir) / "s2-host"
            (host_root / "docs" / "architecture" / "S2_CurrentArchitecture").mkdir(parents=True, exist_ok=True)
            (host_root / "Source" / "S2" / "Private" / "StunTracker").mkdir(parents=True, exist_ok=True)
            (host_root / "Source" / "S2" / "Private" / "Tests").mkdir(parents=True, exist_ok=True)
            (host_root / "Content" / "S2" / "Combat").mkdir(parents=True, exist_ok=True)
            (host_root / "S2.uproject").write_text("{}", encoding="utf-8")
            (
                host_root / "docs" / "architecture" / "S2_CurrentArchitecture" / "stun_flow.md"
            ).write_text(
                "# Stun Flow\n\nStun entry and recovery both pass through the character state tracker.\n",
                encoding="utf-8",
            )
            (
                host_root / "Source" / "S2" / "Private" / "StunTracker" / "SipherStunTrackerComponent.cpp"
            ).write_text(
                "void BeginStun(); void EndStunRecovery();",
                encoding="utf-8",
            )
            (
                host_root / "Source" / "S2" / "Private" / "Tests" / "SipherStunTrackerTests.cpp"
            ).write_text(
                "IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSipherStunTrackerTests, \"S2.Stun.Recovery\", 0)",
                encoding="utf-8",
            )
            (host_root / "Content" / "S2" / "Combat" / "BP_StunRecovery.uasset").write_bytes(b"uasset")
            (host_root / "Content" / "S2" / "Combat" / "BP_StunRecovery.copy").write_text(
                "EventGraph: OnStunEnd -> Clear stun movement lock",
                encoding="utf-8",
            )

            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=ContextOnlyInvestigationLLMManager(),
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "read-only"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": (
                        "Investigate how stun entry and recovery are implemented. Do not modify any files. "
                        "Collect the most relevant docs, gameplay C++ files, tests, and Blueprint assets before "
                        "debugging a bug where a character stays stunned after recovery."
                    ),
                    "task_prompt": (
                        "Investigate how stun entry and recovery are implemented. Do not modify any files. "
                        "Collect the most relevant docs, gameplay C++ files, tests, and Blueprint assets before "
                        "debugging a bug where a character stays stunned after recovery."
                    ),
                    "task_id": "task-1-investigate-stun-recovery-read-only",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            self.assertEqual(result["execution_track"], "bugfix")
            self.assertFalse(result["implementation_requested"])
            self.assertFalse(result["workspace_write_enabled"])
            self.assertEqual(result["workspace_source_file"], "")
            self.assertEqual(result["workspace_test_file"], "")
            self.assertIn("Read-only investigation requested", result["workspace_write_summary"])
            self.assertEqual(result["review_round"], 0)
            self.assertEqual(result["repair_round"], 0)
            self.assertEqual(result["code_attempt"], 0)
            self.assertEqual(result["final_report"]["status"], "investigation-completed")
            self.assertEqual(result["final_report"]["implementation_requested"], False)
            self.assertIn(
                "Source/S2/Private/StunTracker/SipherStunTrackerComponent.cpp",
                result["source_hits"],
            )
            self.assertIn(
                "Source/S2/Private/Tests/SipherStunTrackerTests.cpp",
                result["test_hits"],
            )
            self.assertIn("Content/S2/Combat/BP_StunRecovery.uasset", result["blueprint_hits"])
            self.assertTrue((artifact_dir / "engineer_investigation.md").exists())
            self.assertTrue((artifact_dir / "bug_context.md").exists())
            self.assertTrue((artifact_dir / "investigation_delivery.md").exists())
            self.assertFalse((artifact_dir / "pull_request.md").exists())
            self.assertFalse((artifact_dir / "self_test.txt").exists())

    def test_engineer_workflow_writes_generated_code_into_host_project_roots(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-code-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            host_root.mkdir(parents=True, exist_ok=True)
            (host_root / "src").mkdir(parents=True, exist_ok=True)
            (host_root / "tests").mkdir(parents=True, exist_ok=True)
            (host_root / "src" / "combat_dodge.py").write_text(
                "\n".join(
                    [
                        "def get_dodge_window():",
                        "    return 3",
                    ]
                ),
                encoding="utf-8",
            )
            (host_root / "tests" / "test_combat_dodge.py").write_text(
                "\n".join(
                    [
                        "from src.combat_dodge import get_dodge_window",
                        "",
                        "def test_get_dodge_window():",
                        "    assert get_dodge_window() == 3",
                    ]
                ),
                encoding="utf-8",
            )

            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=ContextOnlyInvestigationLLMManager(),
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "test-run"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": "Fix combat dodge window gameplay bug",
                    "task_prompt": "Fix combat dodge window gameplay bug",
                    "task_id": "task-1-fix-combat-dodge-window",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            workspace_source = host_root / Path(result["workspace_source_file"])
            workspace_test = host_root / Path(result["workspace_test_file"])
            self.assertTrue(result["workspace_write_enabled"])
            self.assertEqual(result["workspace_source_file"], "src/combat_dodge.py")
            self.assertEqual(result["workspace_test_file"], "tests/test_combat_dodge.py")
            self.assertTrue(workspace_source.exists())
            self.assertTrue(workspace_test.exists())
            self.assertIn("def get_dodge_window()", workspace_source.read_text(encoding="utf-8"))
            self.assertIn("test_get_dodge_window", workspace_test.read_text(encoding="utf-8"))
            self.assertTrue((artifact_dir / "workspace_write_manifest.md").exists())
            self.assertEqual(result["execution_track"], "bugfix")
            self.assertFalse(result["requires_architecture_review"])
            self.assertEqual(result["implementation_medium"], "cpp")
            self.assertEqual(result["review_round"], 0)
            self.assertTrue((artifact_dir / "bug_context.md").exists())
            self.assertFalse((artifact_dir / "review_round_1.md").exists())

    def test_engineer_workflow_falls_back_to_local_hits_when_llm_returns_only_context(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-context-only-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            host_root.mkdir(parents=True, exist_ok=True)
            (host_root / "src").mkdir(parents=True, exist_ok=True)
            (host_root / "tests").mkdir(parents=True, exist_ok=True)
            (host_root / "src" / "combat_dodge.py").write_text(
                "\n".join(
                    [
                        "def get_dodge_window():",
                        "    return 3",
                    ]
                ),
                encoding="utf-8",
            )
            (host_root / "tests" / "test_combat_dodge.py").write_text(
                "\n".join(
                    [
                        "from src.combat_dodge import get_dodge_window",
                        "",
                        "def test_get_dodge_window():",
                        "    assert get_dodge_window() == 3",
                    ]
                ),
                encoding="utf-8",
            )

            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=ContextOnlyInvestigationLLMManager(),
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "context-run"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": "Fix combat dodge window gameplay bug",
                    "task_prompt": "Fix combat dodge window gameplay bug",
                    "task_id": "task-1-fix-combat-dodge-window",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            self.assertIn("src/combat_dodge.py", result["source_hits"])
            self.assertIn("tests/test_combat_dodge.py", result["test_hits"])
            self.assertEqual(result["workspace_source_file"], "src/combat_dodge.py")
            self.assertEqual(result["workspace_test_file"], "tests/test_combat_dodge.py")

    def test_load_agentswarm_config_augments_unreal_hosts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-unreal-config-") as temp_dir:
            host_root = Path(temp_dir) / "s2-host"
            host_root.mkdir(parents=True, exist_ok=True)
            (host_root / "S2.uproject").write_text("{}", encoding="utf-8")
            (host_root / "Source").mkdir(parents=True, exist_ok=True)
            (host_root / "Content").mkdir(parents=True, exist_ok=True)

            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)

            self.assertIn("Source", config.source_roots)
            self.assertIn("Plugins", config.source_roots)
            self.assertIn("Source", config.test_roots)
            self.assertIn("docs/architecture", config.doc_roots)
            self.assertIn("Intermediate", config.exclude_roots)
            self.assertIn("Plugins/Marketplace", config.exclude_roots)

    def test_engineer_helpers_normalize_descriptive_hits_for_unreal_hosts(self) -> None:
        engineer_entry = self.project_root / "Workflows" / "gameplay-engineer-workflow" / "entry.py"
        spec = importlib.util.spec_from_file_location("test_engineer_unreal_entry", engineer_entry)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory(prefix="agentswarm-unreal-host-") as temp_dir:
            host_root = Path(temp_dir) / "s2-host"
            (host_root / "Source" / "S2" / "Private" / "StunTracker").mkdir(parents=True, exist_ok=True)
            (host_root / "Source" / "S2" / "Private" / "Tests").mkdir(parents=True, exist_ok=True)
            (host_root / "Content" / "S2" / "GCN").mkdir(parents=True, exist_ok=True)
            (host_root / ".blueprints" / "S2" / "Core" / "Enemy").mkdir(parents=True, exist_ok=True)
            (host_root / "Intermediate" / "PipInstall" / "Lib" / "site-packages" / "numpy" / "f2py").mkdir(
                parents=True,
                exist_ok=True,
            )
            (host_root / "Plugins" / "Marketplace" / "XeSS" / "Source" / "XeSSSDK" / "inc" / "xess").mkdir(
                parents=True,
                exist_ok=True,
            )
            (host_root / "S2.uproject").write_text("{}", encoding="utf-8")
            (host_root / "Source" / "S2" / "Private" / "StunTracker" / "SipherStunTrackerComponent.cpp").write_text(
                "void BeginStun(); void OnStundEnd();",
                encoding="utf-8",
            )
            (host_root / "Source" / "S2" / "Private" / "Tests" / "SipherStunTrackerTests.cpp").write_text(
                "IMPLEMENT_SIMPLE_AUTOMATION_TEST(FSipherStunTrackerTests, \"S2.Stun.Tracker\", 0)",
                encoding="utf-8",
            )
            (host_root / "Content" / "S2" / "GCN" / "GCN_Stun.uasset").write_bytes(b"uasset")
            (host_root / ".blueprints" / "S2" / "Core" / "Enemy" / "BP_EnemyCharacterBase.md").write_text(
                "# BP Enemy Character Base\nOnStunTagChanged",
                encoding="utf-8",
            )
            (
                host_root / "Intermediate" / "PipInstall" / "Lib" / "site-packages" / "numpy" / "f2py" / "crackfortran.py"
            ).write_text("stun recovery noise", encoding="utf-8")
            (
                host_root / "Plugins" / "Marketplace" / "XeSS" / "Source" / "XeSSSDK" / "inc" / "xess" / "xess_debug.h"
            ).write_text("stun recovery marketplace noise", encoding="utf-8")

            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)

            source_hits = module._normalize_relative_hits(
                host_root,
                ["Source/S2/Private/StunTracker/SipherStunTrackerComponent.cpp - main stun flow"],
                config.exclude_roots,
                allowed_suffixes=module.SOURCE_EXTENSIONS,
            )
            test_hits = module._normalize_relative_hits(
                host_root,
                ["Source/S2/Private/Tests/SipherStunTrackerTests.cpp - automation coverage"],
                config.exclude_roots,
                allowed_suffixes=module.SOURCE_EXTENSIONS,
            )
            blueprint_hits = module._normalize_relative_hits(
                host_root,
                ["Content/S2/GCN/GCN_Stun.uasset - gameplay cue"],
                config.exclude_roots,
                allowed_suffixes={module.BLUEPRINT_ASSET_EXTENSION},
            )
            blueprint_text_hits = module._normalize_relative_hits(
                host_root,
                [".blueprints/S2/Core/Enemy/BP_EnemyCharacterBase.md - exported graph"],
                config.exclude_roots,
                allowed_suffixes=module.BLUEPRINT_TEXT_EXTENSIONS,
            )

            self.assertEqual(source_hits, ["Source/S2/Private/StunTracker/SipherStunTrackerComponent.cpp"])
            self.assertEqual(test_hits, ["Source/S2/Private/Tests/SipherStunTrackerTests.cpp"])
            self.assertEqual(blueprint_hits, ["Content/S2/GCN/GCN_Stun.uasset"])
            self.assertEqual(blueprint_text_hits, [".blueprints/S2/Core/Enemy/BP_EnemyCharacterBase.md"])

            fallback_source_hits, fallback_test_hits = module._find_local_code_hits(
                task_prompt="Investigate stun recovery bug in S2 Unreal project",
                scope_root=host_root,
                source_roots=config.source_roots,
                test_roots=config.test_roots,
                exclude_roots=config.exclude_roots,
            )
            self.assertIn("Source/S2/Private/StunTracker/SipherStunTrackerComponent.cpp", fallback_source_hits)
            self.assertIn("Source/S2/Private/Tests/SipherStunTrackerTests.cpp", fallback_test_hits)
            self.assertNotIn(
                "Intermediate/PipInstall/Lib/site-packages/numpy/f2py/crackfortran.py",
                fallback_source_hits,
            )
            self.assertNotIn(
                "Plugins/Marketplace/XeSS/Source/XeSSSDK/inc/xess/xess_debug.h",
                fallback_source_hits,
            )

            workspace_source_file, workspace_test_file = module._resolve_workspace_targets(
                task_id="task-1-investigate-stun-recovery",
                scope_root=host_root,
                source_hits=[],
                test_hits=[],
                source_roots=config.source_roots,
                test_roots=config.test_roots,
                allow_workspace_writes=module._workspace_roots_exist(host_root, config.source_roots),
            )
            self.assertTrue(workspace_source_file.startswith("Source/"))
            self.assertTrue(workspace_test_file.startswith("Source/"))

    def test_engineer_workflow_shortens_artifact_dir_for_long_task_ids(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-long-task-id-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            (host_root / "docs" / "designer").mkdir(parents=True, exist_ok=True)
            (host_root / "src").mkdir(parents=True, exist_ok=True)
            (host_root / "tests").mkdir(parents=True, exist_ok=True)
            (host_root / "docs" / "designer" / "player_movement.md").write_text(
                "# Player Movement\n\nPlayers should regain movement right after spawn.\n",
                encoding="utf-8",
            )
            (host_root / "src" / "player_movement.py").write_text(
                "\n".join(
                    [
                        "class PlayerCharacter:",
                        "    def spawn(self):",
                        "        return None",
                    ]
                ),
                encoding="utf-8",
            )
            (host_root / "tests" / "test_player_movement.py").write_text(
                "\n".join(
                    [
                        "from src.player_movement import PlayerCharacter",
                        "",
                        "def test_player_spawn_smoke():",
                        "    assert PlayerCharacter().spawn() is None",
                    ]
                ),
                encoding="utf-8",
            )

            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=ContextOnlyInvestigationLLMManager(),
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            long_task_id = (
                "task-1-review-host-project-gameplay-docs-player-spawn-movement-and-post-spawn-input-"
                "initialization-sequence-for-regression-analysis"
            )
            run_dir = host_root / ".agentswarm" / "runs" / "test-run"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": "Fix player cannot move after spawn",
                    "task_prompt": "Fix player cannot move after spawn",
                    "task_id": long_task_id,
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            self.assertTrue((artifact_dir / "engineer_investigation.md").exists())
            self.assertNotEqual(artifact_dir.parent.name, long_task_id)
            self.assertLess(len(artifact_dir.parent.name), len(long_task_id))

    def test_engineer_workflow_handles_blueprint_bugfix_with_manual_handoff(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-blueprint-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            blueprint_dir = host_root / "Content" / "Combat"
            blueprint_dir.mkdir(parents=True, exist_ok=True)
            (blueprint_dir / "BP_DodgeCancel.uasset").write_bytes(b"binary-blueprint-placeholder")
            (blueprint_dir / "BP_DodgeCancel.copy").write_text(
                "\n".join(
                    [
                        "EventGraph:",
                        "  State: DodgeCancel",
                        "  Bug: cancel window closes too early during melee recovery",
                    ]
                ),
                encoding="utf-8",
            )

            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=ContextOnlyInvestigationLLMManager(),
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "bp-run"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": "Fix Blueprint dodge cancel bug in melee gameplay",
                    "task_prompt": "Fix Blueprint dodge cancel bug in melee gameplay",
                    "task_id": "task-1-fix-blueprint-dodge-cancel-bug",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            patch_note = host_root / "Content" / "Combat" / "BP_DodgeCancel.agentswarm_fix.md"
            instructions_doc = (artifact_dir / "blueprint_fix_instructions.md").read_text(encoding="utf-8")

            self.assertEqual(result["execution_track"], "bugfix")
            self.assertFalse(result["requires_architecture_review"])
            self.assertEqual(result["implementation_medium"], "blueprint")
            self.assertEqual(result["review_round"], 0)
            self.assertTrue(result["blueprint_manual_action_required"])
            self.assertTrue((artifact_dir / "bug_context.md").exists())
            self.assertTrue((artifact_dir / "blueprint_fix_instructions.md").exists())
            self.assertTrue((artifact_dir / "blueprint_fix_manifest.md").exists())
            self.assertTrue(patch_note.exists())
            self.assertFalse((artifact_dir / "review_round_1.md").exists())
            self.assertEqual(result["final_report"]["status"], "manual-validation-required")
            self.assertIn("Manual Unreal Editor", result["workspace_write_summary"])
            self.assertIn("## Goal", instructions_doc)
            self.assertIn("## Safe Patch Steps", instructions_doc)
            self.assertIn("## Verification Checklist", instructions_doc)
            self.assertIn("BP_DodgeCancel", instructions_doc)

    def test_engineer_workflow_clamps_code_only_mixed_noise_to_code_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-code-only-noise-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            (host_root / "docs" / "designer").mkdir(parents=True, exist_ok=True)
            (host_root / "src").mkdir(parents=True, exist_ok=True)
            (host_root / "tests").mkdir(parents=True, exist_ok=True)
            (host_root / "docs" / "designer" / "player_movement.md").write_text(
                (
                    "# Player Movement\n\n"
                    "Players should regain movement immediately after spawning and remain blocked only while stunned or rooted.\n"
                ),
                encoding="utf-8",
            )
            (host_root / "src" / "player_movement.py").write_text(
                "\n".join(
                    [
                        "class PlayerCharacter:",
                        "    def __init__(self):",
                        "        self.can_move = False",
                        "        self.is_stunned = False",
                        "",
                        "    def spawn(self):",
                        "        self.is_stunned = False",
                        "        return self",
                        "",
                        "    def apply_stun(self):",
                        "        self.is_stunned = True",
                        "",
                        "    def clear_stun(self):",
                        "        self.is_stunned = False",
                        "",
                        "    def move(self):",
                        "        return self.can_move and not self.is_stunned",
                    ]
                ),
                encoding="utf-8",
            )
            (host_root / "tests" / "test_player_movement.py").write_text(
                "\n".join(
                    [
                        "from src.player_movement import PlayerCharacter",
                        "",
                        "def test_player_cannot_move_before_spawn():",
                        "    assert PlayerCharacter().move() is False",
                    ]
                ),
                encoding="utf-8",
            )

            noise_manager = CodeOnlyMixedNoiseLLMManager()
            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=noise_manager,
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "code-only-noise"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": "Fix gameplay bug: the player cannot move after spawning",
                    "task_prompt": "Fix gameplay bug: the player cannot move after spawning",
                    "task_id": "task-1-fix-player-spawn-movement",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            investigation_doc = (artifact_dir / "engineer_investigation.md").read_text(encoding="utf-8")
            pull_request = (artifact_dir / "pull_request.md").read_text(encoding="utf-8")
            bug_context = (artifact_dir / "bug_context.md").read_text(encoding="utf-8")

            self.assertEqual(result["execution_track"], "bugfix")
            self.assertEqual(result["implementation_medium"], "cpp")
            self.assertFalse(result["blueprint_manual_action_required"])
            self.assertEqual(result["blueprint_fix_strategy"], "not-applicable")
            self.assertEqual(result["final_report"]["implementation_medium"], "cpp")
            self.assertFalse(result["final_report"]["blueprint_manual_action_required"])
            self.assertFalse((artifact_dir / "blueprint_fix_instructions.md").exists())
            self.assertFalse((artifact_dir / "blueprint_fix_manifest.md").exists())
            self.assertIn("Implementation Medium: cpp", investigation_doc)
            self.assertIn("## Blueprint Hits", investigation_doc)
            self.assertIn("- None.", investigation_doc)
            self.assertIn("Implementation medium: cpp", pull_request)
            self.assertIn("Blueprint fix strategy: not-applicable", pull_request)
            self.assertIn("Blueprint manual action required: False", pull_request)
            self.assertNotIn("Blueprint manual action required: True", pull_request)
            self.assertIn("Implementation Medium: cpp", bug_context)
            self.assertIn("Blueprint Fix Strategy: not-applicable", bug_context)
            self.assertIn("- No Blueprint assets were matched.", bug_context)

    def test_engineer_workflow_maxes_investigation_and_plan_loops_under_strict_review(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-strict-feature-loop-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            (host_root / "docs" / "designer").mkdir(parents=True, exist_ok=True)
            (host_root / "docs" / "programming").mkdir(parents=True, exist_ok=True)
            (host_root / "src").mkdir(parents=True, exist_ok=True)
            (host_root / "tests").mkdir(parents=True, exist_ok=True)
            (host_root / "docs" / "designer" / "air_dash_recharge.md").write_text(
                (
                    "# Air Dash Recharge\n\n"
                    "A valid wall jump should restore exactly one air dash charge.\n"
                    "The recharge must not exceed the configured air dash charge cap.\n"
                ),
                encoding="utf-8",
            )
            (host_root / "docs" / "programming" / "air_dash_runtime.md").write_text(
                "# Air Dash Runtime\n\nThe air dash runtime owns charge spending and recharge rules.\n",
                encoding="utf-8",
            )
            (host_root / "src" / "traversal_runtime.py").write_text(
                "\n".join(
                    [
                        "class TraversalRuntime:",
                        "    def __init__(self):",
                        "        self.value = 0",
                        "",
                        "    def update(self, value):",
                        "        self.value = value",
                        "        return self.value",
                    ]
                ),
                encoding="utf-8",
            )
            (host_root / "tests" / "test_traversal_runtime.py").write_text(
                "\n".join(
                    [
                        "from src.traversal_runtime import TraversalRuntime",
                        "",
                        "def test_runtime_smoke():",
                        "    state = TraversalRuntime()",
                        "    assert state.update(3) == 3",
                    ]
                ),
                encoding="utf-8",
            )

            strict_manager = StrictLoopingFeatureLLMManager()
            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=strict_manager,
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)
            workflow_module = sys.modules["workflow_agentswarm__gameplay_engineer_workflow"]

            def staged_investigation_quality(_context, state):
                if state["investigation_round"] == 1:
                    checks = [
                        ("Supporting References", True, "Relevant docs or config files support the investigation."),
                        ("Runtime Owner Precision", False, "Replace generic supporting hits with the current code owner that directly drives the gameplay behavior."),
                        ("Current vs Legacy Split", False, "Separate the current runtime path from legacy references so the next prompt does not anchor on stale systems."),
                        ("Ownership Summary", True, "The ownership summary points at a plausible runtime owner or root cause."),
                        ("Root Cause Hypothesis", False, "State the likely failing transition and tie it to the selected runtime owner."),
                        ("Investigation Summary", True, "The investigation summary explains what was learned and what to validate next."),
                        ("Implementation Medium", True, "The workflow classified the work as code, Blueprint, or mixed with evidence-backed rationale."),
                        ("Validation Plan", False, "Name the exact automated test or manual validation flow that should prove the fix."),
                        ("Noise Control", True, "The evidence set stayed focused and avoided low-value noise."),
                    ]
                elif state["investigation_round"] == 2:
                    checks = [
                        ("Supporting References", True, "Relevant docs or config files support the investigation."),
                        ("Runtime Owner Precision", True, "The investigation isolated a current code runtime owner."),
                        ("Current vs Legacy Split", True, "The workflow separated current runtime ownership from legacy references."),
                        ("Ownership Summary", True, "The ownership summary points at a plausible runtime owner or root cause."),
                        ("Root Cause Hypothesis", True, "The investigation names a plausible failure point and ties it to the selected runtime owner."),
                        ("Investigation Summary", True, "The investigation summary explains what was learned and what to validate next."),
                        ("Implementation Medium", True, "The workflow classified the work as code, Blueprint, or mixed with evidence-backed rationale."),
                        ("Validation Plan", False, "Name the exact automated test or manual validation flow that should prove the fix."),
                        ("Noise Control", True, "The evidence set stayed focused and avoided low-value noise."),
                    ]
                else:
                    checks = [
                        ("Supporting References", True, "Relevant docs or config files support the investigation."),
                        ("Runtime Owner Precision", True, "The investigation isolated a current code runtime owner."),
                        ("Current vs Legacy Split", True, "The workflow separated current runtime ownership from legacy references."),
                        ("Ownership Summary", True, "The ownership summary points at a plausible runtime owner or root cause."),
                        ("Root Cause Hypothesis", True, "The investigation names a plausible failure point and ties it to the selected runtime owner."),
                        ("Investigation Summary", True, "The investigation summary explains what was learned and what to validate next."),
                        ("Implementation Medium", True, "The workflow classified the work as code, Blueprint, or mixed with evidence-backed rationale."),
                        ("Validation Plan", True, "The workflow records the exact automated validation path for the next step."),
                        ("Noise Control", True, "The evidence set stayed focused and avoided low-value noise."),
                    ]

                weights = {
                    "Supporting References": 10,
                    "Runtime Owner Precision": 25,
                    "Current vs Legacy Split": 10,
                    "Ownership Summary": 10,
                    "Root Cause Hypothesis": 15,
                    "Investigation Summary": 10,
                    "Implementation Medium": 5,
                    "Validation Plan": 10,
                    "Noise Control": 5,
                }
                normalized_checks = [
                    {
                        "section": label,
                        "score": weights[label] if passed else max(1, round(weights[label] * 0.4)),
                        "max_score": weights[label],
                        "status": "pass" if passed else "needs-work",
                        "rationale": rationale,
                        "action_items": [] if passed else [rationale],
                    }
                    for label, passed, rationale in checks
                ]
                score = sum(weights[label] for label, passed, _ in checks if passed)
                missing_sections = [item["section"] for item in normalized_checks if item["status"] != "pass"]
                blocking_issues = []
                if state["investigation_round"] == 1:
                    blocking_issues = [
                        "Identify the current code runtime owner before drafting the feature plan.",
                        "Name the exact automated validation path before implementation.",
                    ]
                elif state["investigation_round"] == 2:
                    blocking_issues = ["Name the exact automated validation path before implementation."]
                improvement_actions = [rationale for _, passed, rationale in checks if not passed]
                progress = workflow_module.evaluate_quality_loop(
                    workflow_module.INVESTIGATION_LOOP_SPEC,
                    round_index=state["investigation_round"],
                    score=score,
                    approved=score >= workflow_module.INVESTIGATION_LOOP_SPEC.threshold and not blocking_issues,
                    missing_sections=missing_sections,
                    blocking_issues=blocking_issues,
                    improvement_actions=improvement_actions,
                    previous_score=state["investigation_score"] if state["investigation_round"] > 1 else None,
                    prior_stagnated_rounds=(
                        state["investigation_loop_stagnated_rounds"] if state["investigation_round"] > 1 else 0
                    ),
                )
                feedback = workflow_module._compose_loop_feedback(
                    title="Investigation Confidence Review",
                    round_index=state["investigation_round"],
                    score=progress.score,
                    threshold=progress.threshold,
                    approved=progress.approved,
                    blocking_issues=list(progress.blocking_issues),
                    improvement_actions=list(progress.improvement_actions),
                    sections=normalized_checks,
                    loop_reason=progress.reason,
                )
                return (
                    {
                        "investigation_score": progress.score,
                        "investigation_feedback": feedback,
                        "investigation_missing_sections": list(progress.missing_sections),
                        "investigation_blocking_issues": list(progress.blocking_issues),
                        "investigation_improvement_actions": list(progress.improvement_actions),
                        "investigation_approved": progress.approved,
                        "investigation_loop_status": progress.status,
                        "investigation_loop_reason": progress.reason,
                        "investigation_loop_stagnated_rounds": progress.stagnated_rounds,
                        "active_loop_id": progress.loop_id,
                        "active_loop_round": progress.round_index,
                        "active_loop_score": progress.score,
                        "active_loop_threshold": progress.threshold,
                        "active_loop_max_rounds": progress.max_rounds,
                        "active_loop_status": progress.status,
                        "active_loop_reason": progress.reason,
                        "active_loop_should_continue": progress.should_continue,
                        "active_loop_completed": progress.completed,
                        "active_loop_stagnated_rounds": progress.stagnated_rounds,
                        "active_loop_missing_sections": list(progress.missing_sections),
                        "active_loop_blocking_issues": list(progress.blocking_issues),
                        "active_loop_improvement_actions": list(progress.improvement_actions),
                    },
                    normalized_checks,
                )

            run_dir = host_root / ".agentswarm" / "runs" / "strict-feature-loop"
            run_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch.object(
                workflow_module,
                "_evaluate_investigation_quality",
                new=staged_investigation_quality,
            ):
                result = engineer_graph.invoke(
                    {
                        "prompt": "Add gameplay feature: a wall jump should restore one spent air dash charge and keep the existing charge cap.",
                        "task_prompt": "Add gameplay feature: a wall jump should restore one spent air dash charge and keep the existing charge cap.",
                        "task_id": "task-1-add-wall-jump-air-dash-recharge",
                        "run_dir": str(run_dir),
                        "messages": [],
                    }
                )

            artifact_dir = Path(result["artifact_dir"])
            review_round_1 = (artifact_dir / "review_round_1.md").read_text(encoding="utf-8")
            review_round_2 = (artifact_dir / "review_round_2.md").read_text(encoding="utf-8")
            review_round_3 = (artifact_dir / "review_round_3.md").read_text(encoding="utf-8")
            plan_doc = (artifact_dir / "plan_doc.md").read_text(encoding="utf-8")

            self.assertEqual(result["task_type"], "feature")
            self.assertEqual(result["execution_track"], "feature")
            self.assertTrue(result["implementation_requested"])
            self.assertTrue(result["requires_architecture_review"])
            self.assertEqual(result["investigation_round"], 3)
            self.assertEqual(result["investigation_loop_status"], "passed")
            self.assertEqual(result["review_round"], 3)
            self.assertEqual(result["review_loop_status"], "passed")
            self.assertTrue(result["review_approved"])
            self.assertEqual(result["final_report"]["status"], "completed")
            self.assertTrue(result["compile_ok"])
            self.assertTrue(result["tests_ok"])
            self.assertEqual(result["workspace_source_file"], "src/traversal_runtime.py")
            self.assertEqual(result["workspace_test_file"], "tests/test_traversal_runtime.py")
            self.assertTrue((artifact_dir / "engineer_investigation_round_1.md").exists())
            self.assertTrue((artifact_dir / "engineer_investigation_round_2.md").exists())
            self.assertTrue((artifact_dir / "engineer_investigation_round_3.md").exists())
            self.assertTrue((artifact_dir / "investigation_review_round_1.md").exists())
            self.assertTrue((artifact_dir / "investigation_review_round_2.md").exists())
            self.assertTrue((artifact_dir / "investigation_review_round_3.md").exists())
            self.assertTrue((artifact_dir / "review_round_1.md").exists())
            self.assertTrue((artifact_dir / "review_round_2.md").exists())
            self.assertTrue((artifact_dir / "review_round_3.md").exists())
            self.assertIn("- Approved: False", review_round_1)
            self.assertIn("Implementation Steps: Name the owning runtime file and the exact wall-jump recharge hook.", review_round_1)
            self.assertIn("- Approved: False", review_round_2)
            self.assertIn("Acceptance Criteria: Add a player-visible pass condition for the non-wall-jump aerial path.", review_round_2)
            self.assertIn("- Approved: True", review_round_3)
            self.assertIn("The plan is ready for implementation.", review_round_3)
            self.assertIn("non-wall-jump", plan_doc)
            self.assertIn("## Previous Learning Summary", strict_manager.client.strategy_inputs[1])
            self.assertIn("still needs another pass", strict_manager.client.strategy_inputs[1])
            self.assertIn("## Previous Retained Evidence", strict_manager.client.strategy_inputs[2])
            self.assertIn("src/traversal_runtime.py", strict_manager.client.strategy_inputs[2])
            self.assertIn("## Open blocking issues", strict_manager.client.plan_inputs[1])
            self.assertIn(
                "Implementation Steps: Name the owning runtime file and the exact wall-jump recharge hook.",
                strict_manager.client.plan_inputs[1],
            )
            self.assertIn(
                "Acceptance Criteria: Add a player-visible pass condition for the non-wall-jump aerial path.",
                strict_manager.client.plan_inputs[2],
            )
            self.assertEqual(len(strict_manager.client.review_inputs), 3)
            self.assertIn("Review round: 2", strict_manager.client.review_inputs[1])
            self.assertIn("Review round: 3", strict_manager.client.review_inputs[2])
            self.assertIn(
                "def on_wall_jump",
                (host_root / "src" / "traversal_runtime.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "test_non_wall_jump_path_leaves_charges_unchanged",
                (host_root / "tests" / "test_traversal_runtime.py").read_text(encoding="utf-8"),
            )

    def test_engineer_workflow_requires_two_investigation_rounds_before_converging(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-investigation-loop-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            (host_root / "Scripts").mkdir(parents=True, exist_ok=True)
            (host_root / "Validation").mkdir(parents=True, exist_ok=True)
            (host_root / "Scripts" / "runtime.py").write_text("def update_runtime():\n    return 'ok'\n", encoding="utf-8")
            (host_root / "Validation" / "test_runtime.py").write_text(
                "def test_runtime_smoke():\n    assert True\n",
                encoding="utf-8",
            )

            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=InvestigationRetryLLMManager(),
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "investigation-loop"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": "Fix player movement regression",
                    "task_prompt": "Fix player movement regression",
                    "task_id": "task-1-fix-player-movement-regression",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            review_round_1 = (artifact_dir / "investigation_review_round_1.md").read_text(encoding="utf-8")
            review_round_2 = (artifact_dir / "investigation_review_round_2.md").read_text(encoding="utf-8")
            self.assertEqual(result["investigation_round"], 2)
            self.assertEqual(result["investigation_loop_status"], "passed")
            self.assertIn("Scripts/runtime.py", result["source_hits"])
            self.assertIn("Validation/test_runtime.py", result["test_hits"])
            self.assertTrue((artifact_dir / "engineer_investigation_round_1.md").exists())
            self.assertTrue((artifact_dir / "investigation_review_round_1.md").exists())
            self.assertTrue((artifact_dir / "engineer_investigation_round_2.md").exists())
            self.assertTrue((artifact_dir / "investigation_review_round_2.md").exists())
            self.assertIn("Minimum verification depth: 2 round(s)", review_round_1)
            self.assertIn("Approved: False", review_round_1)
            self.assertIn("Approved: True", review_round_2)
            self.assertEqual(result["final_report"]["investigation_loop_status"], "passed")

    def test_engineer_workflow_carries_learning_between_investigation_rounds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-investigation-learning-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            (host_root / "docs" / "archive").mkdir(parents=True, exist_ok=True)
            (host_root / "src").mkdir(parents=True, exist_ok=True)
            (host_root / "tests").mkdir(parents=True, exist_ok=True)
            (host_root / "Gameplay").mkdir(parents=True, exist_ok=True)
            (host_root / "Checks").mkdir(parents=True, exist_ok=True)
            (host_root / "docs" / "archive" / "player_movement_notes.md").write_text(
                "# Player Movement Notes\n\nLegacy note from an older movement implementation.\n",
                encoding="utf-8",
            )
            (host_root / "Gameplay" / "player_movement.py").write_text(
                "def clear_player_movement_lock():\n    return True\n",
                encoding="utf-8",
            )
            (host_root / "Checks" / "test_player_movement.py").write_text(
                "def test_clear_player_movement_lock():\n    assert True\n",
                encoding="utf-8",
            )

            learning_manager = InvestigationLearningCarryForwardLLMManager()
            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=learning_manager,
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "investigation-learning"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": "Investigate player movement regression without changing files",
                    "task_prompt": "Investigate player movement regression without changing files",
                    "task_id": "task-1-investigate-player-movement-learning",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            self.assertEqual(result["investigation_round"], 2)
            self.assertEqual(result["investigation_loop_status"], "passed")
            self.assertFalse(result["implementation_requested"])
            self.assertIn("Gameplay/player_movement.py", result["current_runtime_paths"])
            self.assertIn("Checks/test_player_movement.py", result["test_hits"])
            self.assertTrue(result["investigation_learning_summary"])
            self.assertTrue(result["investigation_learning_focus"])
            self.assertTrue((artifact_dir / "investigation_learning_round_1.md").exists())
            self.assertTrue((artifact_dir / "investigation_learning_round_2.md").exists())
            self.assertTrue((artifact_dir / "investigation_learning_log.md").exists())
            self.assertGreaterEqual(len(learning_manager.client.strategy_inputs), 2)
            self.assertIn("## Previous Learning Summary", learning_manager.client.strategy_inputs[1])
            self.assertIn("still needs a stronger runtime owner", learning_manager.client.strategy_inputs[1])
            self.assertIn("## Previous Open Questions", learning_manager.client.strategy_inputs[1])
            self.assertIn("## Previous Rejected Evidence", learning_manager.client.strategy_inputs[1])

    def test_engineer_workflow_stops_after_repair_loop_stagnates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-repair-loop-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            (host_root / "src").mkdir(parents=True, exist_ok=True)
            (host_root / "tests").mkdir(parents=True, exist_ok=True)
            (host_root / "src" / "runtime.py").write_text("def update_runtime():\n    return 'ok'\n", encoding="utf-8")
            (host_root / "tests" / "test_runtime.py").write_text(
                "def test_runtime_smoke():\n    assert True\n",
                encoding="utf-8",
            )

            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=StuckRepairLLMManager(),
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "repair-loop"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": "Fix player movement regression",
                    "task_prompt": "Fix player movement regression",
                    "task_id": "task-1-fix-player-movement-regression",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            self.assertEqual(result["final_report"]["status"], "repair-blocked")
            self.assertEqual(result["repair_loop_status"], "stagnated")
            self.assertEqual(result["final_report"]["repair_loop_status"], "stagnated")
            self.assertTrue((artifact_dir / "repair_abort.md").exists())
            self.assertTrue((artifact_dir / "repair_round_3.md").exists())

    def test_engineer_workflow_caps_review_rounds_when_llm_never_produces_an_approvable_plan(self) -> None:
        always_bad_registry = load_workflows(
            project_root=self.project_root,
            workflows_root=self.workflows_root,
            llm_manager=AlwaysBadLLMManager(),
        )
        engineer_graph = always_bad_registry.get("gameplay-engineer-workflow").graph
        self.assertIsNotNone(engineer_graph)

        with tempfile.TemporaryDirectory(prefix="langgraph-review-cap-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = engineer_graph.invoke(
                {
                    "prompt": "Add a melee combo extension feature and keep 3C responsiveness stable",
                    "task_prompt": "Add a melee combo extension feature and keep 3C responsiveness stable",
                    "task_id": "task-1-add-melee-combo-feature-review-cap",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            self.assertTrue((artifact_dir / "review_abort.md").exists())

        self.assertEqual(result["review_round"], 3)
        self.assertEqual(result["final_report"]["status"], "review-blocked")
        self.assertEqual(result["final_report"]["review_loop_status"], "stagnated")
        self.assertFalse(result["final_report"]["compile_ok"])
        self.assertFalse(result["final_report"]["tests_ok"])
        self.assertIn("never reached gameplay plan approval", result["summary"])

    def test_engineer_workflow_accepts_reviewer_approval_below_perfect_score(self) -> None:
        mixed_registry = load_workflows(
            project_root=self.project_root,
            workflows_root=self.workflows_root,
            llm_manager=MixedLLMManager(),
        )
        engineer_graph = mixed_registry.get("gameplay-engineer-workflow").graph
        self.assertIsNotNone(engineer_graph)

        with tempfile.TemporaryDirectory(prefix="langgraph-review-approved-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = engineer_graph.invoke(
                {
                    "prompt": "Add a melee combo extension feature",
                    "task_prompt": "Add a melee combo extension feature",
                    "task_id": "task-1-add-melee-combo-feature",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            review_round_1 = (artifact_dir / "review_round_1.md").read_text(encoding="utf-8")
            review_round_2 = (artifact_dir / "review_round_2.md").read_text(encoding="utf-8")

        self.assertEqual(result["review_score"], 95)
        self.assertTrue(result["review_approved"])
        self.assertEqual(result["review_round"], 2)
        self.assertEqual(result["final_report"]["status"], "completed")
        self.assertEqual(result["final_report"]["review_loop_status"], "passed")
        self.assertIn("Approved: False", review_round_1)
        self.assertIn("Minimum review depth is 2 rounds", review_round_1)
        self.assertIn("Approved: True", review_round_2)
        self.assertIn("Loop Status: passed", review_round_2)
        self.assertIn("Risks: 5/10", review_round_2)

    def test_engineer_workflow_codegen_prompt_carries_workspace_context_into_repair_rounds(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-codegen-prompt-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            (host_root / "src").mkdir(parents=True, exist_ok=True)
            (host_root / "tests").mkdir(parents=True, exist_ok=True)
            (host_root / "src" / "player_movement.py").write_text(
                "\n".join(
                    [
                        "class PlayerCharacter:",
                        "    def __init__(self):",
                        "        self.can_move = False",
                        "",
                        "    def spawn(self):",
                        "        return self",
                        "",
                        "    def move(self):",
                        "        return self.can_move",
                    ]
                ),
                encoding="utf-8",
            )
            (host_root / "tests" / "test_player_movement.py").write_text(
                "\n".join(
                    [
                        "from src.player_movement import PlayerCharacter",
                        "",
                        "def test_player_cannot_move_before_spawn():",
                        "    assert PlayerCharacter().move() is False",
                    ]
                ),
                encoding="utf-8",
            )

            prompt_manager = RepairAwareCodegenLLMManager()
            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=prompt_manager,
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )
            engineer_graph = registry.get("gameplay-engineer-workflow").graph
            self.assertIsNotNone(engineer_graph)

            run_dir = host_root / ".agentswarm" / "runs" / "codegen-prompt"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = engineer_graph.invoke(
                {
                    "prompt": "Fix gameplay bug: the player cannot move after spawning",
                    "task_prompt": "Fix gameplay bug: the player cannot move after spawning",
                    "task_id": "task-1-fix-player-spawn-movement",
                    "run_dir": str(run_dir),
                    "messages": [],
                }
            )

            self.assertEqual(result["final_report"]["status"], "completed")
            self.assertTrue(result["compile_ok"])
            self.assertTrue(result["tests_ok"])
            self.assertGreaterEqual(len(prompt_manager.client.inputs), 2)
            self.assertIn("## Current source file contents", prompt_manager.client.inputs[0])
            self.assertIn("class PlayerCharacter:", prompt_manager.client.inputs[0])
            self.assertIn("## Current test file contents", prompt_manager.client.inputs[0])
            self.assertIn("test_player_cannot_move_before_spawn", prompt_manager.client.inputs[0])
            self.assertIn("## Previous self-test output", prompt_manager.client.inputs[1])
            self.assertIn("repair me", prompt_manager.client.inputs[1])

    def test_main_graph_xray_mermaid_includes_workflow_subgraphs(self) -> None:
        graph = build_main_graph(registry=self.registry, llm_manager=self.llm_manager)
        mermaid = graph.get_graph(xray=1).draw_mermaid()

        self.assertIn("subgraph agentswarm__gameplay-engineer-planner", mermaid)
        self.assertIn("subgraph agentswarm__gameplay-engineer-workflow", mermaid)
        self.assertIn("subgraph agentswarm__gameplay-reviewer-workflow", mermaid)

    def test_engineer_graph_xray_mermaid_includes_reviewer_subgraph(self) -> None:
        engineer_graph = self.registry.get("gameplay-engineer-workflow").graph
        self.assertIsNotNone(engineer_graph)

        mermaid = engineer_graph.get_graph(xray=1).draw_mermaid()
        self.assertIn("subgraph gameplay-reviewer-workflow", mermaid)
        self.assertIn("simulate_engineer_investigation", mermaid)
        self.assertNotIn("prepare_doc_search", mermaid)
        self.assertNotIn("capture_doc_context", mermaid)

    def test_initialize_host_project_scaffolds_overlay_files_in_submodule_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            agent_root = host_root / "AgentSwarm"
            agent_root.mkdir(parents=True, exist_ok=True)

            paths, created = initialize_host_project(agent_root=agent_root, host_root=host_root)

            self.assertTrue(paths.is_submodule)
            self.assertTrue(created)
            self.assertTrue(paths.config_path.exists())
            self.assertTrue(paths.manifest_path.exists())
            self.assertTrue((paths.project_workflows_root / ".gitkeep").exists())
            self.assertTrue((paths.project_tools_root / ".gitkeep").exists())
            self.assertTrue((paths.memory_root / "project" / ".gitkeep").exists())
            self.assertTrue((paths.memory_root / "agentswarm" / ".gitkeep").exists())
            self.assertTrue((paths.memory_root / "shared" / ".gitkeep").exists())

    def test_project_tool_override_wins_alias_while_agentswarm_tool_remains_accessible(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-tool-overlay-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            override_dir = paths.project_tools_root / "find-gameplay-blueprints"
            override_dir.mkdir(parents=True, exist_ok=True)
            (override_dir / "Tool.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: find-gameplay-blueprints",
                        "entry: entry.py",
                        "version: 1.0.0",
                        "output_mode: message",
                        "state_keys_shared:",
                        "  - messages",
                        "capabilities:",
                        "  - project override",
                        "---",
                        "Project override for tests.",
                    ]
                ),
                encoding="utf-8",
            )
            (override_dir / "entry.py").write_text(
                "\n".join(
                    [
                        "from langchain_core.tools import tool",
                        "",
                        "def build_tool(context, metadata):",
                        "    @tool(metadata.qualified_name, response_format='content_and_artifact')",
                        "    def find_gameplay_blueprints(task_prompt: str, scope: str = 'host_project'):",
                        "        '''Override blueprint finder.'''",
                        "        return 'project override', {'blueprint_hits': ['Content/project_override/BP_Test.uasset'], 'scope': scope}",
                        "    return find_gameplay_blueprints",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_tools(
                project_root=self.project_root,
                tools_root=self.tools_root,
                llm_manager=self.llm_manager,
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )

            preferred = registry.get("find-gameplay-blueprints")
            fallback = registry.get("agentswarm::find-gameplay-blueprints")

            self.assertEqual(preferred.metadata.namespace, "project")
            self.assertEqual(preferred.metadata.qualified_name, "project::find-gameplay-blueprints")
            self.assertEqual(fallback.metadata.namespace, "agentswarm")

    def test_project_workflow_override_wins_alias_while_agentswarm_workflow_remains_accessible(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-workflow-overlay-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
            override_dir = paths.project_workflows_root / "gameplay-engineer-workflow"
            override_dir.mkdir(parents=True, exist_ok=True)
            (override_dir / "Workflow.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: gameplay-engineer-workflow",
                        "entry: entry.py",
                        "version: 1.0.0",
                        "exposed: true",
                        "capabilities:",
                        "  - project workflow override",
                        "---",
                        "Project workflow override for tests.",
                    ]
                ),
                encoding="utf-8",
            )
            (override_dir / "entry.py").write_text(
                "\n".join(
                    [
                        "from langgraph.graph import END, START, StateGraph",
                        "from typing_extensions import TypedDict",
                        "",
                        "def build_graph(context, metadata):",
                        "    class State(TypedDict):",
                        "        summary: str",
                        "",
                        "    def summarize(state: State):",
                        "        return {'summary': 'project override active'}",
                        "",
                        "    graph = StateGraph(State)",
                        "    graph.add_node('summarize', summarize)",
                        "    graph.add_edge(START, 'summarize')",
                        "    graph.add_edge('summarize', END)",
                        "    return graph",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_agentswarm_config(paths)
            manifest = load_project_manifest(paths)
            registry = load_workflows(
                project_root=self.project_root,
                workflows_root=self.workflows_root,
                llm_manager=self.llm_manager,
                runtime_paths=paths,
                config=config,
                manifest=manifest,
            )

            preferred = registry.get("gameplay-engineer-workflow")
            fallback = registry.get("agentswarm::gameplay-engineer-workflow")

            self.assertEqual(preferred.metadata.namespace, "project")
            self.assertEqual(preferred.metadata.qualified_name, "project::gameplay-engineer-workflow")
            self.assertEqual(fallback.metadata.namespace, "agentswarm")

    def test_main_graph_checkpointer_tracks_thread_state(self) -> None:
        graph = build_main_graph(
            registry=self.registry,
            llm_manager=self.llm_manager,
            checkpointer=InMemorySaver(),
        )
        config = build_runtime_config("test-thread")

        with tempfile.TemporaryDirectory(prefix="langgraph-checkpoints-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = graph.invoke(
                build_initial_state(
                    prompt="Fix combat dodge cancel bug in melee gameplay and keep 3C responsiveness stable",
                    run_dir=str(run_dir),
                ),
                config,
            )

        snapshot = graph.get_state(config)
        history = list(graph.get_state_history(config))

        self.assertIn("gameplay-engineer-workflow", result["final_response"])
        self.assertEqual(snapshot.values["final_response"], result["final_response"])
        self.assertEqual(snapshot.config["configurable"]["thread_id"], "test-thread")
        self.assertGreaterEqual(len(history), 2)

    def test_main_passes_host_root_to_llm_manager_workdir(self) -> None:
        import main as main_module

        with tempfile.TemporaryDirectory(prefix="agentswarm-main-host-root-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            host_root.mkdir(parents=True, exist_ok=True)
            runtime_paths = resolve_runtime_paths(self.project_root, host_root=host_root)
            run_dir = runtime_paths.runs_root / "test-run"
            run_dir.mkdir(parents=True, exist_ok=True)

            args = SimpleNamespace(
                prompt="Fix gameplay bug: the player cannot move after spawning.",
                prompt_parts=[],
                host_root=str(host_root),
                thread_id="",
            )
            fake_config = SimpleNamespace(target_scope="host_project")
            fake_llm_manager = mock.Mock()
            fake_registry = mock.Mock()
            fake_graph = mock.Mock()
            fake_graph.invoke.return_value = {"final_response": "ok"}

            with mock.patch("main._parse_args", return_value=args):
                with mock.patch("main.initialize_host_project", return_value=(runtime_paths, [])):
                    with mock.patch("main.load_agentswarm_config", return_value=fake_config):
                        with mock.patch("main.load_project_manifest", return_value={}):
                            with mock.patch("main._build_run_dir", return_value=run_dir):
                                with mock.patch("main.LLMManager.from_env", return_value=fake_llm_manager) as from_env:
                                    with mock.patch("main.load_workflows", return_value=fake_registry):
                                        with mock.patch("main.build_main_graph", return_value=fake_graph):
                                            with mock.patch("main.build_initial_state", return_value={"prompt": "stub"}):
                                                with mock.patch(
                                                    "main.build_runtime_config",
                                                    return_value={"configurable": {"thread_id": "test-run"}},
                                                ):
                                                    with mock.patch("main.write_memory_summary"):
                                                        main_module.main()

        from_env.assert_called_once_with(working_directory=str(host_root))
        fake_graph.invoke.assert_called_once()

    def test_self_test_harness_supports_module_aliases_and___file__(self) -> None:
        engineer_entry = self.project_root / "Workflows" / "gameplay-engineer-workflow" / "entry.py"
        spec = importlib.util.spec_from_file_location("test_engineer_entry", engineer_entry)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        source_code = "\n".join(
            [
                "from pathlib import Path",
                "",
                "def build_gameplay_change_summary():",
                "    return {",
                '        "task_type": "bugfix",',
                '        "implementation_status": "ready-for-review",',
                '        "unit_tests": ["alias_import", "file_defined"],',
                '        "source_file": Path(__file__).name,',
                "    }",
            ]
        )
        test_code = "\n".join(
            [
                "import importlib",
                "from pathlib import Path",
                "",
                "def _load_builder():",
                '    for name in ["solution", "main", "gameplay_change_summary"]:',
                "        module = importlib.import_module(name)",
                '        if hasattr(module, "build_gameplay_change_summary"):',
                "            return module.build_gameplay_change_summary",
                '    raise AssertionError("builder not found")',
                "",
                "def test_alias_import_and_file_values():",
                "    builder = _load_builder()",
                "    result = builder()",
                '    assert result["source_file"] == "gameplay_change.py"',
                '    assert Path(__file__).name == "test_gameplay_change.py"',
                '    assert Path(__file__).with_name("gameplay_change.py").exists()',
                '    assert result["implementation_status"] == "ready-for-review"',
            ]
        )

        compile_ok, tests_ok, output = module._run_compile_and_tests(source_code, test_code)
        self.assertTrue(compile_ok, output)
        self.assertTrue(tests_ok, output)


if __name__ == "__main__":
    unittest.main()
