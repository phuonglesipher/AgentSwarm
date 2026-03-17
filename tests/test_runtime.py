from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from core.config_loader import load_agentswarm_config, load_project_manifest
from core.host_setup import initialize_host_project
from core.runtime_paths import resolve_runtime_paths
from core.tool_loader import load_tools
from core.workflow_loader import load_workflows
from core.graph_logging import GRAPH_DEBUG_TRACE_FILE
from core.main_graph import build_initial_state, build_main_graph, build_runtime_config


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
        if schema_name == "gameplay_implementation_medium":
            return {
                "implementation_medium": "cpp",
                "reason": "Return code-side execution so the test can focus on the feature review loop.",
            }
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
                        "- Inspect the melee combo state machine and the extension points for the new branch."
                        if feature_prompt
                        else "- Inspect the dodge cancel state machine and the melee timing gates."
                    ),
                    (
                        "- Add the new branch while preserving valid transition order."
                        if feature_prompt
                        else "- Update the timing logic while preserving valid transition order."
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
        if schema_name == "gameplay_implementation_medium":
            return {
                "implementation_medium": "cpp",
                "reason": "The matched test fixture uses source code files rather than Blueprint assets.",
            }
        if schema_name == "gameplay_plan_review":
            return {
                "score": 95,
                "feedback": "The plan is implementation-ready. Only the task type rationale could be more explicit.",
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
                        "score": 5,
                        "status": "needs-work",
                        "rationale": "The plan names the task type but does not fully justify it.",
                        "action_items": [
                            (
                                "Add one sentence justifying why this work is classified as a feature."
                                if feature_prompt
                                else "Add one sentence justifying why this work is classified as a bugfix."
                            )
                        ],
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
                        "score": 10,
                        "status": "pass",
                        "rationale": "Risks and mitigations are documented.",
                        "action_items": [],
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
                "improvement_actions": [
                    (
                        "Add one sentence justifying why this work is classified as a feature."
                        if feature_prompt
                        else "Add one sentence justifying why this work is classified as a bugfix."
                    )
                ],
                "approved": True,
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
            return {
                "source_hits": [],
                "test_hits": [],
                "blueprint_hits": [],
                "blueprint_text_hits": [],
                "code_context": "Potential ownership is around src/combat_dodge.py and its regression tests.",
                "blueprint_context": "",
            }
        if schema_name == "gameplay_implementation_medium":
            return {
                "implementation_medium": "cpp",
                "reason": "The issue appears to be code-owned, not Blueprint-owned.",
            }
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
                return {
                    "source_hits": [],
                    "test_hits": [],
                    "blueprint_hits": [],
                    "blueprint_text_hits": [],
                    "code_context": "",
                    "blueprint_context": "",
                }
            return {
                "source_hits": ["src/runtime.py"],
                "test_hits": ["tests/test_runtime.py"],
                "blueprint_hits": [],
                "blueprint_text_hits": [],
                "code_context": "Movement ownership appears to live in src/runtime.py and its regression tests.",
                "blueprint_context": "",
            }
        if schema_name == "gameplay_implementation_medium":
            return {
                "implementation_medium": "cpp",
                "reason": "The strongest surviving evidence points to code-side movement logic.",
            }
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
            return {
                "source_hits": ["src/runtime.py"],
                "test_hits": ["tests/test_runtime.py"],
                "blueprint_hits": [],
                "blueprint_text_hits": [],
                "code_context": "The bug is likely in src/runtime.py and should be validated by tests/test_runtime.py.",
                "blueprint_context": "",
            }
        if schema_name == "gameplay_implementation_medium":
            return {
                "implementation_medium": "cpp",
                "reason": "The task is fully code-owned in this fixture.",
            }
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
                "find-gameplay-docs",
                "load-blueprint-context",
                "load-markdown-context",
                "load-source-context",
            ],
        )
        qualified_tool_names = [item.qualified_name for item in self.tool_registry.list_metadata()]
        self.assertEqual(
            qualified_tool_names,
            [
                "agentswarm::find-gameplay-blueprints",
                "agentswarm::find-gameplay-code",
                "agentswarm::find-gameplay-docs",
                "agentswarm::load-blueprint-context",
                "agentswarm::load-markdown-context",
                "agentswarm::load-source-context",
            ],
        )

        metadata = self.registry.list_metadata()
        names = [item.name for item in metadata]
        self.assertEqual(
            names,
            [
                "gameplay-engineer-workflow",
                "gameplay-reviewer-workflow",
            ],
        )

        exposed_names = [item.name for item in self.registry.list_metadata(exposed_only=True)]
        self.assertEqual(exposed_names, ["gameplay-engineer-workflow"])

        routed = self.registry.route("Fix a combat gameplay bug affecting melee dodge timing")
        self.assertIsNotNone(routed)
        self.assertEqual(routed.name, "gameplay-engineer-workflow")

    def test_reviewer_workflow_flags_missing_sections(self) -> None:
        result = self.registry.invoke(
            "gameplay-reviewer-workflow",
            {
                "task_prompt": "Fix combat dodge cancel bug",
                "plan_doc": "# Gameplay Implementation Plan\n\n## Overview\n- A short plan.",
                "review_round": 1,
            },
        )

        self.assertLess(result["score"], 100)
        self.assertIn("Unit Tests", result["missing_sections"])
        self.assertFalse(result["approved"])
        self.assertTrue(result["blocking_issues"])
        self.assertTrue(result["improvement_actions"])
        self.assertEqual(len(result["section_reviews"]), 7)
        self.assertEqual(result["loop_status"], "retry")
        self.assertTrue(result["loop_should_continue"])
        self.assertIn("## Section Scores", result["feedback"])

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
            self.assertTrue((artifact_dir / "plan_doc.md").exists())
            self.assertTrue((artifact_dir / "architecture_plan.md").exists())
            self.assertTrue((artifact_dir / "pull_request.md").exists())
            self.assertTrue((artifact_dir / "self_test.txt").exists())

            trace_output = trace_log.read_text(encoding="utf-8")
            self.assertIn("[main_graph] [analyze_prompt] ENTER", trace_output)
            self.assertIn("[main_graph] [dispatch_active_task] ROUTE", trace_output)
            self.assertIn("input_keys=", trace_output)
            self.assertIn("output_keys=", trace_output)
            self.assertIn(f"details={GRAPH_DEBUG_TRACE_FILE}#", trace_output)
            self.assertIn("next=agentswarm__gameplay-engineer-workflow", trace_output)
            self.assertIn("[gameplay-engineer-workflow] [gameplay-reviewer-workflow] SUBGRAPH_ENTER", trace_output)
            self.assertIn("[gameplay-engineer-workflow] [gameplay-reviewer-workflow] SUBGRAPH_EXIT", trace_output)
            self.assertIn("[gameplay-engineer-workflow] [request_review] ENTER", trace_output)
            self.assertIn("[gameplay-reviewer-workflow] [review_plan] ENTER", trace_output)
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
        self.assertIn("agentswarm__find-gameplay-docs", subgraphs)
        self.assertIn("agentswarm__load-markdown-context", subgraphs)
        self.assertNotIn("agentswarm__find-gameplay-blueprints", subgraphs)
        self.assertNotIn("agentswarm__find-gameplay-code", subgraphs)
        self.assertNotIn("agentswarm__load-blueprint-context", subgraphs)
        self.assertNotIn("agentswarm__load-source-context", subgraphs)

    def test_engineer_workflow_uses_doc_tools_and_internal_engineer_investigation(self) -> None:
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
                llm_manager=self.llm_manager,
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
            self.assertTrue((artifact_dir / "engineer_investigation.md").exists())

            tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
            self.assertEqual(
                [message.name for message in tool_messages],
                [
                    "agentswarm::find-gameplay-docs",
                    "agentswarm::load-markdown-context",
                ],
            )
            self.assertIn("docs/designer/combat_design_template.md", tool_messages[0].artifact["doc_hits"])
            self.assertIn("doc_context", tool_messages[1].artifact)

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
                llm_manager=self.llm_manager,
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
                llm_manager=self.llm_manager,
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
                llm_manager=self.llm_manager,
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

    def test_engineer_workflow_retries_investigation_until_context_is_reliable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-host-investigation-loop-") as temp_dir:
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
            self.assertEqual(result["investigation_round"], 2)
            self.assertEqual(result["investigation_loop_status"], "passed")
            self.assertTrue((artifact_dir / "engineer_investigation_round_1.md").exists())
            self.assertTrue((artifact_dir / "investigation_review_round_2.md").exists())
            self.assertEqual(result["final_report"]["investigation_loop_status"], "passed")

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
                    "prompt": "Summarize one accessible gameplay doc without changing files",
                    "task_prompt": "Summarize one accessible gameplay doc without changing files",
                    "task_id": "task-1-summarize-doc",
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
        self.assertIn("never reached approval", result["summary"])

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
            review_round = (artifact_dir / "review_round_1.md").read_text(encoding="utf-8")

        self.assertEqual(result["review_score"], 95)
        self.assertTrue(result["review_approved"])
        self.assertEqual(result["review_round"], 1)
        self.assertEqual(result["final_report"]["status"], "completed")
        self.assertEqual(result["final_report"]["review_loop_status"], "passed")
        self.assertIn("Approved: True", review_round)
        self.assertIn("Loop Status: passed", review_round)
        self.assertIn("Task Type: 5/10", review_round)

    def test_main_graph_xray_mermaid_includes_workflow_subgraphs(self) -> None:
        graph = build_main_graph(registry=self.registry, llm_manager=self.llm_manager)
        mermaid = graph.get_graph(xray=1).draw_mermaid()

        self.assertIn("subgraph agentswarm__gameplay-engineer-workflow", mermaid)
        self.assertIn("subgraph agentswarm__gameplay-reviewer-workflow", mermaid)

    def test_engineer_graph_xray_mermaid_includes_reviewer_subgraph(self) -> None:
        engineer_graph = self.registry.get("gameplay-engineer-workflow").graph
        self.assertIsNotNone(engineer_graph)

        mermaid = engineer_graph.get_graph(xray=1).draw_mermaid()
        self.assertIn("subgraph gameplay-reviewer-workflow", mermaid)
        self.assertIn("subgraph agentswarm__find-gameplay-docs", mermaid)
        self.assertIn("subgraph agentswarm__load-markdown-context", mermaid)
        self.assertIn("simulate_engineer_investigation", mermaid)

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
            override_dir = paths.project_tools_root / "find-gameplay-docs"
            override_dir.mkdir(parents=True, exist_ok=True)
            (override_dir / "Tool.md").write_text(
                "\n".join(
                    [
                        "---",
                        "name: find-gameplay-docs",
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
                        "    def find_gameplay_docs(task_prompt: str, scope: str = 'host_project'):",
                        "        '''Override doc finder.'''",
                        "        return 'project override', {'doc_hits': ['docs/project_override.md'], 'scope': scope}",
                        "    return find_gameplay_docs",
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

            preferred = registry.get("find-gameplay-docs")
            fallback = registry.get("agentswarm::find-gameplay-docs")

            self.assertEqual(preferred.metadata.namespace, "project")
            self.assertEqual(preferred.metadata.qualified_name, "project::find-gameplay-docs")
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
