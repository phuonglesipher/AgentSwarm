from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
import unittest

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

            artifact_dir = (
                run_dir
                / "tasks"
                / "task-1-add-a-melee-combo-extension-feature-and-keep"
                / "gameplay-engineer-workflow"
            )
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
        self.assertIn("agentswarm__find-gameplay-blueprints", subgraphs)
        self.assertIn("agentswarm__find-gameplay-code", subgraphs)
        self.assertIn("agentswarm__find-gameplay-docs", subgraphs)
        self.assertIn("agentswarm__load-blueprint-context", subgraphs)
        self.assertIn("agentswarm__load-markdown-context", subgraphs)
        self.assertIn("agentswarm__load-source-context", subgraphs)

    def test_engineer_workflow_uses_tool_messages_for_doc_and_code_context(self) -> None:
        engineer_graph = self.registry.get("gameplay-engineer-workflow").graph
        self.assertIsNotNone(engineer_graph)

        with tempfile.TemporaryDirectory(prefix="langgraph-tools-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
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

        self.assertIn("docs/designer/combat_design_template.md", result["doc_hits"])
        self.assertIn("# docs/designer/combat_design_template.md", result["doc_context"])
        self.assertIsInstance(result["source_hits"], list)
        self.assertIn("Generated code was saved to workflow artifacts only.", result["workspace_write_summary"])

        tool_messages = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        self.assertEqual(
            [message.name for message in tool_messages],
            [
                "agentswarm::find-gameplay-docs",
                "agentswarm::load-markdown-context",
                "agentswarm::find-gameplay-code",
                "agentswarm::load-source-context",
                "agentswarm::find-gameplay-blueprints",
                "agentswarm::load-blueprint-context",
            ],
        )
        self.assertIn("docs/designer/combat_design_template.md", tool_messages[0].artifact["doc_hits"])
        self.assertIn("doc_context", tool_messages[1].artifact)
        self.assertIn("source_hits", tool_messages[2].artifact)
        self.assertIn("code_context", tool_messages[3].artifact)
        self.assertIn("blueprint_hits", tool_messages[4].artifact)
        self.assertIn("blueprint_context", tool_messages[5].artifact)

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

            artifact_dir = run_dir / "tasks" / "task-1-fix-combat-dodge-window" / "gameplay-engineer-workflow"
            workspace_source = host_root / Path(result["workspace_source_file"])
            workspace_test = host_root / Path(result["workspace_test_file"])
            self.assertTrue(result["workspace_write_enabled"])
            self.assertTrue(workspace_source.exists())
            self.assertTrue(workspace_test.exists())
            self.assertIn("build_gameplay_change_summary", workspace_source.read_text(encoding="utf-8"))
            self.assertIn("test_build_gameplay_change_summary", workspace_test.read_text(encoding="utf-8"))
            self.assertTrue((artifact_dir / "workspace_write_manifest.md").exists())
            self.assertEqual(result["execution_track"], "bugfix")
            self.assertFalse(result["requires_architecture_review"])
            self.assertEqual(result["implementation_medium"], "cpp")
            self.assertEqual(result["review_round"], 0)
            self.assertTrue((artifact_dir / "bug_context.md").exists())
            self.assertFalse((artifact_dir / "review_round_1.md").exists())

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

            artifact_dir = run_dir / "tasks" / "task-1-fix-blueprint-dodge-cancel-bug" / "gameplay-engineer-workflow"
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

            artifact_dir = run_dir / "tasks" / "task-1-summarize-doc" / "gameplay-engineer-workflow"
            self.assertTrue((artifact_dir / "review_abort.md").exists())

        self.assertEqual(result["review_round"], 3)
        self.assertEqual(result["final_report"]["status"], "review-blocked")
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

            artifact_dir = run_dir / "tasks" / "task-1-add-melee-combo-feature" / "gameplay-engineer-workflow"
            review_round = (artifact_dir / "review_round_1.md").read_text(encoding="utf-8")

        self.assertEqual(result["review_score"], 95)
        self.assertTrue(result["review_approved"])
        self.assertEqual(result["review_round"], 1)
        self.assertEqual(result["final_report"]["status"], "completed")
        self.assertIn("Approved: True", review_round)
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
        self.assertIn("subgraph agentswarm__find-gameplay-blueprints", mermaid)
        self.assertIn("subgraph agentswarm__find-gameplay-code", mermaid)
        self.assertIn("subgraph agentswarm__find-gameplay-docs", mermaid)
        self.assertIn("subgraph agentswarm__load-blueprint-context", mermaid)
        self.assertIn("subgraph agentswarm__load-markdown-context", mermaid)
        self.assertIn("subgraph agentswarm__load-source-context", mermaid)

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
