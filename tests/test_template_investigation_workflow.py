from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from core.config_loader import load_agentswarm_config, load_project_manifest
from core.host_setup import initialize_host_project
from core.workflow_loader import load_workflows


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


class DisabledLLMManager:
    def __init__(self) -> None:
        self._client = DisabledLLMClient()

    def resolve(self, profile: str | None = None) -> DisabledLLMClient:
        return self._client

    def is_enabled(self, profile: str | None = None) -> bool:
        return False

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: disabled test client"

    def available_profiles(self) -> list[str]:
        return ["default", "reviewer"]


class LoopingInvestigationLLMClient:
    def __init__(self) -> None:
        self.investigation_calls = 0
        self.review_calls = 0

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "looping investigation test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort
        if "Investigate the host project root like a senior engineer" in instructions:
            self.investigation_calls += 1
            if self.investigation_calls == 1:
                return "\n".join(
                    [
                        "# Template Investigation",
                        "",
                        "## Task Framing",
                        "- Request: investigate player movement regression.",
                        "",
                        "## Project Root Findings",
                        "- src/player_movement.py",
                        "",
                        "## Candidate Ownership",
                        "- src/player_movement.py",
                        "",
                        "## Root Cause Hypothesis",
                        "- The movement gate probably resets in the wrong state.",
                        "",
                        "## Architecture Notes",
                        "- Runtime state lives in the movement module.",
                        "",
                        "## Clean Code Notes",
                        "- Keep the change small.",
                        "",
                        "## Optimization Notes",
                        "- Check hot path assumptions before optimizing.",
                        "",
                        "## Verification Plan",
                        "- Add one regression test.",
                        "",
                        "## Open Questions",
                        "- Need better ownership detail.",
                    ]
                )

            if "Previous reviewer feedback:" not in input_text or "Overall Score:" not in input_text:
                raise AssertionError("Second investigation round should receive the previous review feedback.")
            return "\n".join(
                [
                    "# Template Investigation",
                    "",
                    "## Task Framing",
                    "- Request: investigate player movement regression.",
                    "- Scope: focus on the runtime reset path instead of adjacent movement polish.",
                    "- Revision goal: answer the previous senior review.",
                    "",
                    "## Project Root Findings",
                    "- docs/player_movement.md describes the expected movement recovery behavior.",
                    "- src/player_movement.py owns the runtime reset path.",
                    "- tests/test_player_movement.py validates the recovery sequence.",
                    "",
                    "## Candidate Ownership",
                    "- src/player_movement.py is the most likely owner because it controls `regain_control()` and the stalled state transition.",
                    "- tests/test_player_movement.py is the highest-value regression target for confirmation.",
                    "",
                    "## Root Cause Hypothesis",
                    "- The reset path likely leaves the player in a blocked state after recovery.",
                    "- Confirm whether `regain_control()` clears the same state that movement input checks later.",
                    "",
                    "## Architecture Notes",
                    "- The issue boundary is narrow: the runtime module plus its regression test, not the whole input stack.",
                    "- Validate the handoff between recovery completion and movement state re-enable before touching anything upstream.",
                    "",
                    "## Clean Code Notes",
                    "- Keep ownership obvious by fixing the reset at the source instead of adding another bypass flag.",
                    "- Preserve readability by pairing the runtime change with a targeted regression test and explicit state assertions.",
                    "",
                    "## Optimization Notes",
                    "- This is not a broad performance problem, but avoid repeated state checks in the recovery path while validating the fix.",
                    "- If a hot path shows up, optimize only the confirmed repeated branch in `src/player_movement.py`.",
                    "",
                    "## Verification Plan",
                    "- Reproduce the stuck movement flow through the recovery path.",
                    "- Add or update `tests/test_player_movement.py` to assert movement is restored after recovery.",
                    "- Compare the behavior before and after the suspected reset point to confirm ownership.",
                    "",
                    "## Open Questions",
                    "- Confirm whether the blocked state is also referenced by any adjacent air-movement helper.",
                ]
            )

        if "strict senior engineer reviewing an investigation brief" in instructions:
            self.review_calls += 1
            if self.review_calls == 1:
                return "\n".join(
                    [
                        "# Investigation Review",
                        "",
                        "Decision: REVISE",
                        "Overall Score: 78/100",
                        "",
                        "## Criterion Scores",
                        "- Focus: 20/25 - The brief has a direction but still feels broad.",
                        "- Evidence & Ownership: 12/20 - It names one file, but ownership is not grounded enough.",
                        "- Architecture: 14/20 - Some structure is present, but the system boundary needs more precision.",
                        "- Clean Code: 11/15 - Maintainability is mentioned, not argued.",
                        "- Optimization: 9/10 - Optimization non-goals are clear enough.",
                        "- Verification: 8/10 - Validation is too thin for confidence.",
                        "",
                        "## Blocking Issues",
                        "- Evidence & Ownership: Name the concrete files, modules, docs, or tests that most likely own the issue.",
                        "- Verification: Add concrete validation steps, regression checks, or measurements for the hypothesis.",
                        "",
                        "## Improvement Checklist",
                        "- [ ] Name the concrete files, modules, docs, or tests that most likely own the issue.",
                        "- [ ] Add concrete validation steps, regression checks, or measurements for the hypothesis.",
                        "",
                        "## Senior Engineer Notes",
                        "You are still close, but this is not yet a credible investigation handoff.",
                    ]
                )
            return "\n".join(
                [
                    "# Investigation Review",
                    "",
                    "Decision: APPROVE",
                    "Overall Score: 93/100",
                    "",
                    "## Criterion Scores",
                    "- Focus: 24/25 - The scope is tight and actionable.",
                    "- Evidence & Ownership: 18/20 - Ownership is grounded in the runtime file, docs, and tests.",
                    "- Architecture: 18/20 - The runtime boundary and handoff are clear.",
                    "- Clean Code: 14/15 - Change-safety and maintainability concerns are explicit.",
                    "- Optimization: 9/10 - Performance is treated as a non-goal unless evidence appears.",
                    "- Verification: 10/10 - The validation plan is now concrete.",
                    "",
                    "## Blocking Issues",
                    "- None.",
                    "",
                    "## Improvement Checklist",
                    "- [x] No further investigation changes requested.",
                    "",
                    "## Senior Engineer Notes",
                    "This is strong enough to move into implementation planning.",
                ]
            )

        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")


class LoopingInvestigationLLMManager:
    def __init__(self) -> None:
        self._client = LoopingInvestigationLLMClient()

    def resolve(self, profile: str | None = None) -> LoopingInvestigationLLMClient:
        return self._client

    def is_enabled(self, profile: str | None = None) -> bool:
        return True

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: looping investigation test client"

    def available_profiles(self) -> list[str]:
        return ["default", "reviewer"]


class ProcessOnlyReviewLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "process-only review test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del input_text, effort
        if "Investigate the host project root like a senior engineer" in instructions:
            return "\n".join(
                [
                    "# Template Investigation",
                    "",
                    "## Task Framing",
                    "- Request: investigate player movement regression.",
                    "- Scope: focus on the runtime reset path and its immediate test coverage.",
                    "- Revision goal: produce an implementation-ready investigation brief.",
                    "",
                    "## Project Root Findings",
                    "- docs/player_movement.md defines the expected post-recovery movement contract.",
                    "- src/player_movement.py owns the recovery gate and the movement-enable write.",
                    "- tests/test_player_movement.py covers only the happy-path call order today.",
                    "",
                    "## Candidate Ownership",
                    "- src/player_movement.py is the likely runtime owner.",
                    "- tests/test_player_movement.py is the highest-value regression target.",
                    "",
                    "## Root Cause Hypothesis",
                    "- Movement remains disabled if regain_control is called before recovery ends and is not called again afterward.",
                    "- The class currently ties movement restore to a method call, not to the recovery-end transition.",
                    "",
                    "## Architecture Notes",
                    "- The bug boundary is local to PlayerMovement state transitions, not the wider input stack.",
                    "- The recovery transition and movement-enable state should be owned by one authoritative path.",
                    "",
                    "## Clean Code Notes",
                    "- Keep the fix in the owning runtime path instead of layering another bypass flag.",
                    "- Pair the eventual change with explicit assertions so the invariant stays readable.",
                    "",
                    "## Optimization Notes",
                    "- Performance is not the main concern here; avoid speculative optimization.",
                    "- If the path becomes hot, optimize only the confirmed repeated branch.",
                    "",
                    "## Verification Plan",
                    "- Reproduce the failing call order in a focused regression test.",
                    "- Assert movement is restored immediately after recovery ends.",
                    "- Keep the existing opposite-order case as a safety check.",
                    "",
                    "## Open Questions",
                    "- Confirm whether any other mechanic may intentionally keep movement disabled after recovery.",
                ]
            )

        if "strict senior engineer reviewing an investigation brief" in instructions:
            return "\n".join(
                [
                    "# Investigation Review",
                    "",
                    "Decision: REVISE",
                    "Overall Score: 84/100",
                    "",
                    "## Criterion Scores",
                    "- Focus: 25/25 - The scope is tight and actionable.",
                    "- Evidence & Ownership: 20/20 - The likely owner and test surface are grounded in the project files.",
                    "- Architecture: 20/20 - The runtime boundary is clear.",
                    "- Clean Code: 15/15 - Maintainability concerns are explicit.",
                    "- Optimization: 10/10 - Optimization is handled proportionally.",
                    "- Verification: 10/10 - Validation steps are concrete.",
                    "",
                    "## Blocking Issues",
                    "- Assign a named DRI before implementation starts.",
                    "- Add commit provenance for the regression origin.",
                    "",
                    "## Improvement Checklist",
                    "- [ ] Assign a named DRI before implementation starts.",
                    "- [ ] Add commit provenance for the regression origin.",
                    "",
                    "## Senior Engineer Notes",
                    "Technically the investigation is strong; the remaining asks are process-related.",
                ]
            )

        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")


class ProcessOnlyReviewLLMManager:
    def __init__(self) -> None:
        self._client = ProcessOnlyReviewLLMClient()

    def resolve(self, profile: str | None = None) -> ProcessOnlyReviewLLMClient:
        return self._client

    def is_enabled(self, profile: str | None = None) -> bool:
        return True

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: process-only review test client"

    def available_profiles(self) -> list[str]:
        return ["default", "reviewer"]


class NonePrefixedApprovalLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "none-prefixed approval test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del input_text, effort
        if "Investigate the host project root like a senior engineer" in instructions:
            return "\n".join(
                [
                    "# Template Investigation",
                    "",
                    "## Task Framing",
                    "- Request: investigate player movement regression.",
                    "- Scope: prove the causal chain directly from doc, runtime code, and tests.",
                    "- Revision goal: produce a technically approvable investigation brief.",
                    "",
                    "## Project Root Findings",
                    "- [docs/player_movement.md](C:/temp/docs/player_movement.md) defines the post-recovery movement contract.",
                    "- [src/player_movement.py](C:/temp/src/player_movement.py) contains the early return while recovering.",
                    "- [tests/test_player_movement.py](C:/temp/tests/test_player_movement.py) only covers the happy order.",
                    "",
                    "## Candidate Ownership",
                    "- [src/player_movement.py](C:/temp/src/player_movement.py) is the primary runtime owner.",
                    "- [tests/test_player_movement.py](C:/temp/tests/test_player_movement.py) is the main regression surface.",
                    "",
                    "## Root Cause Hypothesis",
                    "- Movement remains disabled if regain_control runs before recovery ends and no second callback occurs.",
                    "- The class ties restoration to call order instead of the recovery-end transition itself.",
                    "",
                    "## Architecture Notes",
                    "- The bug boundary is local to movement-state transitions and their caller sequencing.",
                    "- The recovery-end transition and movement restore contract are not enforced atomically.",
                    "",
                    "## Clean Code Notes",
                    "- Public state flags allow invalid combinations to persist.",
                    "- The eventual fix should restore the invariant at the owning runtime path.",
                    "",
                    "## Optimization Notes",
                    "- This is correctness-first work; optimization is not the bottleneck.",
                    "- Avoid speculative tuning until the transition contract is stable.",
                    "",
                    "## Verification Plan",
                    "- Reproduce both happy and reversed call order.",
                    "- Add repeated-cycle checks once executable test tooling is available.",
                    "- Confirm the real integration caller if it exists outside this snapshot.",
                    "",
                    "## Open Questions",
                    "- Confirm whether any external system intentionally leaves movement disabled after recovery.",
                ]
            )

        if "strict senior engineer reviewing an investigation brief" in instructions:
            return "\n".join(
                [
                    "# Investigation Review",
                    "",
                    "Decision: REVISE",
                    "Overall Score: 100/100",
                    "",
                    "## Criterion Scores",
                    "- Focus: 25/25 - The investigation is tightly scoped.",
                    "- Evidence & Ownership: 20/20 - Ownership is grounded in project files and tests.",
                    "- Architecture: 20/20 - The boundary and handoff are clear.",
                    "- Clean Code: 15/15 - Maintainability concerns are explicit.",
                    "- Optimization: 10/10 - Optimization is treated proportionally.",
                    "- Verification: 10/10 - Verification is concrete and technical.",
                    "",
                    "## Blocking Issues",
                    "- None. Round-depth gate is satisfied (`2/2` minimum), and the causal chain is technically well-supported for investigation approval.",
                    "",
                    "## Improvement Checklist",
                    "- [ ] Add executable regression tests once pytest is available.",
                    "",
                    "## Senior Engineer Notes",
                    "Investigation quality is already strong enough to approve technically.",
                ]
            )

        raise AssertionError(f"Unexpected generate_text instructions: {instructions}")


class NonePrefixedApprovalLLMManager:
    def __init__(self) -> None:
        self._client = NonePrefixedApprovalLLMClient()

    def resolve(self, profile: str | None = None) -> NonePrefixedApprovalLLMClient:
        return self._client

    def is_enabled(self, profile: str | None = None) -> bool:
        return True

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: none-prefixed approval test client"

    def available_profiles(self) -> list[str]:
        return ["default", "reviewer"]


class TemplateInvestigationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.workflows_root = cls.project_root / "Workflows"

    def _prepare_host_project(self, host_root: Path) -> None:
        (host_root / "docs").mkdir(parents=True, exist_ok=True)
        (host_root / "src").mkdir(parents=True, exist_ok=True)
        (host_root / "tests").mkdir(parents=True, exist_ok=True)
        (host_root / "docs" / "player_movement.md").write_text(
            "# Player Movement\n\nMovement should recover after the respawn sequence.\n",
            encoding="utf-8",
        )
        (host_root / "src" / "player_movement.py").write_text(
            "\n".join(
                [
                    "def regain_control(state):",
                    "    if state.get('recovering'):",
                    "        state['can_move'] = True",
                    "    return state",
                ]
            ),
            encoding="utf-8",
        )
        (host_root / "tests" / "test_player_movement.py").write_text(
            "def test_movement_is_restored_after_recovery():\n    assert True\n",
            encoding="utf-8",
        )

    def _load_registry(self, host_root: Path, llm_manager) -> object:
        paths, _ = initialize_host_project(agent_root=self.project_root, host_root=host_root)
        config = load_agentswarm_config(paths)
        manifest = load_project_manifest(paths)
        return load_workflows(
            project_root=self.project_root,
            workflows_root=self.workflows_root,
            llm_manager=llm_manager,
            runtime_paths=paths,
            config=config,
            manifest=manifest,
        )

    def test_workflow_registers_reviewer_subgraph(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-root-investigation-subgraph-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, DisabledLLMManager())

            workflow = registry.get("template-investigation-workflow").graph
            reviewer_workflow = registry.get("template-investigation-reviewer-workflow").graph

            self.assertIsNotNone(workflow)
            self.assertIsNotNone(reviewer_workflow)
            subgraphs = dict(workflow.get_subgraphs())
            self.assertIn("template-investigation-reviewer-workflow", subgraphs)

    def test_workflow_blocks_without_reviewer_llm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-root-investigation-fallback-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, DisabledLLMManager())

            workflow = registry.get("template-investigation-workflow").graph
            self.assertIsNotNone(workflow)

            run_dir = host_root / ".agentswarm" / "runs" / "fallback"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = workflow.invoke(
                {
                    "task_prompt": "Investigate why player movement does not recover after a respawn flow",
                    "task_id": "task-1-player-movement-investigation",
                    "run_dir": str(run_dir),
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            self.assertEqual(result["review_score"], 0)
            self.assertFalse(result["review_approved"])
            self.assertEqual(result["investigation_round"], 1)
            self.assertEqual(result["review_round"], 1)
            self.assertEqual(result["final_report"]["status"], "review-blocked")
            self.assertEqual(result["loop_status"], "llm-unavailable")
            self.assertIn("Reviewer LLM is unavailable", result["review_feedback"])
            self.assertTrue((artifact_dir / "investigation_round_1.md").exists())
            self.assertTrue((artifact_dir / "review_round_1.md").exists())
            self.assertFalse((artifact_dir / "investigation_round_2.md").exists())
            self.assertFalse((artifact_dir / "review_round_2.md").exists())

    def test_workflow_loops_until_review_score_reaches_threshold(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-root-investigation-loop-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, LoopingInvestigationLLMManager())

            workflow = registry.get("template-investigation-workflow").graph
            self.assertIsNotNone(workflow)

            run_dir = host_root / ".agentswarm" / "runs" / "loop"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = workflow.invoke(
                {
                    "task_prompt": "Investigate the root cause of a player movement recovery regression",
                    "task_id": "task-1-player-movement-investigation",
                    "run_dir": str(run_dir),
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            self.assertEqual(result["investigation_round"], 2)
            self.assertEqual(result["review_round"], 2)
            self.assertEqual(result["review_score"], 93)
            self.assertTrue(result["review_approved"])
            self.assertEqual(result["final_report"]["status"], "completed")
            self.assertEqual(result["loop_status"], "passed")
            self.assertTrue((artifact_dir / "investigation_round_2.md").exists())
            self.assertTrue((artifact_dir / "review_round_2.md").exists())
            self.assertIn("passed senior review", result["summary"])

    def test_process_only_review_feedback_does_not_block_approval(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-root-investigation-process-only-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, ProcessOnlyReviewLLMManager())

            workflow = registry.get("template-investigation-workflow").graph
            self.assertIsNotNone(workflow)

            run_dir = host_root / ".agentswarm" / "runs" / "process-only"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = workflow.invoke(
                {
                    "task_prompt": "Investigate the root cause of a player movement recovery regression",
                    "task_id": "task-1-player-movement-investigation",
                    "run_dir": str(run_dir),
                }
            )

            self.assertEqual(result["investigation_round"], 2)
            self.assertEqual(result["review_round"], 2)
            self.assertEqual(result["review_score"], 100)
            self.assertTrue(result["review_approved"])
            self.assertEqual(result["review_blocking_issues"], [])
            self.assertEqual(result["review_improvement_actions"], [])
            self.assertEqual(result["final_report"]["status"], "completed")

    def test_review_parser_ignores_none_prefixed_blocker_explanations(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-root-investigation-none-prefixed-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, NonePrefixedApprovalLLMManager())

            workflow = registry.get("template-investigation-workflow").graph
            self.assertIsNotNone(workflow)

            run_dir = host_root / ".agentswarm" / "runs" / "none-prefixed"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = workflow.invoke(
                {
                    "task_prompt": "Investigate the root cause of a player movement recovery regression",
                    "task_id": "task-1-player-movement-investigation",
                    "run_dir": str(run_dir),
                }
            )

            self.assertEqual(result["review_round"], 2)
            self.assertEqual(result["review_score"], 100)
            self.assertTrue(result["review_approved"])
            self.assertEqual(result["review_blocking_issues"], [])
            self.assertEqual(result["final_report"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
