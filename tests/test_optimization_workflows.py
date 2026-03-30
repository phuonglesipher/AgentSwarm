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


class LoopingOptimizationLLMClient:
    """Simulates a 2-round optimization investigation loop.

    Round 1 investigation: shallow analysis.
    Round 1 review: score 72/100, REVISE.
    Round 2 investigation: deeper analysis with reviewer feedback.
    Round 2 review: score 90/100, APPROVE.
    """

    def __init__(self) -> None:
        self.investigation_calls = 0
        self.review_calls = 0

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "looping optimization test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort
        # Check reviewer path first — it's more specific.
        if "reviewing an optimization investigation brief" in instructions.lower():
            self.review_calls += 1
            if self.review_calls == 1:
                return "\n".join(
                    [
                        "# Optimization Investigation Review",
                        "",
                        "Decision: REVISE",
                        "Overall Score: 72/100",
                        "",
                        "## Criterion Scores",
                        "- Problem Scoping: 12/15 - Goal is stated but target budget is vague.",
                        "- Profiling Rigor: 15/25 - Numbers are present but not broken down per system.",
                        "- System-Specific Analysis: 14/20 - AI and physics are mentioned but GAS is thin.",
                        "- Optimization Quality: 14/20 - Recommendations lack concrete ms estimates.",
                        "- Risk & Regression: 7/10 - Risk is acknowledged but not quantified.",
                        "- Verification Completeness: 7/10 - Plan is generic.",
                        "",
                        "## Blocking Issues",
                        "- Profiling Rigor: Provide per-system ms breakdown, not just totals.",
                        "- Optimization Quality: Include concrete ms savings estimates per recommendation.",
                        "",
                        "## Improvement Checklist",
                        "- [ ] Break down profiling data per system (AI, physics, GAS).",
                        "- [ ] Add concrete ms savings estimates per recommendation.",
                        "",
                        "## Senior Engineer Notes",
                        "Good direction but needs more concrete profiling data to be actionable.",
                    ]
                )
            return "\n".join(
                [
                    "# Optimization Investigation Review",
                    "",
                    "Decision: APPROVE",
                    "Overall Score: 90/100",
                    "",
                    "## Criterion Scores",
                    "- Problem Scoping: 14/15 - Clear target budget and current measurement.",
                    "- Profiling Rigor: 22/25 - Per-system breakdown with ms numbers.",
                    "- System-Specific Analysis: 18/20 - AI, physics, and GAS are well-covered.",
                    "- Optimization Quality: 18/20 - Ranked recommendations with ms estimates.",
                    "- Risk & Regression: 9/10 - Risks are identified per recommendation.",
                    "- Verification Completeness: 9/10 - Concrete measurement plan.",
                    "",
                    "## Blocking Issues",
                    "- None.",
                    "",
                    "## Improvement Checklist",
                    "- [x] No further investigation changes requested.",
                    "",
                    "## Senior Engineer Notes",
                    "Strong evidence-based investigation with actionable recommendations.",
                ]
            )

        # Investigation path — investigator LLM.
        if "optimize-gamethread-workflow" in instructions.lower() or "game thread performance" in instructions.lower():
            self.investigation_calls += 1
            if self.investigation_calls == 1:
                return "\n".join(
                    [
                        "# Game Thread Optimization Investigation",
                        "",
                        "## Task Framing",
                        "- Request: investigate game thread performance bottleneck.",
                        "",
                        "## Profiling Evidence",
                        "- stat unit shows 18ms game thread, target is 16.6ms.",
                        "",
                        "## Hot Path Identification",
                        "- TickComponent in combat components appears expensive.",
                        "",
                        "## GAS & Ability Overhead",
                        "- 50+ active GameplayEffects per frame.",
                        "",
                        "## AI System Load",
                        "- 50 AI agents ticking every frame via SipherAIScalableFramework.",
                        "",
                        "## Physics Query Audit",
                        "- ~200 overlap queries per frame from combat targeting.",
                        "",
                        "## Batch & Throttle Opportunities",
                        "- AI perception could use significance-based throttling.",
                        "",
                        "## Regression Risk",
                        "- Throttling AI may delay reaction times.",
                        "",
                        "## Optimization Recommendations",
                        "- Throttle AI perception to every 3 frames for distant agents.",
                        "",
                        "## Verification Plan",
                        "- Measure stat unit before and after throttling change.",
                    ]
                )

            return "\n".join(
                [
                    "# Game Thread Optimization Investigation",
                    "",
                    "## Task Framing",
                    "- Request: investigate game thread performance bottleneck.",
                    "- Scope: focus on AI system load and physics query overhead.",
                    "- Revision goal: answer the previous senior review with concrete evidence.",
                    "",
                    "## Profiling Evidence",
                    "- stat unit: game thread 18.2ms avg, 22ms spikes during combat.",
                    "- stat game: TickComponent total 4.1ms, AI tick 2.3ms, physics queries 1.8ms.",
                    "",
                    "## Hot Path Identification",
                    "- USipherEnemyCombatComponent::TickComponent: 0.8ms across 50 enemies.",
                    "- USipherAIController::Tick: 2.3ms total across all AI agents.",
                    "- Physics overlap queries: 1.8ms for 200+ queries/frame.",
                    "",
                    "## GAS & Ability Overhead",
                    "- 50 active GameplayEffects evaluated per frame.",
                    "- Attribute replication at 30Hz, acceptable.",
                    "",
                    "## AI System Load",
                    "- SipherAIScalableFramework runs all 50 agents at full tick rate.",
                    "- Behavior Tree evaluation: 0.04ms per agent, 2.0ms total.",
                    "- Perception queries: 0.006ms per agent per query, 0.3ms total.",
                    "",
                    "## Physics Query Audit",
                    "- 200 OverlapMultiByChannel calls per frame from combat targeting.",
                    "- 50 LineTrace calls per frame from AI line-of-sight checks.",
                    "- Collision channels properly configured, no unnecessary complexity.",
                    "",
                    "## Batch & Throttle Opportunities",
                    "- Distant AI agents (>30m) can use 3-frame tick interval: saves ~1.2ms.",
                    "- Combat targeting can batch overlap queries per group: saves ~0.5ms.",
                    "- GAS evaluation can defer non-critical effects: saves ~0.3ms.",
                    "",
                    "## Regression Risk",
                    "- AI tick throttling may delay reaction for distant agents by 2 frames.",
                    "- Batched physics queries may miss intra-frame position changes.",
                    "- Deferred GAS evaluation may delay visual feedback by 1 frame.",
                    "",
                    "## Optimization Recommendations",
                    "1. Throttle distant AI tick to 3-frame interval: ~1.2ms savings, low risk.",
                    "2. Batch combat overlap queries by group: ~0.5ms savings, medium risk.",
                    "3. Defer non-critical GAS evaluation: ~0.3ms savings, low risk.",
                    "- Total expected savings: ~2.0ms, bringing game thread to ~16.2ms.",
                    "",
                    "## Verification Plan",
                    "- Measure stat unit and stat game before and after each change.",
                    "- Run 50-AI combat stress test with Insights markers.",
                    "- Verify AI reaction time stays within 100ms tolerance.",
                ]
            )

        raise AssertionError(f"Unexpected generate_text instructions: {instructions[:200]}")


class LoopingOptimizationLLMManager:
    def __init__(self) -> None:
        self._client = LoopingOptimizationLLMClient()

    def resolve(self, profile: str | None = None) -> LoopingOptimizationLLMClient:
        return self._client

    def is_enabled(self, profile: str | None = None) -> bool:
        return True

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: looping optimization test client"

    def available_profiles(self) -> list[str]:
        return ["default", "reviewer"]


class ProcessOnlyOptimizationLLMClient:
    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "process-only optimization review test client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        del effort
        # Check reviewer path first — more specific match.
        if "reviewing an optimization investigation brief" in instructions.lower():
            return "\n".join(
                [
                    "# Optimization Investigation Review",
                    "",
                    "Decision: REVISE",
                    "Overall Score: 85/100",
                    "",
                    "## Criterion Scores",
                    "- Problem Scoping: 13/15 - Clear goal and target.",
                    "- Profiling Rigor: 22/25 - Good per-system breakdown.",
                    "- System-Specific Analysis: 17/20 - AI, physics covered well.",
                    "- Optimization Quality: 17/20 - Ranked with ms estimates.",
                    "- Risk & Regression: 8/10 - Risks identified.",
                    "- Verification Completeness: 8/10 - Measurement plan exists.",
                    "",
                    "## Blocking Issues",
                    "- Profile in production build before shipping.",
                    "- Test on actual hardware devkit.",
                    "",
                    "## Improvement Checklist",
                    "- [ ] Profile in production build before shipping.",
                    "- [ ] Test on actual hardware devkit.",
                    "",
                    "## Senior Engineer Notes",
                    "Technically strong; remaining asks are about production validation.",
                ]
            )

        # Investigation path.
        if "optimize-gamethread-workflow" in instructions.lower() or "game thread performance" in instructions.lower():
            return "\n".join(
                [
                    "# Game Thread Optimization Investigation",
                    "",
                    "## Task Framing",
                    "- Request: investigate game thread tick cost.",
                    "- Target: 16.6ms game thread budget.",
                    "",
                    "## Profiling Evidence",
                    "- stat unit: 18ms game thread avg.",
                    "- stat game: TickComponent 4.1ms, AI 2.3ms.",
                    "",
                    "## Hot Path Identification",
                    "- AI tick dominates at 2.3ms across 50 agents.",
                    "",
                    "## GAS & Ability Overhead",
                    "- 50 active effects, 0.5ms evaluation per frame.",
                    "",
                    "## AI System Load",
                    "- 50 agents full tick, 2.3ms total.",
                    "",
                    "## Physics Query Audit",
                    "- 200 overlaps/frame, 1.8ms total.",
                    "",
                    "## Batch & Throttle Opportunities",
                    "- Throttle distant AI: saves 1.2ms.",
                    "",
                    "## Regression Risk",
                    "- Throttling may delay AI reaction by 2 frames.",
                    "",
                    "## Optimization Recommendations",
                    "1. Throttle distant AI: 1.2ms savings.",
                    "2. Batch physics queries: 0.5ms savings.",
                    "",
                    "## Verification Plan",
                    "- stat unit before/after, 50-AI stress test.",
                ]
            )

        raise AssertionError(f"Unexpected generate_text instructions: {instructions[:200]}")


class ProcessOnlyOptimizationLLMManager:
    def __init__(self) -> None:
        self._client = ProcessOnlyOptimizationLLMClient()

    def resolve(self, profile: str | None = None) -> ProcessOnlyOptimizationLLMClient:
        return self._client

    def is_enabled(self, profile: str | None = None) -> bool:
        return True

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: process-only optimization review test client"

    def available_profiles(self) -> list[str]:
        return ["default", "reviewer"]


class OptimizationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.workflows_root = cls.project_root / "Workflows"

    def _prepare_host_project(self, host_root: Path) -> None:
        (host_root / "docs").mkdir(parents=True, exist_ok=True)
        (host_root / "src").mkdir(parents=True, exist_ok=True)
        (host_root / "tests").mkdir(parents=True, exist_ok=True)
        (host_root / "Source" / "S2" / "Private" / "Components").mkdir(parents=True, exist_ok=True)
        (host_root / "Source" / "S2" / "Public" / "Components").mkdir(parents=True, exist_ok=True)
        (host_root / "Config").mkdir(parents=True, exist_ok=True)
        (host_root / "docs" / "performance.md").write_text(
            "# Performance\n\nTarget: 60 FPS with 50+ AI agents.\n",
            encoding="utf-8",
        )
        (host_root / "Source" / "S2" / "Public" / "Components" / "SipherCombatComponent.h").write_text(
            "// Combat component with tick function\nvoid TickComponent(float DeltaTime);\n",
            encoding="utf-8",
        )
        (host_root / "Config" / "DefaultEngine.ini").write_text(
            "[/Script/Engine.StreamingSettings]\nr.Streaming.PoolSize=1024\n",
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

    def test_reviewer_subgraph_registered_for_gamethread(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-opt-subgraph-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, DisabledLLMManager())

            workflow = registry.get("optimize-gamethread-workflow").graph
            reviewer_workflow = registry.get("optimize-investigation-reviewer-workflow").graph

            self.assertIsNotNone(workflow)
            self.assertIsNotNone(reviewer_workflow)
            subgraphs = dict(workflow.get_subgraphs())
            self.assertIn("optimize-investigation-reviewer-workflow", subgraphs)

    def test_reviewer_subgraph_registered_for_streaming(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-opt-streaming-subgraph-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, DisabledLLMManager())

            workflow = registry.get("optimize-streaming-workflow").graph
            self.assertIsNotNone(workflow)
            subgraphs = dict(workflow.get_subgraphs())
            self.assertIn("optimize-investigation-reviewer-workflow", subgraphs)

    def test_reviewer_subgraph_registered_for_rendering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-opt-rendering-subgraph-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, DisabledLLMManager())

            workflow = registry.get("optimize-rendering-workflow").graph
            self.assertIsNotNone(workflow)
            subgraphs = dict(workflow.get_subgraphs())
            self.assertIn("optimize-investigation-reviewer-workflow", subgraphs)

    def test_gamethread_workflow_blocks_without_reviewer_llm(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-opt-gamethread-fallback-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, DisabledLLMManager())

            workflow = registry.get("optimize-gamethread-workflow").graph
            self.assertIsNotNone(workflow)

            run_dir = host_root / ".agentswarm" / "runs" / "fallback"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = workflow.invoke(
                {
                    "task_prompt": "Investigate game thread performance bottleneck with 50+ AI",
                    "task_id": "task-1-gamethread-optimization",
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

    def test_gamethread_workflow_loops_until_review_score_reaches_threshold(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-opt-gamethread-loop-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, LoopingOptimizationLLMManager())

            workflow = registry.get("optimize-gamethread-workflow").graph
            self.assertIsNotNone(workflow)

            run_dir = host_root / ".agentswarm" / "runs" / "loop"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = workflow.invoke(
                {
                    "task_prompt": "Investigate game thread performance bottleneck with 50+ AI agents",
                    "task_id": "task-1-gamethread-optimization",
                    "run_dir": str(run_dir),
                }
            )

            artifact_dir = Path(result["artifact_dir"])
            self.assertEqual(result["investigation_round"], 2)
            self.assertEqual(result["review_round"], 2)
            self.assertEqual(result["review_score"], 90)
            self.assertTrue(result["review_approved"])
            self.assertEqual(result["final_report"]["status"], "completed")
            self.assertEqual(result["loop_status"], "passed")
            self.assertTrue((artifact_dir / "investigation_round_2.md").exists())
            self.assertTrue((artifact_dir / "review_round_2.md").exists())
            self.assertIn("passed senior review", result["summary"])

    def test_process_only_optimization_feedback_does_not_block_approval(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-opt-process-only-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, ProcessOnlyOptimizationLLMManager())

            workflow = registry.get("optimize-gamethread-workflow").graph
            self.assertIsNotNone(workflow)

            run_dir = host_root / ".agentswarm" / "runs" / "process-only"
            run_dir.mkdir(parents=True, exist_ok=True)
            result = workflow.invoke(
                {
                    "task_prompt": "Investigate game thread tick cost for 50+ AI optimization",
                    "task_id": "task-1-gamethread-optimization",
                    "run_dir": str(run_dir),
                }
            )

            self.assertEqual(result["investigation_round"], 2)
            self.assertEqual(result["review_round"], 2)
            self.assertEqual(result["review_score"], 85)
            self.assertTrue(result["review_approved"])
            self.assertEqual(result["review_blocking_issues"], [])
            self.assertEqual(result["review_improvement_actions"], [])
            self.assertEqual(result["final_report"]["status"], "completed")

    def test_all_optimization_workflows_are_discoverable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-opt-discovery-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            self._prepare_host_project(host_root)
            registry = self._load_registry(host_root, DisabledLLMManager())

            gamethread = registry.get("optimize-gamethread-workflow")
            streaming = registry.get("optimize-streaming-workflow")
            rendering = registry.get("optimize-rendering-workflow")
            reviewer = registry.get("optimize-investigation-reviewer-workflow")

            self.assertIsNotNone(gamethread)
            self.assertIsNotNone(streaming)
            self.assertIsNotNone(rendering)
            self.assertIsNotNone(reviewer)

            exposed = [m for m in registry.list_metadata() if m.exposed]
            exposed_names = {m.name for m in exposed}
            self.assertIn("optimize-gamethread-workflow", exposed_names)
            self.assertIn("optimize-streaming-workflow", exposed_names)
            self.assertIn("optimize-rendering-workflow", exposed_names)
            self.assertNotIn("optimize-investigation-reviewer-workflow", exposed_names)


if __name__ == "__main__":
    unittest.main()
