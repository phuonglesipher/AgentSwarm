from __future__ import annotations

import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest


class CompareCodexCliVsAgentSwarmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compare_script = Path(__file__).resolve().parent / "compare_codex_cli_vs_agentswarm.py"
        spec = importlib.util.spec_from_file_location("test_compare_codex_cli_vs_agentswarm_module", compare_script)
        if spec is None or spec.loader is None:
            raise AssertionError(f"Unable to load compare harness from {compare_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.module = module

    def test_build_report_includes_loop_by_loop_comparison_sections(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-compare-report-") as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "source"
            direct_root = temp_root / "direct"
            agent_root = temp_root / "agent"
            (source_root / "src").mkdir(parents=True, exist_ok=True)
            (source_root / "tests").mkdir(parents=True, exist_ok=True)
            (source_root / "src" / "player_movement.py").write_text(
                "class PlayerCharacter:\n    pass\n",
                encoding="utf-8",
            )
            (source_root / "tests" / "test_player_movement.py").write_text(
                "def test_smoke():\n    assert True\n",
                encoding="utf-8",
            )

            shutil.copytree(source_root, direct_root)
            shutil.copytree(source_root, agent_root)

            (direct_root / "src" / "player_movement.py").write_text(
                "class PlayerCharacter:\n    def spawn(self):\n        return self\n",
                encoding="utf-8",
            )
            (direct_root / "tests" / "test_player_movement.py").write_text(
                "def test_player_can_move_after_spawn():\n    assert True\n",
                encoding="utf-8",
            )
            (agent_root / "src" / "player_movement.py").write_text(
                "class PlayerCharacter:\n    def spawn(self):\n        return self\n",
                encoding="utf-8",
            )
            (agent_root / "tests" / "test_player_movement.py").write_text(
                "def test_player_can_move_after_spawn():\n    assert True\n",
                encoding="utf-8",
            )

            artifact_dir = (
                agent_root
                / ".agentswarm"
                / "runs"
                / "20260318_120000"
                / "tasks"
                / "task-1-fix-player-movement"
                / "gameplay-engineer-workflow"
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "engineer_investigation_round_1.md").write_text(
                "\n".join(
                    [
                        "# Engineer Investigation",
                        "",
                        "Implementation Medium: cpp",
                        "",
                        "## Current Runtime Paths",
                        "- None.",
                        "",
                        "## Source Hits",
                        "- docs/archive/player_notes.md",
                        "",
                        "## Test Hits",
                        "- None.",
                        "",
                        "## Ownership Summary",
                        "The first round still needs a live runtime owner.",
                    ]
                ),
                encoding="utf-8",
            )
            (artifact_dir / "investigation_review_round_1.md").write_text(
                "\n".join(
                    [
                        "# Investigation Confidence Review",
                        "",
                        "- Decision: Needs another pass",
                        "- Score: 35/80",
                        "",
                        "## Blocking Issues",
                        "- Identify the current code or Blueprint runtime owner before coding.",
                    ]
                ),
                encoding="utf-8",
            )
            (artifact_dir / "engineer_investigation_round_2.md").write_text(
                "\n".join(
                    [
                        "# Engineer Investigation",
                        "",
                        "Implementation Medium: cpp",
                        "",
                        "## Current Runtime Paths",
                        "- src/player_movement.py",
                        "",
                        "## Source Hits",
                        "- src/player_movement.py",
                        "",
                        "## Test Hits",
                        "- tests/test_player_movement.py",
                        "",
                        "## Ownership Summary",
                        "src/player_movement.py is the live runtime owner and tests/test_player_movement.py validates it.",
                    ]
                ),
                encoding="utf-8",
            )
            (artifact_dir / "investigation_review_round_2.md").write_text(
                "\n".join(
                    [
                        "# Investigation Confidence Review",
                        "",
                        "- Decision: Approved",
                        "- Score: 100/80",
                        "",
                        "## Blocking Issues",
                        "- None.",
                    ]
                ),
                encoding="utf-8",
            )
            (artifact_dir / "review_round_1.md").write_text(
                "\n".join(
                    [
                        "# Review Round 1",
                        "",
                        "- Score: 42",
                        "- Approved: False",
                        "- Loop Status: running",
                        "",
                        "## Blocking Issues",
                        "- Unit Tests: Add the exact automated assertions.",
                        "",
                        "## Improvement Checklist",
                        "- Add a regression test for the spawn path.",
                    ]
                ),
                encoding="utf-8",
            )
            (artifact_dir / "review_round_2.md").write_text(
                "\n".join(
                    [
                        "# Review Round 2",
                        "",
                        "- Score: 96",
                        "- Approved: True",
                        "- Loop Status: passed",
                        "",
                        "## Blocking Issues",
                        "- None.",
                        "",
                        "## Improvement Checklist",
                        "- None.",
                    ]
                ),
                encoding="utf-8",
            )

            report = self.module.build_report(
                prompt="Fix gameplay bug: the player cannot move after spawning.",
                source_host_root=source_root,
                direct_host_root=direct_root,
                agentswarm_host_root=agent_root,
                agent_root=temp_root,
                direct_result=subprocess.CompletedProcess(args=["codex"], returncode=0, stdout="", stderr=""),
                direct_stdout=temp_root / "direct.stdout.txt",
                direct_stderr=temp_root / "direct.stderr.txt",
                agentswarm_result=subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr=""),
                agentswarm_stdout=temp_root / "agent.stdout.txt",
                agentswarm_stderr=temp_root / "agent.stderr.txt",
            )

            self.assertIn("## Direct Codex CLI Final Changed Files", report)
            self.assertIn("- src/player_movement.py", report)
            self.assertIn("- tests/test_player_movement.py", report)
            self.assertIn("## AgentSwarm Investigation Loop vs Direct Codex CLI", report)
            self.assertIn("### Investigation Round 1", report)
            self.assertIn("### Investigation Round 2", report)
            self.assertIn("Overlap with Direct final files: src/player_movement.py, tests/test_player_movement.py", report)
            self.assertIn("## AgentSwarm Architecture Review Loop vs Direct Codex CLI", report)
            self.assertIn("### Review Round 1", report)
            self.assertIn("Blocking issues: Unit Tests: Add the exact automated assertions.", report)
            self.assertIn("### Review Round 2", report)
            self.assertIn("Direct final test changes present already: True", report)

    def test_build_report_understands_root_project_investigation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-compare-root-investigation-") as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "source"
            direct_root = temp_root / "direct"
            agent_root = temp_root / "agent"
            (source_root / "docs").mkdir(parents=True, exist_ok=True)
            (source_root / "src").mkdir(parents=True, exist_ok=True)
            (source_root / "tests").mkdir(parents=True, exist_ok=True)
            (source_root / "docs" / "player_movement.md").write_text(
                "# Player Movement\n\nMovement should recover after respawn.\n",
                encoding="utf-8",
            )
            (source_root / "src" / "player_movement.py").write_text(
                "class PlayerMovement:\n    pass\n",
                encoding="utf-8",
            )
            (source_root / "tests" / "test_player_movement.py").write_text(
                "def test_smoke():\n    assert True\n",
                encoding="utf-8",
            )

            shutil.copytree(source_root, direct_root)
            shutil.copytree(source_root, agent_root)

            artifact_dir = (
                agent_root
                / ".agentswarm"
                / "runs"
                / "20260319_120000"
                / "tasks"
                / "task-1-investigate-player-movement"
                / "root-project-investigation-workflow"
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "investigation_round_1.md").write_text(
                "\n".join(
                    [
                        "# Root Project Investigation",
                        "",
                        "## Project Root Findings",
                        "- [docs/player_movement.md](C:/temp/docs/player_movement.md) defines the movement contract.",
                        "- [src/player_movement.py](C:/temp/src/player_movement.py) owns the recovery gate.",
                        "- [tests/test_player_movement.py](C:/temp/tests/test_player_movement.py) misses the failing order.",
                        "",
                        "## Candidate Ownership",
                        "- src/player_movement.py is the likely owner.",
                    ]
                ),
                encoding="utf-8",
            )
            (artifact_dir / "review_round_1.md").write_text(
                "\n".join(
                    [
                        "# Investigation Review",
                        "",
                        "Decision: APPROVE",
                        "Overall Score: 94/100",
                        "",
                        "## Blocking Issues",
                        "- None.",
                        "",
                        "## Improvement Checklist",
                        "- [x] No further investigation changes requested.",
                    ]
                ),
                encoding="utf-8",
            )
            (artifact_dir / "final_report.md").write_text(
                "\n".join(
                    [
                        "# Root Project Investigation Final Report",
                        "",
                        "- Status: completed",
                        "- Loop Status: passed",
                    ]
                ),
                encoding="utf-8",
            )

            report = self.module.build_report(
                prompt="Investigate the root cause of the player movement recovery regression without changing files.",
                source_host_root=source_root,
                direct_host_root=direct_root,
                agentswarm_host_root=agent_root,
                agent_root=temp_root,
                direct_result=subprocess.CompletedProcess(args=["codex"], returncode=0, stdout="", stderr=""),
                direct_stdout=temp_root / "direct.stdout.txt",
                direct_stderr=temp_root / "direct.stderr.txt",
                agentswarm_result=subprocess.CompletedProcess(args=["python"], returncode=0, stdout="", stderr=""),
                agentswarm_stdout=temp_root / "agent.stdout.txt",
                agentswarm_stderr=temp_root / "agent.stderr.txt",
            )

            self.assertIn("Artifact flavor: root-project-investigation", report)
            self.assertIn("Final workflow status: completed", report)
            self.assertIn("### Investigation Round 1", report)
            self.assertIn("Decision: APPROVE", report)
            self.assertIn("Score: 94/100", report)
            self.assertIn("Source hits: src/player_movement.py", report)
            self.assertIn("Test hits: tests/test_player_movement.py", report)


if __name__ == "__main__":
    unittest.main()
