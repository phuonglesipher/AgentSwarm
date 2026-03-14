from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
import unittest

from core.blueprint_loader import load_blueprints
from core.main_graph import build_main_graph


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


class BlueprintDrivenRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.blueprints_root = cls.project_root / "Blueprints"
        cls.llm_manager = FakeLLMManager()
        cls.registry = load_blueprints(
            project_root=cls.project_root,
            blueprints_root=cls.blueprints_root,
            llm_manager=cls.llm_manager,
        )

    def test_load_blueprints_and_route_gameplay_task(self) -> None:
        metadata = self.registry.list_metadata()
        names = [item.name for item in metadata]
        self.assertEqual(
            names,
            [
                "gameplay-engineer-blueprint",
                "gameplay-reviewer-blueprint",
            ],
        )

        exposed_names = [item.name for item in self.registry.list_metadata(exposed_only=True)]
        self.assertEqual(exposed_names, ["gameplay-engineer-blueprint"])

        routed = self.registry.route("Fix a combat gameplay bug affecting melee dodge timing")
        self.assertIsNotNone(routed)
        self.assertEqual(routed.name, "gameplay-engineer-blueprint")

    def test_reviewer_blueprint_flags_missing_sections(self) -> None:
        result = self.registry.invoke(
            "gameplay-reviewer-blueprint",
            {
                "task_prompt": "Fix combat dodge cancel bug",
                "plan_doc": "# Gameplay Implementation Plan\n\n## Overview\n- A short plan.",
                "review_round": 1,
            },
        )

        self.assertLess(result["score"], 100)
        self.assertIn("Unit Tests", result["missing_sections"])
        self.assertFalse(result["approved"])

    def test_main_graph_runs_end_to_end_without_llm(self) -> None:
        graph = build_main_graph(registry=self.registry, llm_manager=self.llm_manager)
        with tempfile.TemporaryDirectory(prefix="langgraph-tests-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = graph.invoke(
                {
                    "prompt": "Fix combat dodge cancel bug in melee gameplay and keep 3C responsiveness stable",
                    "run_dir": str(run_dir),
                    "tasks": [],
                    "results": [],
                    "final_response": "",
                    "routing_notes": [],
                }
            )

            self.assertIn("gameplay-engineer-blueprint", result["final_response"])
            self.assertTrue((run_dir / "summary.md").exists())

            artifact_dir = (
                run_dir
                / "tasks"
                / "task-1-fix-combat-dodge-cancel-bug-in-melee-gameplay"
                / "gameplay-engineer-blueprint"
            )
            self.assertTrue((artifact_dir / "plan_doc.md").exists())
            self.assertTrue((artifact_dir / "pull_request.md").exists())
            self.assertTrue((artifact_dir / "self_test.txt").exists())

    def test_self_test_harness_supports_module_aliases_and___file__(self) -> None:
        engineer_entry = self.project_root / "Blueprints" / "gameplay-engineer-blueprint" / "entry.py"
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
