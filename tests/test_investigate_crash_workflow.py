from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from core.config_loader import load_agentswarm_config, load_project_manifest
from core.host_setup import initialize_host_project
from core.runtime_paths import resolve_runtime_paths
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


class TestInvestigateCrashWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls._host_root = Path(cls._tmp.name) / "host_project"
        cls._host_root.mkdir(parents=True)
        (cls._host_root / "src").mkdir()
        (cls._host_root / "src" / "main.cpp").write_text("int main() { return 0; }", encoding="utf-8")
        (cls._host_root / "docs").mkdir()
        (cls._host_root / "docs" / "README.md").write_text("# Project", encoding="utf-8")
        (cls._host_root / "Saved" / "Crashes").mkdir(parents=True)
        (cls._host_root / "Saved" / "Logs").mkdir(parents=True)
        (cls._host_root / "Saved" / "Logs" / "crash.log").write_text(
            "LogWindows: Error: begin: stack for UAT\nLogCore: Error: appError called: Fatal error\n",
            encoding="utf-8",
        )

        cls._agent_root = Path(__file__).resolve().parent.parent
        initialize_host_project(cls._agent_root, cls._host_root)

        paths = resolve_runtime_paths(cls._agent_root, host_root=cls._host_root)
        config = load_agentswarm_config(paths)
        manifest = load_project_manifest(paths)

        cls._registry = load_workflows(
            project_root=cls._agent_root,
            workflows_root=paths.built_in_workflows_root,
            llm_manager=DisabledLLMManager(),
            runtime_paths=paths,
            config=config,
            manifest=manifest,
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _get_workflow(self):
        return self._registry.get("investigate-crash-workflow")

    # -- Workflow loading --

    def test_workflow_loads_and_registers_reviewer(self):
        runtime = self._get_workflow()
        self.assertIsNotNone(runtime)
        self.assertIsNotNone(runtime.graph)

    # -- Fallback without LLM --

    def test_fallback_without_llm(self):
        runtime = self._get_workflow()
        run_dir = Path(self._tmp.name) / "run_fallback"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = runtime.invoke({
            "task_prompt": "Game crashes with access violation when loading level 3",
            "task_id": "crash-test-fallback",
            "run_dir": str(run_dir),
        })

        investigation_doc = result.get("investigation_doc", "")
        self.assertIn("## Crash Context", investigation_doc)
        self.assertIn("## Call Stack Analysis", investigation_doc)
        self.assertIn("## Root Cause Hypothesis", investigation_doc)
        self.assertIn("## Domain Classification", investigation_doc)
        self.assertIn("## Fix Recommendations", investigation_doc)
        self.assertIn("## Verification Plan", investigation_doc)

    # -- Report generation --

    def test_generate_report_fallback_without_llm(self):
        """When LLM is unavailable, generate_report produces fallback report."""
        runtime = self._get_workflow()
        run_dir = Path(self._tmp.name) / "run_report"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = runtime.invoke({
            "task_prompt": "TDR timeout crash with 50 enemies",
            "task_id": "crash-test-report",
            "run_dir": str(run_dir),
        })

        crash_report = result.get("crash_report", "")
        # Fallback report should have structured sections
        self.assertIn("Crash Analysis Report", crash_report)
        self.assertIn("Root Cause", crash_report)
        self.assertIn("Domain Classification", crash_report)

    # -- Save findings --

    def test_save_findings_skipped_on_rejection(self):
        """When not approved, no memory file should be written."""
        runtime = self._get_workflow()
        run_dir = Path(self._tmp.name) / "run_no_save"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = runtime.invoke({
            "task_prompt": "Random crash investigation",
            "task_id": "crash-test-nosave",
            "run_dir": str(run_dir),
        })

        # Without LLM, review never approves, so no memory should be written
        memory_path = self._host_root.parent / ".agentswarm" / "memory" / "crash_analysis"
        if memory_path.exists():
            md_files = list(memory_path.glob("*.md"))
            self.assertEqual(len(md_files), 0, "No memory files should be written when not approved")

    # -- Final report structure --

    def test_final_report_includes_domain_and_report(self):
        runtime = self._get_workflow()
        run_dir = Path(self._tmp.name) / "run_structure"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = runtime.invoke({
            "task_prompt": "Access violation in combo graph",
            "task_id": "crash-test-structure",
            "run_dir": str(run_dir),
        })

        final_report = result.get("final_report", {})
        self.assertIn("domain_classification", final_report)
        self.assertIn("crash_report", final_report)

    # -- Routing --

    def test_routing_matches_crash_prompts(self):
        crash_prompts = [
            "investigate crash in combo system",
            "debug access violation on PS5",
            "investigate TDR timeout crash with 50 enemies",
            "analyze crash dump from QA",
            "investigate stability issue causing freeze",
        ]
        for prompt in crash_prompts:
            match = self._registry.route(prompt)
            self.assertIsNotNone(
                match,
                f"Expected a workflow match for: '{prompt}'",
            )

    def test_routing_does_not_steal_other_prompts(self):
        other_prompts = [
            "fix gameplay bug in dodge mechanic",
            "optimize game thread tick functions",
            "optimize rendering draw calls",
        ]
        for prompt in other_prompts:
            match = self._registry.route(prompt)
            if match is not None:
                self.assertNotEqual(
                    match.name,
                    "investigate-crash-workflow",
                    f"Crash workflow should not match: '{prompt}'",
                )

    # -- Extract section helper (tested in TestExtractSection below) --


class TestExtractSection(unittest.TestCase):
    """Isolated tests for _extract_section that don't need workflow loading."""

    def test_extract_existing_section(self):
        # Import directly from the module
        import importlib.util
        import sys
        module_path = Path(__file__).resolve().parent.parent / "Workflows" / "StabilityWorkflows" / "investigate-crash-workflow" / "entry.py"
        spec = importlib.util.spec_from_file_location("crash_entry", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["crash_entry"] = module
        spec.loader.exec_module(module)

        doc = (
            "## Root Cause Hypothesis\n"
            "Null pointer in TickComponent at line 42.\n\n"
            "## Domain Classification\n"
            "engine — threading violation.\n"
        )
        self.assertIn("Null pointer", module._extract_section(doc, "Root Cause Hypothesis"))
        self.assertIn("engine", module._extract_section(doc, "Domain Classification"))
        self.assertEqual(module._extract_section(doc, "Missing"), "Not found.")

    def test_fallback_investigation_doc_has_all_sections(self):
        import importlib.util
        import sys
        module_path = Path(__file__).resolve().parent.parent / "Workflows" / "StabilityWorkflows" / "investigate-crash-workflow" / "entry.py"
        spec = importlib.util.spec_from_file_location("crash_entry2", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["crash_entry2"] = module
        spec.loader.exec_module(module)

        doc = module._fallback_investigation_doc(
            task_prompt="Game crashes on level load",
            investigation_round=1,
            project_snapshot="### Host Root Layout\n- src/",
            relevant_docs=["docs/README.md"],
            relevant_source=["src/main.cpp"],
            relevant_tests=[],
            previous_investigation="",
            review_feedback="",
            improvement_actions=[],
        )
        for section in module.INVESTIGATION_SECTIONS:
            self.assertIn(
                f"## {section}",
                doc,
                f"Fallback doc missing section: {section}",
            )


if __name__ == "__main__":
    unittest.main()
