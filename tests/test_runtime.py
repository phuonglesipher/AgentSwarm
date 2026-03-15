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
                "find-gameplay-docs",
                "load-markdown-context",
            ],
        )
        qualified_tool_names = [item.qualified_name for item in self.tool_registry.list_metadata()]
        self.assertEqual(
            qualified_tool_names,
            [
                "agentswarm::find-gameplay-docs",
                "agentswarm::load-markdown-context",
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

    def test_main_graph_runs_end_to_end_without_llm(self) -> None:
        graph = build_main_graph(registry=self.registry, llm_manager=self.llm_manager)
        with tempfile.TemporaryDirectory(prefix="langgraph-tests-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            result = graph.invoke(
                build_initial_state(
                    prompt="Fix combat dodge cancel bug in melee gameplay and keep 3C responsiveness stable",
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
                / "task-1-fix-combat-dodge-cancel-bug-in-melee-gameplay"
                / "gameplay-engineer-workflow"
            )
            self.assertTrue((artifact_dir / "plan_doc.md").exists())
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

    def test_engineer_workflow_uses_tool_messages_for_doc_context(self) -> None:
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
