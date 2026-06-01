from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from langgraph.checkpoint.memory import InMemorySaver

from core.config_loader import load_agentswarm_config
from core.host_setup import initialize_host_project
from core.main_graph import build_initial_state, build_main_graph, build_runtime_config
from core.runtime_paths import resolve_runtime_paths
from core.tool_loader import load_tools
from core.workflow_loader import load_workflows


class DisabledLLMClient:
    def is_enabled(self) -> bool:
        return False

    def describe(self) -> str:
        return "disabled test client"

    def generate_text(self, **kwargs) -> str:
        raise AssertionError("LLM should not be called")

    def generate_json(self, **kwargs) -> dict:
        raise AssertionError("LLM should not be called")


class FakeLLMManager:
    def __init__(self) -> None:
        self._client = DisabledLLMClient()

    def resolve(self, profile: str | None = None) -> DisabledLLMClient:
        return self._client

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: disabled test client"

    def available_profiles(self) -> list[str]:
        return ["default"]


class AgentProcessingRuntimeTests(unittest.TestCase):
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

    def test_loads_single_agent_processing_workflow(self) -> None:
        names = [item.name for item in self.registry.list_metadata()]
        self.assertEqual(names, ["agent-processing-workflow"])
        exposed_names = [item.name for item in self.registry.list_metadata(exposed_only=True)]
        self.assertEqual(exposed_names, ["agent-processing-workflow"])
        self.assertEqual(load_agentswarm_config(resolve_runtime_paths(self.project_root)).active_workflows, ("agent-processing-workflow",))

    def test_main_graph_routes_to_agent_processing_workflow(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-agent-processing-") as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch(
                "workflow_agentswarm__agent_processing_workflow.run_codex_prompt"
            ) as run_codex:
                run_codex.return_value = SimpleNamespace(
                    success=True,
                    exit_code=0,
                    output_text="agent processing ok",
                    stdout="",
                    stderr="",
                    command=["codex"],
                    output_path=str(run_dir / "last.txt"),
                    stdout_path=str(run_dir / "stdout.txt"),
                    stderr_path=str(run_dir / "stderr.txt"),
                )
                graph = build_main_graph(
                    registry=self.registry,
                    llm_manager=self.llm_manager,
                    checkpointer=InMemorySaver(),
                )
                result = graph.invoke(
                    build_initial_state(
                        prompt="Implement a tiny AgentTask smoke change.",
                        run_dir=str(run_dir),
                    ),
                    build_runtime_config("agent-processing-smoke"),
                )

        self.assertIn("agent-processing-workflow", result["final_response"])
        self.assertIn("agent processing ok", result["final_response"])

    def test_main_passes_host_root_to_llm_manager_workdir(self) -> None:
        import main as main_module

        with tempfile.TemporaryDirectory(prefix="agentswarm-main-host-root-") as temp_dir:
            host_root = Path(temp_dir) / "host-project"
            host_root.mkdir(parents=True, exist_ok=True)
            runtime_paths = resolve_runtime_paths(self.project_root, host_root=host_root)
            run_dir = runtime_paths.runs_root / "test-run"
            run_dir.mkdir(parents=True, exist_ok=True)

            args = SimpleNamespace(
                prompt="Process this task.",
                prompt_parts=[],
                host_root=str(host_root),
                thread_id="",
                mode="workflow",
                tools="",
                once=False,
                poll=False,
                interval_seconds=60,
                dry_run=False,
                agent_task_repo="sipherxyz/s2",
                agent_task_label="AgentTask",
                project_owner="sipherxyz",
                project_number=5,
                codex_profile="ai-gateway",
                codex_model="gpt-5.5",
                reasoning_effort="xhigh",
                codex_sandbox="danger-full-access",
                codex_timeout_seconds=1800,
                max_tasks=0,
            )
            fake_config = SimpleNamespace(target_scope="host_project", active_workflows=("agent-processing-workflow",))
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


if __name__ == "__main__":
    unittest.main()
