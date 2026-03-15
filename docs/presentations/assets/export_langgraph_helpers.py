from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.main_graph import build_main_graph
from core.workflow_loader import load_workflows


class DisabledLLMClient:
    def is_enabled(self) -> bool:
        return False

    def describe(self) -> str:
        return "disabled export client"

    def generate_text(self, *, instructions: str, input_text: str, effort: str | None = None) -> str:
        raise RuntimeError("LLM should not be called while exporting diagrams")

    def generate_json(
        self,
        *,
        instructions: str,
        input_text: str,
        schema_name: str,
        schema: dict,
        effort: str | None = None,
    ) -> dict:
        raise RuntimeError("LLM should not be called while exporting diagrams")


class FakeLLMManager:
    def __init__(self) -> None:
        self._client = DisabledLLMClient()

    def resolve(self, profile: str | None = None) -> DisabledLLMClient:
        return self._client

    def describe(self, profile: str | None = None) -> str:
        return f"{profile or 'default'}: disabled export client"

    def available_profiles(self) -> list[str]:
        return ["default"]


def _load_registry():
    project_root = PROJECT_ROOT
    workflows_root = project_root / "Workflows"
    llm_manager = FakeLLMManager()
    registry = load_workflows(
        project_root=project_root,
        workflows_root=workflows_root,
        llm_manager=llm_manager,
    )
    return registry, llm_manager


def load_main_graph():
    registry, llm_manager = _load_registry()
    return build_main_graph(registry=registry, llm_manager=llm_manager).get_graph(xray=1)


def load_engineer_graph():
    registry, _ = _load_registry()
    return registry.get("gameplay-engineer-workflow").graph.get_graph(xray=1)
