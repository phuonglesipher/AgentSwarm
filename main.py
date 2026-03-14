from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from core.blueprint_loader import load_blueprints
from core.llm import LLMManager
from core.main_graph import build_main_graph


def _build_run_dir(project_root: Path) -> Path:
    runs_dir = project_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blueprint-driven LangGraph agent runner")
    parser.add_argument("--prompt", default="", help="Prompt that should be executed by the agent")
    parser.add_argument("prompt_parts", nargs="*", help="Optional prompt parts when --prompt is omitted")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prompt = args.prompt.strip() or " ".join(args.prompt_parts).strip()
    if not prompt:
        raise SystemExit("A prompt is required. Use --prompt \"...\" or pass a positional prompt.")

    project_root = Path(__file__).resolve().parent
    blueprints_root = project_root / "Blueprints"
    run_dir = _build_run_dir(project_root)
    llm_manager = LLMManager.from_env()

    registry = load_blueprints(project_root=project_root, blueprints_root=blueprints_root, llm_manager=llm_manager)
    main_graph = build_main_graph(registry=registry, llm_manager=llm_manager)

    result = main_graph.invoke(
        {
            "prompt": prompt,
            "run_dir": str(run_dir),
            "tasks": [],
            "results": [],
            "final_response": "",
            "routing_notes": [],
        }
    )

    print(result["final_response"])
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
