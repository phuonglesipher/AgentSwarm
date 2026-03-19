# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the entry point for the AgentSwarm workflow-driven LangGraph runtime.
- `AgentSwarm.bat` is the Windows launcher that forwards a prompt to `main.py`.
- `AgentSwarm.sh` is the macOS/Linux launcher that forwards a prompt to `main.py`.
- `core/` contains shared runtime code: workflow loading, graph orchestration, LLM abstraction, models, and routing helpers.
- `Workflows/<name>/` contains one workflow per folder, with a required `Workflow.md` metadata file and `entry.py` graph implementation.
- `.codex/skill/` contains repo-local Codex skills; use `workflow-creator` when designing or refactoring workflow architecture around reuse, subgraphs, and score-gated loops.
- `docs/` stores supporting gameplay and design references used by workflows.
- `tests/` contains repository-level unit tests.
- `runs/` is generated output from executions; treat it as disposable runtime artifact data.

## Build, Test, and Development Commands
- `python3 main.py --prompt "Fix combat dodge cancel bug..."` runs the main graph locally.
- `AgentSwarm.bat Fix combat dodge cancel bug...` runs the same flow from Windows.
- `./AgentSwarm.sh Fix combat dodge cancel bug...` runs the same flow from macOS/Linux.
- `python3 -m unittest discover -s tests -v` runs the full test suite.
- `python3 -m py_compile main.py core/*.py Workflows/*/entry.py tests/test_runtime.py tests/test_short_term_memory_demo.py` performs a fast syntax check.
- `codex login` is required before Codex CLI-backed LLM flows can run successfully.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and explicit type hints where practical.
- Keep module filenames in `snake_case.py`.
- Keep workflow folders in lowercase kebab-case, for example `gameplay-engineer-workflow/`.
- Each workflow must expose `build_graph(...)` from `entry.py`.
- Prefer small, focused helper functions in `core/` and workflow modules. Keep comments brief and only where logic is non-obvious.

## Workflow Architecture Defaults
- When adding or refactoring a workflow, inspect `Workflows/*/Workflow.md` first and reuse an existing workflow or internal subgraph before creating a new one.
- Prefer a parent workflow plus internal subgraphs when a capability is reusable or when the reviewer logic should evolve independently. Wire reusable child graphs with `context.get_workflow_graph(...)`.
- Keep internal reviewer or support workflows `exposed: false` unless they are intended for direct user routing.
- For non-trivial investigation, planning, design, or implementation-prep flows, default to a strict loop: work state -> reviewer workflow -> score and approval gate -> iterate until the work reaches the threshold or the loop stops.
- Default review guardrails to at least 2 review rounds and a passing score of `>= 90`, with reviewer criteria centered on focus, evidence and ownership, architecture, clean code, optimization, and verification quality.
- Keep prompts in natural language and pass only the specific artifacts or feedback needed from the previous state. Do not serialize the whole state into JSON for prompt handoff unless structured machine output is truly required.

## Testing Guidelines
- Tests use the standard library `unittest` runner.
- Name test files `test_*.py` and test functions `test_*`.
- Cover deterministic fallback behavior first; LLM-backed flows should degrade safely when Codex or API auth is unavailable.
- When fixing workflow runtime behavior, add or update a regression test in `tests/test_runtime.py`.

## Commit & Pull Request Guidelines
- Follow the existing history style: short, imperative commit messages such as `Fix Codex self-test harness for generated gameplay code`.
- Keep commits scoped to one logical change.
- PRs should include: a short summary, the prompt used for validation when relevant, test results, and any notable artifact path under `runs/`.

## Security & Configuration Tips
- Start from `.env.example` for local configuration.
- Do not commit secrets, auth tokens, or generated files from `runs/`.
- If you add a new workflow-specific model profile, wire it through `core/llm.py` and document the env vars in `.env.example`.
