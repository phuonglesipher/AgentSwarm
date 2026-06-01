# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the entry point for the AgentSwarm runtime and AgentTask queue mode.
- `AgentSwarm.bat` is the Windows launcher that forwards a prompt to `main.py`.
- `AgentSwarm.sh` is the macOS/Linux launcher that forwards a prompt to `main.py`.
- `core/` contains shared runtime code: workflow loading, graph orchestration, LLM abstraction, models, and routing helpers.
- `Workflows/AgentWorkflows/agent-processing-workflow/` contains the single active workflow. It delegates code changes to Codex through the ai-gateway profile.
- `.codex/skill/` contains repo-local Codex skills; use `workflow-creator` when designing or refactoring workflow architecture around reuse, subgraphs, and score-gated loops.
- `docs/` stores supporting gameplay and design references used by workflows.
- `tests/` contains repository-level unit tests.
- `runs/` is generated output from executions; treat it as disposable runtime artifact data.

## Build, Test, and Development Commands
- `python3 main.py --prompt "Fix combat dodge cancel bug..."` runs the single workflow locally.
- `powershell -ExecutionPolicy Bypass -File .\Run-AgentTask.ps1 -Once` processes one AgentTask scan from GitHub Projects.
- `powershell -ExecutionPolicy Bypass -File .\Run-AgentTask.ps1 -Poll -IntervalSeconds 60` runs the AgentTask polling loop.
- `AgentSwarm.bat Fix combat dodge cancel bug...` runs the same flow from Windows.
- `./AgentSwarm.sh Fix combat dodge cancel bug...` runs the same flow from macOS/Linux.
- `python3 -m unittest discover -s tests -v` runs the full test suite.
- `python3 -m compileall main.py core Workflows tests` performs a fast recursive syntax check.
- `codex login` is required before Codex CLI-backed LLM flows can run successfully.
- Default Codex task processing uses `codex --profile ai-gateway --model gpt-5.5` with reasoning effort `xhigh`.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and explicit type hints where practical.
- Keep module filenames in `snake_case.py`.
- Keep workflow folders in lowercase kebab-case, for example `agent-processing-workflow/`.
- Each workflow must expose `build_graph(...)` from `entry.py`.
- Prefer small, focused helper functions in `core/` and workflow modules. Keep comments brief and only where logic is non-obvious.

## Workflow Architecture Defaults
- Do not add domain-specific workflows by default. Keep AgentSwarm as a queue runner plus one `agent-processing-workflow`.
- Put queue/status/git/PR behavior in `core/agent_task_runner.py`.
- Put Codex invocation behavior in `core/codex_runner.py` or `agent-processing-workflow`.
- Keep prompts in natural language and pass the GitHub issue body through as the task prompt.

## Testing Guidelines
- Tests use the standard library `unittest` runner.
- Name test files `test_*.py` and test functions `test_*`.
- Cover deterministic fallback behavior first; LLM-backed flows should degrade safely when Codex, Claude Code, or API auth is unavailable.
- When fixing workflow runtime behavior, add or update a regression test in `tests/test_runtime.py`.

## Commit & Pull Request Guidelines
- Follow the existing history style: short, imperative commit messages such as `Fix Codex self-test harness for generated gameplay code`.
- Keep commits scoped to one logical change.
- PRs should include: a short summary, the prompt used for validation when relevant, test results, and any notable artifact path under `runs/`.

## Security & Configuration Tips
- Start from `.env.example` for local configuration.
- Do not commit secrets, auth tokens, or generated files from `runs/`.
- If you add a new workflow-specific model profile, wire it through `core/llm.py` and document the env vars in `.env.example`.
