# AgentSwarm

AgentTask-oriented LangGraph/Codex orchestration framework for Unreal Engine projects. Python 3.11+.

Deployable to any UE project as a host — not tied to a specific game.

## See Also

- `AGENTS.md` — project structure, build/test commands, coding style, commit guidelines

## Architecture

Main graph (`core/main_graph.py`) now routes prompts to a single minimal workflow: `agent-processing-workflow`.

Four engine abstractions drive all workflow logic:

| Engine | Files | Role |
|--------|-------|------|
| ReviewEngine | `core/review/engine.py`, `profile.py` | LLM-scored review with criteria, hard blockers, process filtering |
| PlanEngine | `core/plan/engine.py`, `profile.py` | Strategy-driven plan generation with fallback templates |
| Scoring | `core/scoring/engine.py` | ScorePolicy + MAD confidence, normalizes rubric to 0-100 |
| DecisionEngine | `core/decision/engine.py`, `profile.py` | LLM-powered graph routing with branch descriptions, fallback to default |

Legacy planning/review engines remain as support code for now, but new task execution should go through the AgentTask runner plus `agent-processing-workflow`.

## AgentTask Philosophy

Core design principle: keep orchestration outside Codex and keep implementation inside Codex.

- `core/agent_task_runner.py` owns GitHub queue polling, Project status, issue comments, git branch setup, commit/push, and PR creation.
- `agent-processing-workflow` owns a single Codex execution against the prepared prompt.
- Codex must not update GitHub status, create commits, push branches, or open PRs; the runner handles those lifecycle steps.
- Default Codex execution uses the `ai-gateway` profile, model `gpt-5.5`, and reasoning effort `xhigh`.

## Data Model Conventions

- Profile/Spec/Policy/Assessment types are `@dataclass(frozen=True)`
- Lightweight criterion types use `NamedTuple` (ReviewCriterion, PlanCriterion, HardBlocker)
- State flows through `TypedDict` (MainState, InvestigationLoopState), not Pydantic
- Immutable sequences use `tuple`, not `list`, in frozen dataclasses
- Scores are always normalized to 0-100 integers

## Workflow & Tool Authoring

- `Workflow.md` frontmatter: name, entry, version, exposed, capabilities
- `Tool.md` frontmatter: name, entry, version, output_mode, state_keys_shared, capabilities
- `entry.py` must expose `build_graph(context, metadata)`
- Shared reusable → `Workflows/Share/`; domain-specific → `Workflows/{Domain}Workflows/`
- Reviewer subgraphs: `exposed: false`, wire via `context.get_workflow_graph()`
- `agent-processing-workflow` is the canonical workflow. Add runner behavior in `core/agent_task_runner.py` instead of creating new domain workflows.

## LLM Abstraction

- `core/llm.py`: LLMClient ABC → CodexCliLLMClient, ClaudeCodeLLMClient, ResponsesLLMClient
- `core/executor.py`: ClaudeCodeExecutorClient — multi-turn subprocess with tool access
- Single-turn clients for analysis/review; executor for implementation
- Anti-recursion guard in executor prevents workflow pipeline re-invocation

## Host Project Integration

AgentSwarm reads host project config from `agentswarm.yaml` (source_roots, doc_roots, test_roots). Workflows operate ON the host project; AgentSwarm provides the orchestration. Tools in `Tools/` are host-project-aware (e.g., find-gameplay-code searches host source_roots).
