# AgentSwarm

General-purpose LangGraph workflow orchestration framework for Unreal Engine projects. Python 3.11+.

Deployable to any UE project as a host — not tied to a specific game.

## See Also

- `AGENTS.md` — project structure, build/test commands, coding style, commit guidelines

## Architecture

Main graph (`core/main_graph.py`) decomposes prompts → tasks → workflow dispatch → results.

Three engine abstractions drive all workflow logic:

| Engine | Files | Role |
|--------|-------|------|
| ReviewEngine | `core/review/engine.py`, `profile.py` | LLM-scored review with criteria, hard blockers, process filtering |
| PlanEngine | `core/plan/engine.py`, `profile.py` | Strategy-driven plan generation with fallback templates |
| Scoring | `core/scoring/engine.py` | ScorePolicy + MAD confidence, normalizes rubric to 0-100 |

Each engine is configured by a **frozen Profile dataclass** (ReviewProfile, PlanProfile). Never subclass engines; configure via profiles.

## Quality Loop Philosophy

Core design principle: **loop to increment quality**.

- `evaluate_quality_loop()` in `core/quality_loop.py` is the decision gate
- `QualityLoopSpec` controls: threshold (default 90), min_rounds (2), max_rounds, stagnation_limit
- A round cannot pass before min_rounds even if score is perfect
- Stagnation detection stops futile loops (score delta < min_score_delta for N rounds)
- When writing new workflow nodes: wire the quality loop, do not hand-code pass/fail logic

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
- `template-investigation-workflow` is the canonical loop pattern

## LLM Abstraction

- `core/llm.py`: LLMClient ABC → CodexCliLLMClient, ClaudeCodeLLMClient, ResponsesLLMClient
- `core/executor.py`: ClaudeCodeExecutorClient — multi-turn subprocess with tool access
- Single-turn clients for analysis/review; executor for implementation
- Anti-recursion guard in executor prevents workflow pipeline re-invocation

## Host Project Integration

AgentSwarm reads host project config from `agentswarm.yaml` (source_roots, doc_roots, test_roots). Workflows operate ON the host project; AgentSwarm provides the orchestration. Tools in `Tools/` are host-project-aware (e.g., find-gameplay-code searches host source_roots).
