# Workflow Authoring

## Rules

- Check `Workflows/Share/` for existing templates before creating new workflows
- `template-investigation-workflow` is the canonical investigation + review loop — use it as reference
- `Workflow.md` frontmatter is required; parsed by `core/front_matter.py`
- `entry.py` must define `build_graph(context: WorkflowContext, metadata: WorkflowMetadata)` returning a LangGraph `StateGraph`
- Use `trace_graph_node()` for node decoration and `trace_route_decision()` for edge logging
- Prompts must be natural language — do not serialize full state as JSON for LLM input
- Internal reviewer/support workflows should be `exposed: false`
- Wire reviewer subgraphs via `context.get_workflow_graph()`

## Placement

| Type | Directory |
|------|-----------|
| Reusable across domains | `Workflows/Share/` |
| Domain-specific | `Workflows/{Domain}Workflows/` |
| Support/reviewer (not user-routable) | Same directory, `exposed: false` |
