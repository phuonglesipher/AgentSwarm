---
name: langgraph-mermaid-viewer
description: Export LangGraph graphs to Mermaid and standalone HTML viewers. Use when Codex needs to inspect, open, or share the structure of a `StateGraph`, `CompiledStateGraph`, or LangGraph `Graph`, especially for debugging routing, reviewing node and edge topology, or producing a browser-viewable graph snapshot from Python code.
---

# LangGraph Mermaid Viewer

## Overview
Visualize LangGraph topology without editing production graph code. Create a tiny helper loader, return the graph object you want to inspect, and run the bundled exporter to generate Mermaid text plus a standalone HTML viewer.

## Quick Start
1. Create a helper Python file that exposes `load_graph()`.
2. Return a `StateGraph`, compiled graph, LangGraph `Graph`, or Mermaid string.
3. Run `scripts/export_langgraph_mermaid.py`.
4. Open the generated HTML file.

Example:
```bash
python .codex/skill/langgraph-mermaid-viewer/scripts/export_langgraph_mermaid.py \
  --helper /tmp/load_graph.py \
  --output-html runs/graphs/main-graph.html \
  --output-mermaid runs/graphs/main-graph.mmd \
  --open
```

## Create the Helper Loader
Write a one-off helper file that returns the graph object:

```python
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

class State(TypedDict):
    text: str

def node_a(state: State):
    return {"text": state["text"] + "a"}

def load_graph():
    graph = StateGraph(State)
    graph.add_node("node_a", node_a)
    graph.add_edge(START, "node_a")
    return graph
```

If the target graph needs runtime dependencies, stub only the minimum required for construction. Do not change production code just to export a diagram.

## Work With This Repository
For this repo, `core/main_graph.py` and blueprint `entry.py` files often need small test doubles to construct graphs safely. When importing a blueprint entry from `Blueprints/<name>/entry.py`, prefer `importlib.util.spec_from_file_location(...)` because blueprint folder names use hyphens.

Save outputs under `runs/graphs/` unless the user asks for another location.

## Troubleshooting
- If graph construction fails, simplify the helper and replace runtime services with minimal stubs.
- If the helper already returns a compiled graph, keep it; the exporter handles compiled and uncompiled graphs.
- If labels contain HTML, let the exporter handle escaping. Do not pre-escape Mermaid text yourself.
