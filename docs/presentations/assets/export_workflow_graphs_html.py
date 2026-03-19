from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path

from export_langgraph_helpers import load_engineer_graph, load_main_graph


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "presentations" / "workflow-graphs.html"


def _build_html(main_graph_mermaid: str, engineer_graph_mermaid: str) -> str:
    exported_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AgentSwarm Workflow Graph Export</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --panel: #fffaf2;
      --ink: #18222c;
      --muted: #5f6b75;
      --line: rgba(24, 34, 44, 0.12);
      --teal: #0f766e;
      --coral: #d86a48;
      --shadow: 0 22px 52px rgba(24, 34, 44, 0.12);
      --radius: 24px;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(216, 106, 72, 0.12), transparent 26%),
        linear-gradient(180deg, #fff9ef 0%, var(--bg) 100%);
      color: var(--ink);
    }}

    main {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 40px 24px 64px;
      display: grid;
      gap: 28px;
    }}

    .hero {{
      display: grid;
      gap: 14px;
    }}

    .eyebrow {{
      display: inline-flex;
      align-items: center;
      gap: 10px;
      width: fit-content;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid var(--line);
      font-size: 0.8rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--teal);
    }}

    .eyebrow::before {{
      content: "";
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--teal), var(--coral));
    }}

    h1 {{
      margin: 0;
      font-size: clamp(2.5rem, 5vw, 4.6rem);
      line-height: 0.96;
      letter-spacing: -0.04em;
    }}

    p {{
      margin: 0;
      max-width: 72ch;
      color: var(--muted);
      line-height: 1.6;
      font-size: 1.02rem;
    }}

    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 6px;
    }}

    .chip {{
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.82);
      border: 1px solid var(--line);
      font-size: 0.92rem;
      color: var(--ink);
      box-shadow: 0 8px 20px rgba(24, 34, 44, 0.05);
    }}

    .grid {{
      display: grid;
      gap: 24px;
    }}

    .panel {{
      background: rgba(255, 250, 242, 0.88);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .panel-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 20px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.58);
    }}

    .panel-header h2 {{
      margin: 0;
      font-size: clamp(1.4rem, 2vw, 2rem);
      letter-spacing: -0.03em;
    }}

    .panel-body {{
      padding: 12px;
    }}

    .mermaid-shell {{
      background: white;
      border-radius: 18px;
      padding: 18px;
      border: 1px solid rgba(24, 34, 44, 0.08);
      overflow: auto;
    }}

    .source {{
      padding: 0 22px 22px;
      display: grid;
      gap: 10px;
    }}

    details {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.62);
      overflow: hidden;
    }}

    summary {{
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 700;
      color: var(--ink);
    }}

    pre {{
      margin: 0;
      padding: 16px;
      overflow: auto;
      background: #fffdf8;
      border-top: 1px solid var(--line);
      color: #22303b;
      font-size: 0.9rem;
      line-height: 1.45;
    }}

    a {{
      color: var(--teal);
      text-decoration: none;
    }}

    a:hover {{
      text-decoration: underline;
    }}

    @media (max-width: 860px) {{
      main {{
        padding: 24px 14px 40px;
      }}
      .panel-header {{
        flex-direction: column;
        align-items: flex-start;
      }}
    }}
  </style>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{
      startOnLoad: true,
      theme: "base",
      securityLevel: "loose",
      flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: "linear" }},
      themeVariables: {{
        primaryColor: "#f4f0ff",
        primaryTextColor: "#18222c",
        primaryBorderColor: "#8d78f0",
        lineColor: "#406173",
        secondaryColor: "#eef7f5",
        tertiaryColor: "#fff6ef",
        fontFamily: "Segoe UI, Helvetica Neue, Arial, sans-serif",
      }},
    }});
  </script>
</head>
<body>
  <main>
    <section class="hero">
      <span class="eyebrow">HTML Export</span>
      <h1>AgentSwarm workflow graphs</h1>
      <p>This page is generated from the current LangGraph runtime structure. Open it in a browser to inspect the latest main graph and the latest <code>gameplay-engineer-workflow</code> graph.</p>
      <div class="meta">
        <div class="chip">Exported at: {escape(exported_at)}</div>
        <div class="chip">Source: <a href="assets/main-graph.mmd">main-graph.mmd</a></div>
        <div class="chip">Source: <a href="assets/gameplay-engineer-graph.mmd">gameplay-engineer-graph.mmd</a></div>
      </div>
    </section>

    <section class="grid">
      <article class="panel">
        <div class="panel-header">
          <h2>Main Graph</h2>
          <span class="chip">Control plane</span>
        </div>
        <div class="panel-body">
          <div class="mermaid-shell">
            <pre class="mermaid">{escape(main_graph_mermaid)}</pre>
          </div>
        </div>
        <div class="source">
          <details>
            <summary>Show Mermaid source</summary>
            <pre>{escape(main_graph_mermaid)}</pre>
          </details>
        </div>
      </article>

      <article class="panel">
        <div class="panel-header">
          <h2>Gameplay Engineer Workflow</h2>
          <span class="chip">Feature + bugfix split</span>
        </div>
        <div class="panel-body">
          <div class="mermaid-shell">
            <pre class="mermaid">{escape(engineer_graph_mermaid)}</pre>
          </div>
        </div>
        <div class="source">
          <details>
            <summary>Show Mermaid source</summary>
            <pre>{escape(engineer_graph_mermaid)}</pre>
          </details>
        </div>
      </article>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    main_graph_mermaid = load_main_graph().draw_mermaid()
    engineer_graph_mermaid = load_engineer_graph().draw_mermaid()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        _build_html(
            main_graph_mermaid=main_graph_mermaid,
            engineer_graph_mermaid=engineer_graph_mermaid,
        ),
        encoding="utf-8",
    )
    print(f"Exported HTML graph viewer to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
