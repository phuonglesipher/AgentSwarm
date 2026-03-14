#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import importlib.util
from pathlib import Path
import sys
import textwrap
import webbrowser
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a LangGraph graph from a helper module to Mermaid and standalone HTML."
    )
    parser.add_argument("--helper", required=True, help="Path to a helper Python file exposing load_graph().")
    parser.add_argument(
        "--callable",
        default="load_graph",
        dest="callable_name",
        help="Callable name inside the helper module. Defaults to load_graph.",
    )
    parser.add_argument("--output-html", required=True, help="Output HTML file path.")
    parser.add_argument("--output-mermaid", help="Optional Mermaid .mmd output path.")
    parser.add_argument("--title", help="Optional HTML title. Defaults to the HTML filename stem.")
    parser.add_argument(
        "--theme",
        default="neutral",
        choices=["default", "neutral", "dark", "forest", "base"],
        help="Mermaid theme for the HTML viewer.",
    )
    parser.add_argument("--open", action="store_true", help="Open the generated HTML in the default browser.")
    return parser.parse_args()


def _load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path.stem] = module
    spec.loader.exec_module(module)
    return module


def _resolve_mermaid(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate

    if hasattr(candidate, "draw_mermaid") and callable(candidate.draw_mermaid):
        return candidate.draw_mermaid()

    if hasattr(candidate, "get_graph") and callable(candidate.get_graph):
        return _resolve_mermaid(candidate.get_graph())

    if hasattr(candidate, "compile") and callable(candidate.compile):
        return _resolve_mermaid(candidate.compile())

    raise TypeError(
        "Unsupported graph object. Return a StateGraph, CompiledStateGraph, Graph, or Mermaid string."
    )


def _build_html(mermaid_text: str, title: str, theme: str) -> str:
    escaped_title = html.escape(title)
    escaped_mermaid = html.escape(mermaid_text)
    return textwrap.dedent(
        f"""\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>{escaped_title}</title>
          <style>
            :root {{
              --bg: #f6f5ef;
              --fg: #1f1f1f;
              --panel: #ffffff;
              --border: #d9d4c7;
            }}
            * {{ box-sizing: border-box; }}
            body {{
              margin: 0;
              font-family: Georgia, "Times New Roman", serif;
              background: radial-gradient(circle at top, #fff8db 0%, var(--bg) 55%);
              color: var(--fg);
            }}
            main {{
              max-width: 1400px;
              margin: 0 auto;
              padding: 24px;
            }}
            .panel {{
              background: var(--panel);
              border: 1px solid var(--border);
              border-radius: 16px;
              padding: 16px;
              box-shadow: 0 12px 30px rgba(24, 24, 24, 0.08);
            }}
            h1 {{
              margin: 0 0 12px 0;
              font-size: 2rem;
            }}
            p {{
              margin: 0 0 16px 0;
            }}
            #diagram {{
              overflow: auto;
            }}
          </style>
        </head>
        <body>
          <main>
            <div class="panel">
              <h1>{escaped_title}</h1>
              <p>Generated from LangGraph Mermaid output.</p>
              <textarea id="mermaid-source" hidden>{escaped_mermaid}</textarea>
              <pre class="mermaid" id="diagram"></pre>
            </div>
          </main>
          <script type="module">
            import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
            mermaid.initialize({{ startOnLoad: false, theme: "{theme}" }});
            const source = document.getElementById("mermaid-source").value;
            const target = document.getElementById("diagram");
            target.textContent = source;
            await mermaid.run({{ nodes: [target] }});
          </script>
        </body>
        </html>
        """
    )


def main() -> None:
    args = _parse_args()
    helper_path = Path(args.helper).resolve()
    if not helper_path.exists():
        raise SystemExit(f"Helper file does not exist: {helper_path}")

    module = _load_module(helper_path)
    if not hasattr(module, args.callable_name):
        raise SystemExit(f"Helper module does not define {args.callable_name}()")

    factory = getattr(module, args.callable_name)
    if not callable(factory):
        raise SystemExit(f"{args.callable_name} is not callable")

    graph_candidate = factory()
    mermaid_text = _resolve_mermaid(graph_candidate)

    output_html = Path(args.output_html).resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or output_html.stem.replace("-", " ").title()
    output_html.write_text(_build_html(mermaid_text, title, args.theme), encoding="utf-8")

    if args.output_mermaid:
        output_mermaid = Path(args.output_mermaid).resolve()
        output_mermaid.parent.mkdir(parents=True, exist_ok=True)
        output_mermaid.write_text(mermaid_text, encoding="utf-8")

    print(f"HTML: {output_html}")
    if args.output_mermaid:
        print(f"Mermaid: {Path(args.output_mermaid).resolve()}")

    if args.open:
        webbrowser.open(output_html.as_uri())


if __name__ == "__main__":
    main()
