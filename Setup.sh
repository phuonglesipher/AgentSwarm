#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install "langgraph>=1.1,<2" "langchain-core>=1.2,<2"
"$PYTHON_BIN" -m core.host_setup --agent-root "$SCRIPT_DIR"

echo "Installed LangGraph and LangChain Core tool support with $PYTHON_BIN"
