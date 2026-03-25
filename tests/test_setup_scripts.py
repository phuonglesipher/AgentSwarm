from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SetupScriptTests(unittest.TestCase):
    def _read(self, relative_path: str) -> str:
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_setup_bat_requires_python3(self) -> None:
        script = self._read("Setup.bat")
        self.assertIn('call :resolve_python3', script)
        self.assertIn('py -3 -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)"', script)
        self.assertNotIn("py -2", script)
        self.assertIn("Python 3 is required to set up AgentSwarm.", script)

    def test_agentswarm_bat_requires_python3(self) -> None:
        script = self._read("AgentSwarm.bat")
        self.assertIn('call :resolve_python3', script)
        self.assertIn('set "PYTHON_BIN=py -3"', script)
        self.assertNotIn("python main.py --prompt", script)
        self.assertIn("Python 3 is required to run AgentSwarm.", script)

    def test_setup_sh_requires_python3(self) -> None:
        script = self._read("Setup.sh")
        self.assertIn('PYTHON_BIN="${PYTHON:-python3}"', script)
        self.assertIn("command -v python3", script)
        self.assertIn("sys.version_info[0] == 3", script)
        self.assertIn("Python 3 is required to set up AgentSwarm.", script)

    def test_agentswarm_sh_requires_python3(self) -> None:
        script = self._read("AgentSwarm.sh")
        self.assertIn('PYTHON_BIN="${PYTHON:-python3}"', script)
        self.assertIn("sys.version_info[0] == 3", script)
        self.assertIn('exec "$PYTHON_BIN" "$SCRIPT_DIR/main.py" --prompt "$PROMPT"', script)
        self.assertIn("Python 3 is required to run AgentSwarm.", script)


if __name__ == "__main__":
    unittest.main()
