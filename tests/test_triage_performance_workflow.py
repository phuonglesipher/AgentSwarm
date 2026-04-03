from __future__ import annotations

import unittest
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys


def _load_entry():
    """Load entry.py as a module for testing."""
    entry_path = Path(__file__).resolve().parent.parent / (
        "Workflows/OptimizationWorkflows/triage-performance-workflow/entry.py"
    )
    spec = spec_from_file_location("triage_entry", entry_path)
    mod = module_from_spec(spec)
    sys.modules["triage_entry"] = mod
    spec.loader.exec_module(mod)
    return mod


entry = _load_entry()


class TestClassifyFromThreadBreakdown(unittest.TestCase):
    """Test deterministic classification from optick thread data."""

    def test_gamethread_dominant(self):
        raw = (
            '{"per_thread_scopes": {'
            '"GameThread": [{"name": "Tick", "total_ms": 120, "avg_ms": 4.0, "calls": 30}], '
            '"RenderThread": [{"name": "Draw", "total_ms": 20, "avg_ms": 0.7, "calls": 30}]'
            "}}"
        )
        result = entry._classify_from_thread_breakdown(raw)
        self.assertEqual(result, ["gamethread"])

    def test_rendering_dominant(self):
        raw = (
            '{"per_thread_scopes": {'
            '"RenderThread": [{"name": "Draw", "total_ms": 150, "avg_ms": 5.0, "calls": 30}], '
            '"RHIThread": [{"name": "Submit", "total_ms": 80, "avg_ms": 2.7, "calls": 30}], '
            '"GameThread": [{"name": "Tick", "total_ms": 30, "avg_ms": 1.0, "calls": 30}]'
            "}}"
        )
        result = entry._classify_from_thread_breakdown(raw)
        self.assertIn("rendering", result)
        # GameThread at 30ms vs rendering total 230ms — below 30% threshold
        self.assertNotIn("gamethread", result)

    def test_streaming_dominant(self):
        raw = (
            '{"per_thread_scopes": {'
            '"AsyncLoadingThread": [{"name": "AsyncLoad", "total_ms": 200, "avg_ms": 6.7, "calls": 30}], '
            '"GameThread": [{"name": "Tick", "total_ms": 40, "avg_ms": 1.3, "calls": 30}]'
            "}}"
        )
        result = entry._classify_from_thread_breakdown(raw)
        self.assertIn("streaming", result)

    def test_multiple_bottlenecks(self):
        raw = (
            '{"per_thread_scopes": {'
            '"GameThread": [{"name": "Tick", "total_ms": 100, "avg_ms": 3.3, "calls": 30}], '
            '"RenderThread": [{"name": "Draw", "total_ms": 90, "avg_ms": 3.0, "calls": 30}]'
            "}}"
        )
        result = entry._classify_from_thread_breakdown(raw)
        self.assertIn("gamethread", result)
        self.assertIn("rendering", result)

    def test_empty_input(self):
        result = entry._classify_from_thread_breakdown("")
        self.assertEqual(result, [])

    def test_no_json(self):
        result = entry._classify_from_thread_breakdown("some random text without json")
        self.assertEqual(result, [])


class TestLLMClassifyPrompt(unittest.TestCase):
    """Test keyword-based fallback classification."""

    def test_gamethread_keywords(self):
        result = entry._llm_classify_prompt("game thread tick is slow with 50 AI")
        self.assertIn("gamethread", result)

    def test_rendering_keywords(self):
        result = entry._llm_classify_prompt("too many draw calls and GPU is maxed")
        self.assertIn("rendering", result)

    def test_streaming_keywords(self):
        result = entry._llm_classify_prompt("hitches when loading new areas, world partition streaming")
        self.assertIn("streaming", result)

    def test_generic_prompt_no_match(self):
        result = entry._llm_classify_prompt("FPS drops in level X")
        self.assertEqual(result, [])

    def test_multiple_domains(self):
        result = entry._llm_classify_prompt("game thread tick is slow and too many draw calls from render thread")
        self.assertIn("gamethread", result)
        self.assertIn("rendering", result)


class TestRoutingHints(unittest.TestCase):
    """Test that new routing domain hints are present in registry."""

    def test_new_hints_present(self):
        from core.registry import _ROUTING_DOMAIN_HINTS

        for hint in ("fps", "frame", "framerate", "profiling", "bottleneck", "hitch", "spike", "capture"):
            self.assertIn(hint, _ROUTING_DOMAIN_HINTS, f"Missing routing hint: {hint}")


if __name__ == "__main__":
    unittest.main()
