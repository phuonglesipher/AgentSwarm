from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.tool_engine import ToolEngine, ToolEngineConfig


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _make_context():
    ctx = MagicMock()
    ctx.resolve_scope_root.return_value = Path(".")
    return ctx


def _make_metadata(name: str):
    meta = MagicMock()
    meta.qualified_name = f"agentswarm::{name}"
    return meta


def _build_issues_tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gi", "Tools/github-issues/entry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_tool(_make_context(), _make_metadata("github-issues")), mod


def _build_pr_tool():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gp", "Tools/github-pr/entry.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_tool(_make_context(), _make_metadata("github-pr")), mod


def _mock_subprocess_ok(stdout: str):
    return MagicMock(returncode=0, stdout=stdout, stderr="")


def _mock_subprocess_fail(stderr: str):
    return MagicMock(returncode=1, stdout="", stderr=stderr)


# ------------------------------------------------------------------ #
#  github-issues tests
# ------------------------------------------------------------------ #


class TestGitHubIssuesTool(unittest.TestCase):
    """Test github-issues tool via tool.invoke() which returns a string."""

    def setUp(self):
        self.tool, self.mod = _build_issues_tool()

    def _invoke(self, args, mock_result=None, side_effect=None):
        """Invoke tool with mocked subprocess, return result string."""
        with patch.object(self.mod, "subprocess") as mock_sp:
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            if side_effect:
                mock_sp.run.side_effect = side_effect
            else:
                mock_sp.run.return_value = mock_result or _mock_subprocess_ok("")
            result = self.tool.invoke(args)
            return result, mock_sp

    def test_list_issues(self):
        issues = [
            {"number": 1, "title": "Bug A", "state": "OPEN", "labels": [], "assignees": [], "url": "..."},
            {"number": 2, "title": "Bug B", "state": "OPEN", "labels": [], "assignees": [], "url": "..."},
        ]
        msg, _ = self._invoke({"action": "list", "limit": 10}, _mock_subprocess_ok(json.dumps(issues)))
        self.assertIn("2 issue(s)", msg)

    def test_list_with_filters(self):
        _, mock_sp = self._invoke(
            {"action": "list", "state": "closed", "labels": "bug,critical", "assignee": "john"},
            _mock_subprocess_ok("[]"),
        )
        args = mock_sp.run.call_args[0][0]
        self.assertIn("--state", args)
        self.assertIn("closed", args)
        self.assertIn("--label", args)
        self.assertIn("bug", args)
        self.assertIn("critical", args)
        self.assertIn("--assignee", args)
        self.assertIn("john", args)

    def test_get_issue(self):
        issue = {"number": 42, "title": "Fix X", "body": "...", "state": "OPEN",
                 "labels": [], "assignees": [], "comments": [], "url": "..."}
        msg, _ = self._invoke({"action": "get", "issue_number": 42}, _mock_subprocess_ok(json.dumps(issue)))
        self.assertIn("#42", msg)
        self.assertIn("Fix X", msg)

    def test_get_requires_issue_number(self):
        msg, _ = self._invoke({"action": "get", "issue_number": 0})
        self.assertIn("issue_number is required", msg)

    def test_comment(self):
        msg, mock_sp = self._invoke(
            {"action": "comment", "issue_number": 10, "body": "Fixed in PR #5"},
            _mock_subprocess_ok(""),
        )
        self.assertIn("Comment added", msg)
        args = mock_sp.run.call_args[0][0]
        self.assertIn("--body", args)
        self.assertIn("Fixed in PR #5", args)

    def test_comment_requires_body(self):
        msg, _ = self._invoke({"action": "comment", "issue_number": 1, "body": ""})
        self.assertIn("body is required", msg)

    def test_close_issue(self):
        msg, mock_sp = self._invoke({"action": "close", "issue_number": 7}, _mock_subprocess_ok(""))
        self.assertIn("closed", msg.lower())
        args = mock_sp.run.call_args[0][0]
        self.assertEqual(args[1:3], ["issue", "close"])

    def test_reopen_issue(self):
        msg, _ = self._invoke({"action": "reopen", "issue_number": 7}, _mock_subprocess_ok(""))
        self.assertIn("reopen", msg.lower())

    def test_label_issue(self):
        msg, mock_sp = self._invoke(
            {"action": "label", "issue_number": 3, "labels": "bug,priority-high"},
            _mock_subprocess_ok(""),
        )
        self.assertIn("label", msg.lower())
        args = mock_sp.run.call_args[0][0]
        self.assertIn("--add-label", args)
        self.assertIn("bug", args)
        self.assertIn("priority-high", args)

    def test_label_requires_labels(self):
        msg, _ = self._invoke({"action": "label", "issue_number": 1, "labels": ""})
        self.assertIn("labels is required", msg)

    def test_unknown_action(self):
        msg, _ = self._invoke({"action": "delete"})
        self.assertIn("Unknown action", msg)

    def test_gh_not_found(self):
        msg, _ = self._invoke(
            {"action": "list"},
            side_effect=FileNotFoundError("gh not found"),
        )
        self.assertIn("not found", msg.lower())

    def test_gh_cli_error(self):
        msg, _ = self._invoke({"action": "list"}, _mock_subprocess_fail("HTTP 401: Bad credentials"))
        self.assertIn("401", msg)

    def test_repo_flag(self):
        _, mock_sp = self._invoke({"action": "list", "repo": "org/repo"}, _mock_subprocess_ok("[]"))
        args = mock_sp.run.call_args[0][0]
        self.assertIn("--repo", args)
        self.assertIn("org/repo", args)

    def test_timeout_handling(self):
        msg, _ = self._invoke(
            {"action": "list"},
            side_effect=subprocess.TimeoutExpired(cmd="gh", timeout=30),
        )
        self.assertIn("timed out", msg)


# ------------------------------------------------------------------ #
#  github-pr tests
# ------------------------------------------------------------------ #


class TestGitHubPRTool(unittest.TestCase):

    def setUp(self):
        self.tool, self.mod = _build_pr_tool()

    def _invoke(self, args, mock_result=None, side_effect=None):
        with patch.object(self.mod, "subprocess") as mock_sp:
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            if side_effect:
                mock_sp.run.side_effect = side_effect
            else:
                mock_sp.run.return_value = mock_result or _mock_subprocess_ok("")
            result = self.tool.invoke(args)
            return result, mock_sp

    def test_create_pr(self):
        msg, mock_sp = self._invoke(
            {"action": "create", "title": "Fix bug", "body": "Fixes #10", "base": "main", "draft": True},
            _mock_subprocess_ok("https://github.com/org/repo/pull/42"),
        )
        self.assertIn("created", msg.lower())
        self.assertIn("https://", msg)
        args = mock_sp.run.call_args[0][0]
        self.assertIn("--title", args)
        self.assertIn("Fix bug", args)
        self.assertIn("--body", args)
        self.assertIn("--base", args)
        self.assertIn("main", args)
        self.assertIn("--draft", args)

    def test_create_requires_title(self):
        msg, _ = self._invoke({"action": "create", "title": ""})
        self.assertIn("title is required", msg)

    def test_list_prs(self):
        prs = [{"number": 1, "title": "PR A", "state": "OPEN", "url": "...",
                 "headRefName": "feature", "baseRefName": "main"}]
        msg, _ = self._invoke({"action": "list"}, _mock_subprocess_ok(json.dumps(prs)))
        self.assertIn("1 PR(s)", msg)

    def test_get_pr(self):
        pr = {"number": 5, "title": "Add feature", "body": "...", "state": "OPEN",
              "url": "...", "headRefName": "feat", "baseRefName": "main",
              "reviews": [], "comments": []}
        msg, _ = self._invoke({"action": "get", "pr_number": 5}, _mock_subprocess_ok(json.dumps(pr)))
        self.assertIn("#5", msg)
        self.assertIn("Add feature", msg)

    def test_get_requires_pr_number(self):
        msg, _ = self._invoke({"action": "get", "pr_number": 0})
        self.assertIn("pr_number is required", msg)

    def test_unknown_action(self):
        msg, _ = self._invoke({"action": "merge"})
        self.assertIn("Unknown action", msg)

    def test_create_with_labels(self):
        _, mock_sp = self._invoke(
            {"action": "create", "title": "Fix", "labels": "bug,hotfix"},
            _mock_subprocess_ok("https://github.com/org/repo/pull/1"),
        )
        args = mock_sp.run.call_args[0][0]
        self.assertIn("--label", args)
        self.assertIn("bug", args)
        self.assertIn("hotfix", args)


# ------------------------------------------------------------------ #
#  ToolEngine integration
# ------------------------------------------------------------------ #


class MockLLM:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self._index = 0

    def is_enabled(self) -> bool:
        return True

    def describe(self) -> str:
        return "mock"

    def generate_json(self, **kwargs) -> dict:
        if self._index >= len(self._responses):
            return {"reasoning": "done", "done": True, "final_answer": "exhausted"}
        r = self._responses[self._index]
        self._index += 1
        return r


class TestToolEngineIntegration(unittest.TestCase):
    """Verify github tools work with ToolEngine orchestration."""

    def test_issues_tool_via_engine(self):
        tool, mod = _build_issues_tool()
        issues_data = [{"number": 1, "title": "Test", "state": "OPEN",
                        "labels": [], "assignees": [], "url": "..."}]

        llm = MockLLM([
            {
                "reasoning": "List open issues",
                "tool_calls": [{"tool_name": tool.name, "arguments": {"action": "list", "limit": 5}}],
                "done": False,
            },
            {
                "reasoning": "Done",
                "done": True,
                "final_answer": "Found 1 open issue: #1 Test",
            },
        ])

        engine = ToolEngine(
            config=ToolEngineConfig(system_id="test-github", max_turns=3),
            tools=[tool],
            llm=llm,
        )

        with patch.object(mod, "subprocess") as mock_sp:
            mock_sp.run.return_value = _mock_subprocess_ok(json.dumps(issues_data))
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            result = engine.gather(task="List open issues")

        self.assertTrue(result.success)
        self.assertEqual(result.turns_used, 2)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].tool_name, tool.name)
        self.assertIn("1 issue(s)", result.tool_calls[0].result)

    def test_pr_tool_via_engine(self):
        tool, mod = _build_pr_tool()

        llm = MockLLM([
            {
                "reasoning": "Create a PR",
                "tool_calls": [{"tool_name": tool.name, "arguments": {
                    "action": "create", "title": "Fix bug", "body": "Closes #1",
                }}],
                "done": False,
            },
            {
                "reasoning": "PR created",
                "done": True,
                "final_answer": "Created PR at https://github.com/org/repo/pull/42",
            },
        ])

        engine = ToolEngine(
            config=ToolEngineConfig(system_id="test-github-pr", max_turns=3),
            tools=[tool],
            llm=llm,
        )

        with patch.object(mod, "subprocess") as mock_sp:
            mock_sp.run.return_value = _mock_subprocess_ok("https://github.com/org/repo/pull/42")
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            result = engine.gather(task="Create a PR for bug fix")

        self.assertTrue(result.success)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertIn("created", result.tool_calls[0].result.lower())


if __name__ == "__main__":
    unittest.main()
