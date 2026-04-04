from __future__ import annotations

import json
import subprocess

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata


def _run_gh(
    args: list[str],
    *,
    repo: str = "",
    timeout: int = 30,
) -> tuple[bool, str]:
    cmd = ["gh"] + args
    if repo:
        cmd.extend(["--repo", repo])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = (result.stdout if result.returncode == 0 else result.stderr).strip()
        return result.returncode == 0, output
    except FileNotFoundError:
        return False, "gh CLI not found. Install from https://cli.github.com"
    except subprocess.TimeoutExpired:
        return False, f"gh command timed out after {timeout}s"


def _error(action: str, message: str) -> tuple[str, dict[str, object]]:
    return f"GitHub error ({action}): {message}", {"error": message, "action": action}


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def github_issues(
        action: str,
        repo: str = "",
        issue_number: int = 0,
        body: str = "",
        labels: str = "",
        state: str = "open",
        assignee: str = "",
        limit: int = 20,
    ) -> tuple[str, dict[str, object]]:
        """Interact with GitHub issues.

        Args:
            action: Operation to perform. One of: list, get, comment, close, reopen, label.
            repo: Repository in owner/repo format. Empty string uses the current directory repo.
            issue_number: Issue number. Required for get, comment, close, reopen, label.
            body: Comment body text. Required for comment action.
            labels: Comma-separated label names. Used by label action (add) and list filter.
            state: Issue state filter for list action. One of: open, closed, all.
            assignee: Assignee filter for list action.
            limit: Maximum results for list action.
        """
        if action == "list":
            args = [
                "issue", "list",
                "--json", "number,title,state,labels,assignees,url",
                "--state", state,
                "--limit", str(limit),
            ]
            if labels:
                for label in labels.split(","):
                    label = label.strip()
                    if label:
                        args.extend(["--label", label])
            if assignee:
                args.extend(["--assignee", assignee])

            ok, output = _run_gh(args, repo=repo)
            if not ok:
                return _error("list", output)

            issues = json.loads(output) if output else []
            return (
                f"Found {len(issues)} issue(s) matching filters (state={state}).",
                {"issues": issues, "action": "list", "state": state},
            )

        if action == "get":
            if not issue_number:
                return _error("get", "issue_number is required")
            args = [
                "issue", "view", str(issue_number),
                "--json", "number,title,body,state,labels,assignees,comments,url",
            ]
            ok, output = _run_gh(args, repo=repo)
            if not ok:
                return _error("get", output)

            issue = json.loads(output)
            return (
                f"Issue #{issue_number}: {issue.get('title', '')} [{issue.get('state', '')}]",
                {"issue": issue, "action": "get"},
            )

        if action == "comment":
            if not issue_number:
                return _error("comment", "issue_number is required")
            if not body:
                return _error("comment", "body is required")
            args = ["issue", "comment", str(issue_number), "--body", body]
            ok, output = _run_gh(args, repo=repo)
            if not ok:
                return _error("comment", output)
            return (
                f"Comment added to issue #{issue_number}.",
                {"action": "comment", "issue_number": issue_number},
            )

        if action in ("close", "reopen"):
            if not issue_number:
                return _error(action, "issue_number is required")
            args = ["issue", action, str(issue_number)]
            ok, output = _run_gh(args, repo=repo)
            if not ok:
                return _error(action, output)
            return (
                f"Issue #{issue_number} {action}d.",
                {"action": action, "issue_number": issue_number},
            )

        if action == "label":
            if not issue_number:
                return _error("label", "issue_number is required")
            if not labels:
                return _error("label", "labels is required")
            label_list = [l.strip() for l in labels.split(",") if l.strip()]
            args = ["issue", "edit", str(issue_number)]
            for label in label_list:
                args.extend(["--add-label", label])
            ok, output = _run_gh(args, repo=repo)
            if not ok:
                return _error("label", output)
            return (
                f"Added label(s) {label_list} to issue #{issue_number}.",
                {"action": "label", "issue_number": issue_number, "labels": label_list},
            )

        return _error(action, f"Unknown action: {action}. Use: list, get, comment, close, reopen, label.")

    return github_issues
