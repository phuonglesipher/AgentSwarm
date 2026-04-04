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
    def github_pr(
        action: str,
        repo: str = "",
        pr_number: int = 0,
        title: str = "",
        body: str = "",
        base: str = "",
        head: str = "",
        draft: bool = False,
        labels: str = "",
        state: str = "open",
        limit: int = 10,
    ) -> tuple[str, dict[str, object]]:
        """Interact with GitHub pull requests.

        Args:
            action: Operation to perform. One of: create, list, get.
            repo: Repository in owner/repo format. Empty string uses the current directory repo.
            pr_number: PR number. Required for get action.
            title: PR title. Required for create action.
            body: PR description body. Used by create action.
            base: Base branch for create. Empty string uses repo default branch.
            head: Head branch for create. Empty string uses current branch.
            draft: Create PR as draft. Used by create action.
            labels: Comma-separated label names. Used by create action.
            state: PR state filter for list action. One of: open, closed, merged, all.
            limit: Maximum results for list action.
        """
        if action == "create":
            if not title:
                return _error("create", "title is required")
            args = ["pr", "create", "--title", title]
            if body:
                args.extend(["--body", body])
            else:
                args.extend(["--body", ""])
            if base:
                args.extend(["--base", base])
            if head:
                args.extend(["--head", head])
            if draft:
                args.append("--draft")
            if labels:
                for label in labels.split(","):
                    label = label.strip()
                    if label:
                        args.extend(["--label", label])

            ok, output = _run_gh(args, repo=repo, timeout=60)
            if not ok:
                return _error("create", output)
            return (
                f"Pull request created: {output}",
                {"action": "create", "url": output, "title": title},
            )

        if action == "list":
            args = [
                "pr", "list",
                "--json", "number,title,state,url,headRefName,baseRefName",
                "--state", state,
                "--limit", str(limit),
            ]
            ok, output = _run_gh(args, repo=repo)
            if not ok:
                return _error("list", output)

            prs = json.loads(output) if output else []
            return (
                f"Found {len(prs)} PR(s) matching filters (state={state}).",
                {"prs": prs, "action": "list", "state": state},
            )

        if action == "get":
            if not pr_number:
                return _error("get", "pr_number is required")
            args = [
                "pr", "view", str(pr_number),
                "--json", "number,title,body,state,url,headRefName,baseRefName,reviews,comments",
            ]
            ok, output = _run_gh(args, repo=repo)
            if not ok:
                return _error("get", output)

            pr = json.loads(output)
            return (
                f"PR #{pr_number}: {pr.get('title', '')} [{pr.get('state', '')}]",
                {"pr": pr, "action": "get"},
            )

        return _error(action, f"Unknown action: {action}. Use: create, list, get.")

    return github_pr
