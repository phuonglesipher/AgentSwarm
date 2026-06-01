from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

from core.config_loader import AgentSwarmConfig, ProjectManifest
from core.llm import LLMManager
from core.runtime_paths import RuntimePaths
from core.text_utils import slugify
from core.workflow_loader import load_workflows


DEFAULT_REPO = "sipherxyz/s2"
DEFAULT_PROJECT_OWNER = "sipherxyz"
DEFAULT_PROJECT_NUMBER = 5
DEFAULT_LABEL = "AgentTask"
TODO_STATUSES = {"todo", "to do"}
IN_PROGRESS_STATUS = "In Progress"
DONE_STATUS = "Done"
BLOCKED_STATUS = "Blocked"
LOCAL_EXCLUDES = (".agentswarm/",)


@dataclass(frozen=True)
class AgentTaskRunnerConfig:
    repo: str = DEFAULT_REPO
    project_owner: str = DEFAULT_PROJECT_OWNER
    project_number: int = DEFAULT_PROJECT_NUMBER
    label: str = DEFAULT_LABEL
    once: bool = True
    poll: bool = False
    interval_seconds: int = 60
    dry_run: bool = False
    codex_profile: str = "ai-gateway"
    codex_model: str = "gpt-5.5"
    reasoning_effort: str = "xhigh"
    codex_sandbox: str = "danger-full-access"
    codex_timeout_seconds: int = 1800
    max_tasks: int = 0


@dataclass(frozen=True)
class ProjectStatusInfo:
    project_id: str
    status_field_id: str
    options: dict[str, str]


@dataclass(frozen=True)
class AgentTaskIssue:
    number: int
    title: str
    body: str
    url: str
    updated_at: str
    labels: tuple[str, ...]
    project_item_id: str
    project_status: str


@dataclass(frozen=True)
class ParsedAgentTask:
    base_branch: str
    new_branch: str
    execution_prompt: str


class RunnerError(RuntimeError):
    pass


class LocalPreflightError(RunnerError):
    pass


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _run(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = 120,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise RunnerError(f"Required command not found on PATH: {name}")


def _compact_output(completed: subprocess.CompletedProcess[str], limit: int = 4000) -> str:
    parts = []
    if completed.stdout.strip():
        parts.append(completed.stdout.strip())
    if completed.stderr.strip():
        parts.append(completed.stderr.strip())
    text = "\n".join(parts).strip()
    if len(text) > limit:
        text = text[:limit] + "\n[... truncated ...]"
    return text


def _is_runtime_overlay_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip().strip('"')
    return normalized == ".agentswarm" or normalized.startswith(".agentswarm/")


def ensure_local_git_excludes(host_root: Path) -> None:
    completed = _run(["git", "rev-parse", "--git-dir"], cwd=host_root, timeout=30)
    if completed.returncode != 0:
        raise RunnerError(f"Unable to locate git directory:\n{_compact_output(completed)}")

    raw_git_dir = completed.stdout.strip()
    if not raw_git_dir:
        raise RunnerError("Unable to locate git directory.")

    git_dir = Path(raw_git_dir)
    if not git_dir.is_absolute():
        git_dir = host_root / git_dir
    exclude_path = git_dir / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    additions = [pattern for pattern in LOCAL_EXCLUDES if pattern not in existing.splitlines()]
    if additions:
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        exclude_path.write_text(existing + prefix + "\n".join(additions) + "\n", encoding="utf-8")


def _gh_json(args: list[str], *, cwd: Path, timeout: int = 120, input_text: str | None = None) -> Any:
    completed = _run(["gh", *args], cwd=cwd, timeout=timeout, input_text=input_text)
    if completed.returncode != 0:
        raise RunnerError(_compact_output(completed) or f"gh command failed: {' '.join(args)}")
    output = completed.stdout.strip()
    return json.loads(output) if output else None


def _gh_plain(args: list[str], *, cwd: Path, timeout: int = 120, input_text: str | None = None) -> str:
    completed = _run(["gh", *args], cwd=cwd, timeout=timeout, input_text=input_text)
    if completed.returncode != 0:
        raise RunnerError(_compact_output(completed) or f"gh command failed: {' '.join(args)}")
    return completed.stdout.strip()


def validate_environment(host_root: Path, config: AgentTaskRunnerConfig) -> None:
    for command in ("gh", "git", "python", "codex"):
        _require_command(command)

    auth = _run(["gh", "auth", "status"], cwd=host_root, timeout=30)
    auth_text = _compact_output(auth)
    if auth.returncode != 0:
        raise RunnerError(f"gh auth is not ready:\n{auth_text}")
    if "project" not in auth_text:
        raise RunnerError("GitHub token is missing the project scope. Run: gh auth refresh -s project")

    if config.dry_run:
        return

    if config.codex_profile.strip().lower() == "ai-gateway" and not os.getenv("AI_GATEWAY_API_KEY", "").strip():
        raise RunnerError(
            "AI_GATEWAY_API_KEY is required for Codex profile `ai-gateway`. "
            "Set it in the current shell, `D:\\s2\\.env.local`, or `D:\\s2\\AgentSwarm\\.env.local`, then rerun."
        )

    codex_command = shutil.which("codex") or "codex"
    codex_check = _run(
        [
            codex_command,
            "--profile",
            config.codex_profile,
            "--model",
            config.codex_model,
            "--config",
            f'model_reasoning_effort="{config.reasoning_effort}"',
            "--sandbox",
            "read-only",
            "--ask-for-approval",
            "never",
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--color",
            "never",
            "Reply exactly: OK",
        ],
        cwd=host_root,
        timeout=180,
    )
    if codex_check.returncode != 0 or "OK" not in (codex_check.stdout + codex_check.stderr):
        output = _compact_output(codex_check)
        if "AI_GATEWAY_API_KEY" in output:
            raise RunnerError(
                "Codex ai-gateway self-check failed because AI_GATEWAY_API_KEY is missing. "
                "Set it in the current shell, `D:\\s2\\.env.local`, or `D:\\s2\\AgentSwarm\\.env.local`, then rerun."
            )
        if "refresh_token_reused" in output or "Please log out and sign in again" in output:
            raise RunnerError(
                "Codex auth refresh failed. Run `codex logout` then `codex login`, or provide a valid "
                "AI_GATEWAY_API_KEY for the `ai-gateway` profile."
            )
        raise RunnerError(f"Codex ai-gateway self-check failed:\n{output}")


def get_project_status_info(host_root: Path, config: AgentTaskRunnerConfig) -> ProjectStatusInfo:
    query = """
    query($owner: String!, $number: Int!) {
      organization(login: $owner) {
        projectV2(number: $number) {
          id
          fields(first: 50) {
            nodes {
              ... on ProjectV2SingleSelectField {
                id
                name
                options {
                  id
                  name
                }
              }
            }
          }
        }
      }
    }
    """
    data = _gh_json(
        [
            "api",
            "graphql",
            "-f",
            f"owner={config.project_owner}",
            "-F",
            f"number={config.project_number}",
            "-f",
            f"query={query}",
        ],
        cwd=host_root,
    )
    project = data.get("data", {}).get("organization", {}).get("projectV2")
    if not project:
        raise RunnerError(f"Project not found: {config.project_owner}/{config.project_number}")
    for field in project.get("fields", {}).get("nodes", []):
        if field and field.get("name") == "Status":
            options = {
                str(option["name"]): str(option["id"])
                for option in field.get("options", [])
                if option.get("name") and option.get("id")
            }
            return ProjectStatusInfo(
                project_id=str(project["id"]),
                status_field_id=str(field["id"]),
                options=options,
            )
    raise RunnerError("Project Status field was not found.")


def _issue_project_item(issue: dict[str, Any], config: AgentTaskRunnerConfig) -> tuple[str, str] | None:
    raw_items = issue.get("projectItems", []) or []
    if isinstance(raw_items, dict):
        raw_items = raw_items.get("nodes", []) or []
    for item in raw_items:
        project = item.get("project") or {}
        if str(project.get("number")) != str(config.project_number):
            continue
        owner_login = str(project.get("owner", {}).get("login", "")).lower()
        if owner_login and owner_login != config.project_owner.lower():
            continue
        status = ""
        raw_values = item.get("fieldValues", []) or []
        if isinstance(raw_values, dict):
            raw_values = raw_values.get("nodes", []) or []
        for field_value in raw_values:
            if field_value.get("field", {}).get("name") == "Status":
                status = str(field_value.get("name") or "")
                break
        return str(item.get("id") or ""), status
    return None


def list_todo_agent_tasks(host_root: Path, config: AgentTaskRunnerConfig) -> list[AgentTaskIssue]:
    repo_owner, repo_name = config.repo.split("/", 1)
    query = """
    query($owner: String!, $name: String!, $label: String!) {
      repository(owner: $owner, name: $name) {
        issues(first: 100, states: OPEN, labels: [$label], orderBy: {field: UPDATED_AT, direction: ASC}) {
          nodes {
            number
            title
            body
            url
            updatedAt
            labels(first: 30) {
              nodes { name }
            }
            projectItems(first: 20) {
              nodes {
                id
                project {
                  number
                  owner {
                    ... on Organization { login }
                    ... on User { login }
                  }
                }
                fieldValues(first: 30) {
                  nodes {
                    ... on ProjectV2ItemFieldSingleSelectValue {
                      name
                      field {
                        ... on ProjectV2SingleSelectField { name }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    data = _gh_json(
        [
            "api",
            "graphql",
            "-f",
            f"owner={repo_owner}",
            "-f",
            f"name={repo_name}",
            "-f",
            f"label={config.label}",
            "-f",
            f"query={query}",
        ],
        cwd=host_root,
        timeout=120,
    )
    issues = data.get("data", {}).get("repository", {}).get("issues", {}).get("nodes", [])
    tasks: list[AgentTaskIssue] = []
    for issue in issues or []:
        item = _issue_project_item(issue, config)
        if item is None:
            continue
        item_id, status = item
        if not item_id or status.strip().lower() not in TODO_STATUSES:
            continue
        tasks.append(
            AgentTaskIssue(
                number=int(issue["number"]),
                title=str(issue.get("title") or ""),
                body=str(issue.get("body") or ""),
                url=str(issue.get("url") or ""),
                updated_at=str(issue.get("updatedAt") or ""),
                labels=tuple(
                    str(label.get("name") or "")
                    for label in (
                        issue.get("labels", {}).get("nodes", [])
                        if isinstance(issue.get("labels"), dict)
                        else issue.get("labels", [])
                    )
                ),
                project_item_id=item_id,
                project_status=status,
            )
        )
    return sorted(tasks, key=lambda task: (task.updated_at, task.number))


def update_project_status(
    host_root: Path,
    info: ProjectStatusInfo,
    issue: AgentTaskIssue,
    status_name: str,
) -> None:
    option_id = info.options.get(status_name)
    if option_id is None:
        raise RunnerError(f"Project status option not found: {status_name}")
    mutation = """
    mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
      updateProjectV2ItemFieldValue(input: {
        projectId: $projectId,
        itemId: $itemId,
        fieldId: $fieldId,
        value: { singleSelectOptionId: $optionId }
      }) {
        projectV2Item { id }
      }
    }
    """
    _gh_json(
        [
            "api",
            "graphql",
            "-f",
            f"projectId={info.project_id}",
            "-f",
            f"itemId={issue.project_item_id}",
            "-f",
            f"fieldId={info.status_field_id}",
            "-f",
            f"optionId={option_id}",
            "-f",
            f"query={mutation}",
        ],
        cwd=host_root,
    )


def comment_issue(host_root: Path, config: AgentTaskRunnerConfig, issue_number: int, body: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] issue #{issue_number} comment:\n{body}\n")
        return
    _gh_plain(
        ["issue", "comment", str(issue_number), "--repo", config.repo, "--body", body],
        cwd=host_root,
        timeout=120,
    )


def parse_agent_task(issue: AgentTaskIssue) -> ParsedAgentTask:
    body = issue.body.replace("\r\n", "\n")

    def find_heading(names: tuple[str, ...]) -> str:
        escaped = "|".join(re.escape(name) for name in names)
        pattern = re.compile(
            rf"^##+\s*(?:{escaped})\s*\n(?P<value>.*?)(?=^##+\s+|\Z)",
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        match = pattern.search(body)
        return match.group("value").strip() if match else ""

    base_branch = find_heading(("Base Branch", "Base branch", "Git Base Branch"))
    new_branch = find_heading(("New Branch", "Target Branch", "Branch"))
    prompt = find_heading(("Execution Prompt", "Prompt", "Task Description"))
    if not base_branch:
        raise RunnerError("AgentTask issue is missing required section: Base Branch")
    if not prompt:
        raise RunnerError("AgentTask issue is missing required section: Execution Prompt")
    base_branch = base_branch.splitlines()[0].strip().strip("`")
    new_branch = new_branch.splitlines()[0].strip().strip("`") if new_branch else ""
    if not new_branch:
        new_branch = f"agenttask/{issue.number}-{slugify(issue.title, fallback='task')}"
        if len(new_branch) > 80:
            new_branch = new_branch[:80].rstrip("-")
    return ParsedAgentTask(base_branch=base_branch, new_branch=new_branch, execution_prompt=prompt)


def ensure_clean_worktree(host_root: Path) -> None:
    completed = _run(["git", "status", "--porcelain"], cwd=host_root, timeout=30)
    if completed.returncode != 0:
        raise RunnerError(f"Unable to inspect git status:\n{_compact_output(completed)}")
    dirty_lines = []
    for line in completed.stdout.splitlines():
        path = line[3:].strip() if len(line) > 3 else line.strip()
        if _is_runtime_overlay_path(path):
            continue
        dirty_lines.append(line)
    if dirty_lines:
        raise LocalPreflightError("Git worktree is not clean. Commit, stash, or move local changes before running AgentTask.")


def run_git(args: list[str], host_root: Path) -> str:
    completed = _run(["git", *args], cwd=host_root, timeout=300)
    if completed.returncode != 0:
        raise RunnerError(f"git {' '.join(args)} failed:\n{_compact_output(completed)}")
    return _compact_output(completed)


def current_branch(host_root: Path) -> str:
    completed = _run(["git", "branch", "--show-current"], cwd=host_root, timeout=30)
    if completed.returncode != 0:
        raise RunnerError(f"Unable to inspect current git branch:\n{_compact_output(completed)}")
    branch = completed.stdout.strip()
    if not branch:
        raise RunnerError("Git HEAD is detached; AgentTask requires a named branch.")
    return branch


def git_ref_exists(host_root: Path, ref: str) -> bool:
    completed = _run(["git", "rev-parse", "--verify", "--quiet", ref], cwd=host_root, timeout=30)
    return completed.returncode == 0


def validate_branch_plan(host_root: Path, parsed: ParsedAgentTask) -> None:
    if parsed.new_branch == parsed.base_branch:
        return
    if git_ref_exists(host_root, f"refs/heads/{parsed.base_branch}"):
        return
    if git_ref_exists(host_root, f"refs/remotes/origin/{parsed.base_branch}"):
        return
    raise LocalPreflightError(
        f"Base branch `{parsed.base_branch}` does not exist locally or on origin. "
        "Fix the AgentTask branch fields before running."
    )


def build_issue_prompt(issue: AgentTaskIssue, parsed: ParsedAgentTask, effective_branch: str | None = None) -> str:
    working_branch = effective_branch or parsed.new_branch
    return "\n".join(
        [
            f"# AgentTask Issue #{issue.number}: {issue.title}",
            "",
            f"Issue URL: {issue.url}",
            f"Base Branch: {parsed.base_branch}",
            f"Working Branch: {working_branch}",
            f"Labels: {', '.join(issue.labels)}",
            "",
            "## Execution Prompt",
            parsed.execution_prompt,
            "",
            "## Full Issue Body",
            issue.body,
        ]
    )


def invoke_agent_processing(
    *,
    prompt: str,
    issue: AgentTaskIssue,
    parsed: ParsedAgentTask,
    runtime_paths: RuntimePaths,
    config: AgentSwarmConfig,
    manifest: ProjectManifest,
    llm_manager: LLMManager,
    run_dir: Path,
    runner_config: AgentTaskRunnerConfig,
) -> dict[str, Any]:
    registry = load_workflows(
        project_root=runtime_paths.agent_root,
        workflows_root=runtime_paths.built_in_workflows_root,
        llm_manager=llm_manager,
        runtime_paths=runtime_paths,
        config=config,
        manifest=manifest,
    )
    payload = {
        "prompt": prompt,
        "original_prompt": prompt,
        "task_prompt": prompt,
        "task_id": f"agent-task-{issue.number}",
        "run_dir": str(run_dir),
        "host_root": str(runtime_paths.host_root),
        "codex_profile": runner_config.codex_profile,
        "codex_model": runner_config.codex_model,
        "reasoning_effort": runner_config.reasoning_effort,
        "codex_sandbox": runner_config.codex_sandbox,
        "codex_timeout_seconds": runner_config.codex_timeout_seconds,
        "base_branch": parsed.base_branch,
        "new_branch": parsed.new_branch,
    }
    workflow_result = registry.invoke("agent-processing-workflow", payload)
    status = "failed" if workflow_result.get("active_task_error") else "completed"
    return {
        "final_response": workflow_result.get("summary", ""),
        "tasks": [
            {
                "id": f"agent-task-{issue.number}",
                "status": status,
                "workflow_name": "agentswarm::agent-processing-workflow",
                "output": {
                    "summary": workflow_result.get("summary", ""),
                    "final_report": workflow_result.get("final_report", {}),
                },
                "error": workflow_result.get("active_task_error"),
            }
        ],
        "workflow_result": workflow_result,
    }


def changed_files(host_root: Path) -> list[str]:
    completed = _run(["git", "status", "--porcelain"], cwd=host_root, timeout=30)
    if completed.returncode != 0:
        raise RunnerError(f"Unable to inspect git changes:\n{_compact_output(completed)}")
    files: list[str] = []
    for line in completed.stdout.splitlines():
        if len(line) > 3:
            path = line[3:].strip()
            if not _is_runtime_overlay_path(path):
                files.append(path)
    return files


def create_commit_push_pr(
    host_root: Path,
    config: AgentTaskRunnerConfig,
    issue: AgentTaskIssue,
    parsed: ParsedAgentTask,
    run_dir: Path,
    head_branch: str | None = None,
    create_pr: bool = True,
) -> str:
    effective_head = head_branch or parsed.new_branch
    files = changed_files(host_root)
    if not files:
        raise RunnerError("Codex completed but produced no git changes.")
    run_git(["add", "--all"], host_root)
    commit_title = f"[AgentTask #{issue.number}] {issue.title}"
    run_git(["commit", "-m", commit_title], host_root)
    run_git(["push", "-u", "origin", effective_head], host_root)
    if not create_pr:
        return f"Pushed current branch `{effective_head}`. PR creation skipped because the task requested working on the current branch."
    body = "\n".join(
        [
            f"Closes #{issue.number}",
            "",
            "Created by AgentTask runner.",
            f"Artifacts: `{run_dir}`",
            "",
            "Changed files:",
            *[f"- `{path}`" for path in files],
        ]
    )
    return _gh_plain(
        [
            "pr",
            "create",
            "--repo",
            config.repo,
            "--base",
            parsed.base_branch,
            "--head",
            effective_head,
            "--title",
            commit_title,
            "--body",
            body,
        ],
        cwd=host_root,
        timeout=180,
    )


def process_issue(
    *,
    issue: AgentTaskIssue,
    status_info: ProjectStatusInfo,
    runtime_paths: RuntimePaths,
    config: AgentSwarmConfig,
    manifest: ProjectManifest,
    llm_manager: LLMManager,
    runner_config: AgentTaskRunnerConfig,
    run_dir: Path,
) -> None:
    host_root = runtime_paths.host_root
    parsed = parse_agent_task(issue)
    effective_branch = parsed.new_branch
    should_create_pr = True

    if not runner_config.dry_run:
        ensure_clean_worktree(host_root)
        run_git(["fetch", "origin"], host_root)
        validate_branch_plan(host_root, parsed)

    comment_issue(
        host_root,
        runner_config,
        issue.number,
        f"AgentTask claimed at {_now_stamp()}.\n\nBase branch: `{parsed.base_branch}`\nWorking branch: `{parsed.new_branch}`",
        runner_config.dry_run,
    )
    if not runner_config.dry_run:
        update_project_status(host_root, status_info, issue, IN_PROGRESS_STATUS)

    comment_issue(host_root, runner_config, issue.number, f"Git setup started at {_now_stamp()}.", runner_config.dry_run)
    if not runner_config.dry_run:
        active_branch = current_branch(host_root)

        if parsed.new_branch == parsed.base_branch:
            effective_branch = active_branch
            should_create_pr = False
            comment_issue(
                host_root,
                runner_config,
                issue.number,
                (
                    f"Base branch and working branch both resolve to `{parsed.base_branch}`; "
                    f"continuing on current branch `{active_branch}`."
                ),
                False,
            )
            run_git(["pull", "--ff-only", "origin", active_branch], host_root)
        else:
            if active_branch == parsed.base_branch:
                comment_issue(
                    host_root,
                    runner_config,
                    issue.number,
                    f"Already on base branch `{parsed.base_branch}`; skipping switch and pulling latest.",
                    False,
                )
            else:
                comment_issue(
                    host_root,
                    runner_config,
                    issue.number,
                    f"Switching from `{active_branch}` to base branch `{parsed.base_branch}`.",
                    False,
                )
                run_git(["switch", parsed.base_branch], host_root)
            run_git(["pull", "--ff-only", "origin", parsed.base_branch], host_root)
            comment_issue(
                host_root,
                runner_config,
                issue.number,
                f"Creating working branch `{parsed.new_branch}`.",
                False,
            )
            run_git(["switch", "-c", parsed.new_branch], host_root)

    prompt = build_issue_prompt(issue, parsed, effective_branch=effective_branch)
    comment_issue(
        host_root,
        runner_config,
        issue.number,
        f"Codex processing started at {_now_stamp()} with `{runner_config.codex_model}` / `{runner_config.reasoning_effort}`.",
        runner_config.dry_run,
    )
    result = (
        {"final_response": "[dry-run] Codex execution skipped.", "tasks": []}
        if runner_config.dry_run
        else invoke_agent_processing(
            prompt=prompt,
            issue=issue,
            parsed=parsed,
            runtime_paths=runtime_paths,
            config=config,
            manifest=manifest,
            llm_manager=llm_manager,
            run_dir=run_dir,
            runner_config=runner_config,
        )
    )
    final_response = str(result.get("final_response", "")).strip()
    if not runner_config.dry_run and any(task.get("status") == "failed" for task in result.get("tasks", [])):
        raise RunnerError(final_response or "Agent processing failed.")

    if runner_config.dry_run:
        comment_issue(host_root, runner_config, issue.number, "[dry-run] Would commit, push, create PR, and set Done.", True)
        return

    pr_url = create_commit_push_pr(
        host_root,
        runner_config,
        issue,
        parsed,
        run_dir,
        head_branch=effective_branch,
        create_pr=should_create_pr,
    )
    comment_issue(
        host_root,
        runner_config,
        issue.number,
        "\n".join(
            [
                f"AgentTask completed at {_now_stamp()}.",
                "",
                f"Delivery: {pr_url}",
                f"Artifacts: `{run_dir}`",
                "",
                final_response[:4000],
            ]
        ),
        False,
    )
    update_project_status(host_root, status_info, issue, DONE_STATUS)


def run_agent_task_loop(
    *,
    runtime_paths: RuntimePaths,
    config: AgentSwarmConfig,
    manifest: ProjectManifest,
    llm_manager: LLMManager,
    runner_config: AgentTaskRunnerConfig,
    run_dir_factory,
) -> int:
    validate_environment(runtime_paths.host_root, runner_config)
    ensure_local_git_excludes(runtime_paths.host_root)
    status_info = get_project_status_info(runtime_paths.host_root, runner_config)
    processed = 0
    while True:
        tasks = list_todo_agent_tasks(runtime_paths.host_root, runner_config)
        if runner_config.max_tasks > 0:
            tasks = tasks[: runner_config.max_tasks]
        if not tasks:
            print(f"No {runner_config.label} issues with Project Status Todo.")
        for issue in tasks:
            run_dir = run_dir_factory()
            try:
                process_issue(
                    issue=issue,
                    status_info=status_info,
                    runtime_paths=runtime_paths,
                    config=config,
                    manifest=manifest,
                    llm_manager=llm_manager,
                    runner_config=runner_config,
                    run_dir=run_dir,
                )
                processed += 1
            except Exception as exc:
                if isinstance(exc, LocalPreflightError):
                    print(
                        f"AgentTask local preflight blocked at {_now_stamp()}.\n\n"
                        f"Error:\n```\n{str(exc)[:4000]}\n```"
                    )
                    continue
                message = f"AgentTask blocked at {_now_stamp()}.\n\nError:\n```\n{str(exc)[:4000]}\n```"
                print(message)
                try:
                    comment_issue(runtime_paths.host_root, runner_config, issue.number, message, runner_config.dry_run)
                    if not runner_config.dry_run and BLOCKED_STATUS in status_info.options:
                        update_project_status(runtime_paths.host_root, status_info, issue, BLOCKED_STATUS)
                except Exception as comment_exc:
                    print(f"Failed to report issue error: {comment_exc}")
        if runner_config.once or not runner_config.poll:
            return processed
        time.sleep(max(1, runner_config.interval_seconds))
