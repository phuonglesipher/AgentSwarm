from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.agent_task_runner import (
    AgentTaskIssue,
    AgentTaskRunnerConfig,
    LocalPreflightError,
    ProjectStatusInfo,
    RunnerError,
    build_issue_prompt,
    ensure_local_git_excludes,
    list_todo_agent_tasks,
    parse_agent_task,
    process_issue,
    run_agent_task_loop,
    update_project_status,
    validate_branch_plan,
    validate_environment,
)
from core.codex_runner import CodexRunConfig, build_codex_command


def _issue(body: str) -> AgentTaskIssue:
    return AgentTaskIssue(
        number=25537,
        title="task_s02_test_agent_task",
        body=body,
        url="https://github.com/sipherxyz/s2/issues/25537",
        updated_at="2026-06-01T00:00:00Z",
        labels=("task", "AgentTask"),
        project_item_id="PVTI_item",
        project_status="Todo",
    )


class AgentTaskParserTests(unittest.TestCase):
    def test_parse_agent_task_from_template_sections(self) -> None:
        parsed = parse_agent_task(
            _issue(
                "\n".join(
                    [
                        "## Base Branch",
                        "main",
                        "",
                        "## New Branch",
                        "agenttask/custom-branch",
                        "",
                        "## Execution Prompt",
                        "Fix the test task.",
                    ]
                )
            )
        )

        self.assertEqual(parsed.base_branch, "main")
        self.assertEqual(parsed.new_branch, "agenttask/custom-branch")
        self.assertEqual(parsed.execution_prompt, "Fix the test task.")

    def test_parse_agent_task_generates_branch_when_missing(self) -> None:
        parsed = parse_agent_task(
            _issue(
                "\n".join(
                    [
                        "## Base Branch",
                        "develop",
                        "",
                        "## Execution Prompt",
                        "Implement the issue.",
                    ]
                )
            )
        )

        self.assertEqual(parsed.base_branch, "develop")
        self.assertTrue(parsed.new_branch.startswith("agenttask/25537-task-s02-test-agent-task"))

    def test_parse_agent_task_requires_base_branch_and_prompt(self) -> None:
        with self.assertRaisesRegex(Exception, "Base Branch"):
            parse_agent_task(_issue("## Execution Prompt\nDo work."))
        with self.assertRaisesRegex(Exception, "Execution Prompt"):
            parse_agent_task(_issue("## Base Branch\nmain"))

    def test_build_issue_prompt_contains_issue_context(self) -> None:
        issue = _issue("## Base Branch\nmain\n\n## Execution Prompt\nDo work.")
        parsed = parse_agent_task(issue)
        prompt = build_issue_prompt(issue, parsed)

        self.assertIn("Issue #25537", prompt)
        self.assertIn("Base Branch: main", prompt)
        self.assertIn("Do work.", prompt)


class AgentTaskProjectTests(unittest.TestCase):
    def test_list_todo_agent_tasks_filters_project_status(self) -> None:
        payload = {
            "data": {
                "repository": {
                    "issues": {
                        "nodes": [
                            {
                                "number": 1,
                                "title": "todo task",
                                "body": "body",
                                "url": "url",
                                "updatedAt": "2026-01-01T00:00:00Z",
                                "labels": {"nodes": [{"name": "AgentTask"}]},
                                "projectItems": {
                                    "nodes": [
                                        {
                                            "id": "item-1",
                                            "project": {"number": 5, "owner": {"login": "sipherxyz"}},
                                            "fieldValues": {
                                                "nodes": [{"field": {"name": "Status"}, "name": "To Do"}]
                                            },
                                        }
                                    ]
                                },
                            },
                            {
                                "number": 2,
                                "title": "done task",
                                "body": "body",
                                "url": "url",
                                "updatedAt": "2026-01-02T00:00:00Z",
                                "labels": {"nodes": [{"name": "AgentTask"}]},
                                "projectItems": {
                                    "nodes": [
                                        {
                                            "id": "item-2",
                                            "project": {"number": 5, "owner": {"login": "sipherxyz"}},
                                            "fieldValues": {
                                                "nodes": [{"field": {"name": "Status"}, "name": "Done"}]
                                            },
                                        }
                                    ]
                                },
                            },
                        ]
                    }
                }
            }
        }
        with mock.patch("core.agent_task_runner._gh_json", return_value=payload):
            tasks = list_todo_agent_tasks(Path("."), AgentTaskRunnerConfig())

        self.assertEqual([task.number for task in tasks], [1])

    def test_update_project_status_uses_option_id(self) -> None:
        info = ProjectStatusInfo(
            project_id="project",
            status_field_id="field",
            options={"In Progress": "option-in-progress"},
        )
        issue = _issue("## Base Branch\nmain\n\n## Execution Prompt\nDo work.")
        calls: list[list[str]] = []

        def fake_gh_json(args, **kwargs):
            del kwargs
            calls.append(args)
            return {"data": {}}

        with mock.patch("core.agent_task_runner._gh_json", side_effect=fake_gh_json):
            update_project_status(Path("."), info, issue, "In Progress")

        flattened = " ".join(calls[0])
        self.assertIn("projectId=project", flattened)
        self.assertIn("itemId=PVTI_item", flattened)
        self.assertIn("optionId=option-in-progress", flattened)


class CodexRunnerTests(unittest.TestCase):
    def test_build_codex_command_defaults_to_gateway_55_xhigh(self) -> None:
        command = build_codex_command(CodexRunConfig(working_directory="D:/s2"), Path("out.txt"))
        text = " ".join(command)

        self.assertIn("--profile ai-gateway", text)
        self.assertIn("--model gpt-5.5", text)
        self.assertIn('model_reasoning_effort="xhigh"', text)
        self.assertIn("--sandbox danger-full-access", text)
        self.assertIn("--cd D:/s2", text)

    def test_validate_environment_uses_resolved_codex_command(self) -> None:
        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs):
            del kwargs
            calls.append(args)
            if args[:3] == ["gh", "auth", "status"]:
                return SimpleNamespace(returncode=0, stdout="Token scopes: 'repo', 'project'", stderr="")
            return SimpleNamespace(returncode=0, stdout="OK", stderr="")

        with mock.patch.dict("core.agent_task_runner.os.environ", {"AI_GATEWAY_API_KEY": "test-key"}):
            with mock.patch("core.agent_task_runner.shutil.which") as which:
                which.side_effect = lambda name: "C:/bin/codex.cmd" if name == "codex" else f"C:/bin/{name}.exe"
                with mock.patch("core.agent_task_runner._run", side_effect=fake_run):
                    validate_environment(Path("."), AgentTaskRunnerConfig())

        codex_call = [call for call in calls if call and call[0] == "C:/bin/codex.cmd"][0]
        self.assertEqual(codex_call[0], "C:/bin/codex.cmd")

    def test_validate_environment_reports_missing_ai_gateway_key_before_codex_self_check(self) -> None:
        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs):
            del kwargs
            calls.append(args)
            return SimpleNamespace(returncode=0, stdout="Token scopes: 'repo', 'project'", stderr="")

        with mock.patch.dict("core.agent_task_runner.os.environ", {}, clear=True):
            with mock.patch("core.agent_task_runner.shutil.which") as which:
                which.side_effect = lambda name: "C:/bin/codex.cmd" if name == "codex" else f"C:/bin/{name}.exe"
                with mock.patch("core.agent_task_runner._run", side_effect=fake_run):
                    with self.assertRaisesRegex(RunnerError, "AI_GATEWAY_API_KEY"):
                        validate_environment(Path("."), AgentTaskRunnerConfig())

        self.assertFalse(any(call and call[0] == "C:/bin/codex.cmd" for call in calls))

    def test_validate_environment_does_not_require_gateway_key_for_other_profiles(self) -> None:
        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs):
            del kwargs
            calls.append(args)
            if args[:3] == ["gh", "auth", "status"]:
                return SimpleNamespace(returncode=0, stdout="Token scopes: 'repo', 'project'", stderr="")
            return SimpleNamespace(returncode=0, stdout="OK", stderr="")

        with mock.patch.dict("core.agent_task_runner.os.environ", {}, clear=True):
            with mock.patch("core.agent_task_runner.shutil.which") as which:
                which.side_effect = lambda name: "C:/bin/codex.cmd" if name == "codex" else f"C:/bin/{name}.exe"
                with mock.patch("core.agent_task_runner._run", side_effect=fake_run):
                    validate_environment(Path("."), AgentTaskRunnerConfig(codex_profile="default"))

        self.assertTrue(any(call and call[0] == "C:/bin/codex.cmd" for call in calls))

    def test_validate_environment_summarizes_codex_auth_refresh_failure(self) -> None:
        def fake_run(args: list[str], **kwargs):
            del kwargs
            if args[:3] == ["gh", "auth", "status"]:
                return SimpleNamespace(returncode=0, stdout="Token scopes: 'repo', 'project'", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="refresh_token_reused Please log out and sign in again")

        with mock.patch.dict("core.agent_task_runner.os.environ", {"AI_GATEWAY_API_KEY": "test-key"}):
            with mock.patch("core.agent_task_runner.shutil.which") as which:
                which.side_effect = lambda name: "C:/bin/codex.cmd" if name == "codex" else f"C:/bin/{name}.exe"
                with mock.patch("core.agent_task_runner._run", side_effect=fake_run):
                    with self.assertRaisesRegex(RunnerError, "codex logout"):
                        validate_environment(Path("."), AgentTaskRunnerConfig())

    def test_ensure_local_git_excludes_adds_runtime_overlay(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agentswarm-git-exclude-") as temp_dir:
            root = Path(temp_dir)
            git_dir = root / ".git"
            (git_dir / "info").mkdir(parents=True)
            exclude_path = git_dir / "info" / "exclude"
            exclude_path.write_text("# local excludes\n", encoding="utf-8")

            with mock.patch(
                "core.agent_task_runner._run",
                return_value=SimpleNamespace(returncode=0, stdout=".git\n", stderr=""),
            ):
                ensure_local_git_excludes(root)

            self.assertIn(".agentswarm/", exclude_path.read_text(encoding="utf-8"))


class AgentTaskProcessTests(unittest.TestCase):
    def test_process_issue_skips_base_switch_when_already_on_base_branch(self) -> None:
        issue = _issue("## Base Branch\nmain\n\n## New Branch\nagenttask/test\n\n## Execution Prompt\nDo work.")
        status_info = ProjectStatusInfo(
            project_id="project",
            status_field_id="field",
            options={"In Progress": "in-progress", "Done": "done"},
        )
        runtime_paths = SimpleNamespace(host_root=Path("."))
        git_calls: list[list[str]] = []

        def fake_run_git(args: list[str], host_root: Path) -> str:
            del host_root
            git_calls.append(args)
            return ""

        with mock.patch("core.agent_task_runner.ensure_clean_worktree"):
            with mock.patch("core.agent_task_runner.comment_issue"):
                with mock.patch("core.agent_task_runner.update_project_status"):
                    with mock.patch("core.agent_task_runner.current_branch", return_value="main"):
                        with mock.patch("core.agent_task_runner.run_git", side_effect=fake_run_git):
                            with mock.patch(
                                "core.agent_task_runner.invoke_agent_processing",
                                return_value={"final_response": "ok", "tasks": []},
                            ):
                                with mock.patch("core.agent_task_runner.create_commit_push_pr", return_value="pr-url"):
                                    process_issue(
                                        issue=issue,
                                        status_info=status_info,
                                        runtime_paths=runtime_paths,
                                        config=mock.Mock(),
                                        manifest=mock.Mock(),
                                        llm_manager=mock.Mock(),
                                        runner_config=AgentTaskRunnerConfig(),
                                        run_dir=Path("run"),
                                    )

        self.assertIn(["fetch", "origin"], git_calls)
        self.assertIn(["pull", "--ff-only", "origin", "main"], git_calls)
        self.assertIn(["switch", "-c", "agenttask/test"], git_calls)
        self.assertNotIn(["switch", "main"], git_calls)

    def test_process_issue_does_not_create_branch_when_working_branch_matches_base(self) -> None:
        issue = _issue("## Base Branch\nmain\n\n## New Branch\nmain\n\n## Execution Prompt\nDo work.")
        status_info = ProjectStatusInfo(
            project_id="project",
            status_field_id="field",
            options={"In Progress": "in-progress", "Done": "done"},
        )
        runtime_paths = SimpleNamespace(host_root=Path("."))
        git_calls: list[list[str]] = []

        def fake_run_git(args: list[str], host_root: Path) -> str:
            del host_root
            git_calls.append(args)
            return ""

        with mock.patch("core.agent_task_runner.ensure_clean_worktree"):
            with mock.patch("core.agent_task_runner.comment_issue"):
                with mock.patch("core.agent_task_runner.update_project_status"):
                    with mock.patch("core.agent_task_runner.current_branch", return_value="current-feature"):
                        with mock.patch("core.agent_task_runner.run_git", side_effect=fake_run_git):
                            with mock.patch(
                                "core.agent_task_runner.invoke_agent_processing",
                                return_value={"final_response": "ok", "tasks": []},
                            ):
                                with mock.patch("core.agent_task_runner.create_commit_push_pr", return_value="pushed") as delivery:
                                    process_issue(
                                        issue=issue,
                                        status_info=status_info,
                                        runtime_paths=runtime_paths,
                                        config=mock.Mock(),
                                        manifest=mock.Mock(),
                                        llm_manager=mock.Mock(),
                                        runner_config=AgentTaskRunnerConfig(),
                                        run_dir=Path("run"),
                                    )

        self.assertIn(["fetch", "origin"], git_calls)
        self.assertIn(["pull", "--ff-only", "origin", "current-feature"], git_calls)
        self.assertNotIn(["switch", "main"], git_calls)
        self.assertNotIn(["switch", "-c", "main"], git_calls)
        self.assertEqual(delivery.call_args.kwargs["head_branch"], "current-feature")
        self.assertFalse(delivery.call_args.kwargs["create_pr"])

    def test_process_issue_does_not_mutate_github_when_worktree_is_dirty(self) -> None:
        issue = _issue("## Base Branch\nmain\n\n## Execution Prompt\nDo work.")
        status_info = ProjectStatusInfo(
            project_id="project",
            status_field_id="field",
            options={"In Progress": "in-progress", "Done": "done"},
        )

        with mock.patch("core.agent_task_runner.ensure_clean_worktree", side_effect=RunnerError("dirty")):
            with mock.patch("core.agent_task_runner.comment_issue") as comment_issue:
                with mock.patch("core.agent_task_runner.update_project_status") as update_project_status:
                    with self.assertRaisesRegex(RunnerError, "dirty"):
                        process_issue(
                            issue=issue,
                            status_info=status_info,
                            runtime_paths=SimpleNamespace(host_root=Path(".")),
                            config=mock.Mock(),
                            manifest=mock.Mock(),
                            llm_manager=mock.Mock(),
                            runner_config=AgentTaskRunnerConfig(),
                            run_dir=Path("run"),
                        )

        comment_issue.assert_not_called()
        update_project_status.assert_not_called()

    def test_validate_branch_plan_requires_existing_base_for_new_branch(self) -> None:
        parsed = parse_agent_task(
            _issue("## Base Branch\nmissing-base\n\n## New Branch\nagenttask/test\n\n## Execution Prompt\nDo work.")
        )
        with mock.patch("core.agent_task_runner.git_ref_exists", return_value=False):
            with self.assertRaisesRegex(LocalPreflightError, "missing-base"):
                validate_branch_plan(Path("."), parsed)

    def test_validate_branch_plan_allows_current_branch_mode_without_base_ref(self) -> None:
        parsed = parse_agent_task(_issue("## Base Branch\nsame\n\n## New Branch\nsame\n\n## Execution Prompt\nDo work."))
        with mock.patch("core.agent_task_runner.git_ref_exists") as git_ref_exists:
            validate_branch_plan(Path("."), parsed)
        git_ref_exists.assert_not_called()

    def test_loop_does_not_comment_when_local_preflight_fails(self) -> None:
        issue = _issue("## Base Branch\nmain\n\n## Execution Prompt\nDo work.")
        runner_config = AgentTaskRunnerConfig()

        with mock.patch("core.agent_task_runner.validate_environment"):
            with mock.patch(
                "core.agent_task_runner.get_project_status_info",
                return_value=ProjectStatusInfo("project", "field", {"Blocked": "blocked"}),
            ):
                with mock.patch("core.agent_task_runner.list_todo_agent_tasks", return_value=[issue]):
                    with mock.patch("core.agent_task_runner.process_issue", side_effect=LocalPreflightError("dirty")):
                        with mock.patch("core.agent_task_runner.comment_issue") as comment_issue:
                            with mock.patch("core.agent_task_runner.update_project_status") as update_project_status:
                                processed = run_agent_task_loop(
                                    runtime_paths=SimpleNamespace(host_root=Path(".")),
                                    config=mock.Mock(),
                                    manifest=mock.Mock(),
                                    llm_manager=mock.Mock(),
                                    runner_config=runner_config,
                                    run_dir_factory=lambda: Path("run"),
                                )

        self.assertEqual(processed, 0)
        comment_issue.assert_not_called()
        update_project_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
