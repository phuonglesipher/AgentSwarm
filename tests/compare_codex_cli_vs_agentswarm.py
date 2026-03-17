from __future__ import annotations

import argparse
from datetime import datetime
import shutil
from pathlib import Path
import os
import subprocess
import sys
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the same prompt through direct Codex CLI and AgentSwarm, then save a comparison bundle."
    )
    parser.add_argument("--host-root", required=True, help="Host project root used for both runs.")
    parser.add_argument("--prompt", required=True, help="Prompt to compare across both execution modes.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to a short path under the system temp directory.",
    )
    parser.add_argument("--codex-command", default=os.getenv("CODEX_COMMAND", "codex"), help="Codex CLI command.")
    parser.add_argument(
        "--model",
        default=os.getenv("CODEX_MODEL", "gpt-5.3-codex"),
        help="Model name for the direct Codex CLI run.",
    )
    return parser.parse_args()


def run_command(
    *,
    command: list[str],
    cwd: Path,
    prompt: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=prompt,
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def copy_host_project(source_root: Path, destination_root: Path) -> None:
    if destination_root.exists():
        shutil.rmtree(destination_root)
    shutil.copytree(
        source_root,
        destination_root,
        ignore=shutil.ignore_patterns(".agentswarm", "__pycache__", ".pytest_cache"),
    )


def resolve_command_path(command: str) -> str:
    if not command:
        return command
    if Path(command).suffix or Path(command).is_absolute():
        return command

    candidates = [command]
    if os.name == "nt":
        candidates = [f"{command}.cmd", f"{command}.exe", command]

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return command


def build_report(
    *,
    prompt: str,
    source_host_root: Path,
    direct_host_root: Path,
    agentswarm_host_root: Path,
    agent_root: Path,
    direct_result: subprocess.CompletedProcess[str],
    direct_stdout: Path,
    direct_stderr: Path,
    agentswarm_result: subprocess.CompletedProcess[str],
    agentswarm_stdout: Path,
    agentswarm_stderr: Path,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return "\n".join(
        [
            "# Codex CLI vs AgentSwarm Comparison",
            "",
            f"- Generated at: {generated_at}",
            f"- Source host root: `{source_host_root}`",
            f"- Direct Codex host copy: `{direct_host_root}`",
            f"- AgentSwarm host copy: `{agentswarm_host_root}`",
            f"- AgentSwarm root: `{agent_root}`",
            "",
            "## Prompt",
            "",
            prompt,
            "",
            "## Direct Codex CLI",
            "",
            f"- Exit code: {direct_result.returncode}",
            f"- Stdout: `{direct_stdout}`",
            f"- Stderr: `{direct_stderr}`",
            "",
            "## AgentSwarm",
            "",
            f"- Exit code: {agentswarm_result.returncode}",
            f"- Stdout: `{agentswarm_stdout}`",
            f"- Stderr: `{agentswarm_stderr}`",
            "",
            "## Review Checklist",
            "",
            "- Did both runs identify the same gameplay problem?",
            "- Did direct Codex CLI reference the correct host-project files?",
            "- Did AgentSwarm classify the task correctly and route to the expected workflow?",
            "- Did AgentSwarm create useful artifacts such as traversal logs or task reports?",
            "- Which run produced the clearer next action for implementation?",
        ]
    )


def main() -> None:
    args = parse_args()
    agent_root = Path(__file__).resolve().parents[1]
    host_root = Path(args.host_root).resolve()
    if not host_root.exists():
        raise SystemExit(f"Host root does not exist: {host_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(tempfile.gettempdir()) / f"as-compare-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_host_root = output_dir / "direct"
    agentswarm_host_root = output_dir / "agent"
    copy_host_project(host_root, direct_host_root)
    copy_host_project(host_root, agentswarm_host_root)

    direct_stdout = output_dir / "direct-codex.stdout.txt"
    direct_stderr = output_dir / "direct-codex.stderr.txt"
    agentswarm_stdout = output_dir / "agentswarm.stdout.txt"
    agentswarm_stderr = output_dir / "agentswarm.stderr.txt"
    report_path = output_dir / "comparison.md"

    direct_command = [
        resolve_command_path(args.codex_command),
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--color",
        "never",
        "--sandbox",
        "workspace-write",
        "-m",
        args.model,
        "-",
    ]
    direct_result = run_command(command=direct_command, cwd=direct_host_root, prompt=args.prompt)
    write_text(direct_stdout, direct_result.stdout)
    write_text(direct_stderr, direct_result.stderr)

    agentswarm_command = [
        sys.executable,
        str(agent_root / "main.py"),
        "--host-root",
        str(agentswarm_host_root),
        "--prompt",
        args.prompt,
    ]
    agentswarm_result = run_command(command=agentswarm_command, cwd=agent_root)
    write_text(agentswarm_stdout, agentswarm_result.stdout)
    write_text(agentswarm_stderr, agentswarm_result.stderr)

    write_text(
        report_path,
        build_report(
            prompt=args.prompt,
            source_host_root=host_root,
            direct_host_root=direct_host_root,
            agentswarm_host_root=agentswarm_host_root,
            agent_root=agent_root,
            direct_result=direct_result,
            direct_stdout=direct_stdout,
            direct_stderr=direct_stderr,
            agentswarm_result=agentswarm_result,
            agentswarm_stdout=agentswarm_stdout,
            agentswarm_stderr=agentswarm_stderr,
        ),
    )

    print(f"Comparison bundle: {output_dir}")
    print(f"Review report: {report_path}")
    print(f"Direct host copy: {direct_host_root}")
    print(f"AgentSwarm host copy: {agentswarm_host_root}")


if __name__ == "__main__":
    main()
