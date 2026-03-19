from __future__ import annotations

import argparse
from datetime import datetime
import shutil
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
from typing import Iterable


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


def iter_project_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        if any(part in {".agentswarm", "__pycache__", ".pytest_cache"} for part in parts):
            continue
        yield relative


def list_changed_files(source_root: Path, result_root: Path) -> list[str]:
    source_files = {path.as_posix() for path in iter_project_files(source_root)}
    result_files = {path.as_posix() for path in iter_project_files(result_root)}
    changed: list[str] = []
    for relative_str in sorted(source_files | result_files):
        source_path = source_root / relative_str
        result_path = result_root / relative_str
        if not source_path.exists() or not result_path.exists():
            changed.append(relative_str)
            continue
        if source_path.read_bytes() != result_path.read_bytes():
            changed.append(relative_str)
    return changed


def find_latest_workflow_artifact_dir(host_root: Path) -> Path | None:
    runs_root = host_root / ".agentswarm" / "runs"
    if not runs_root.exists():
        return None
    workflow_dirs = [path for path in runs_root.glob("*/tasks/*/*") if path.is_dir()]
    if not workflow_dirs:
        return None
    return max(workflow_dirs, key=lambda path: path.stat().st_mtime)


def read_text_if_exists(path: Path | None) -> str:
    if path is None or not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def extract_prefixed_value(markdown: str, prefix: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return ""


def extract_section_lines(markdown: str, heading: str) -> list[str]:
    lines = markdown.splitlines()
    collecting = False
    section: list[str] = []
    for line in lines:
        if line.strip() == heading:
            collecting = True
            continue
        if collecting and line.startswith("## "):
            break
        if collecting:
            stripped = line.strip()
            if stripped:
                section.append(stripped)
    return section


def extract_section_bullets(markdown: str, heading: str) -> list[str]:
    bullets: list[str] = []
    for line in extract_section_lines(markdown, heading):
        if line.startswith("- "):
            bullets.append(line[2:].strip())
    return bullets


def extract_section_text(markdown: str, heading: str) -> str:
    lines = extract_section_lines(markdown, heading)
    return "\n".join(lines).strip()


def parse_round_number(path: Path) -> int:
    stem = path.stem
    suffix = stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def extract_review_score(markdown: str) -> str:
    overall_match = re.search(r"Overall Score:\s*(\d{1,3})\s*/\s*100", markdown, flags=re.IGNORECASE)
    if overall_match:
        return f"{overall_match.group(1)}/100"
    score = extract_prefixed_value(markdown, "- Score: ")
    return score or "Unknown"


def extract_review_decision(markdown: str) -> str:
    decision_match = re.search(r"Decision:\s*(.+)", markdown, flags=re.IGNORECASE)
    if decision_match:
        return decision_match.group(1).strip()
    decision = extract_prefixed_value(markdown, "- Decision: ")
    if decision:
        return decision
    approved = extract_prefixed_value(markdown, "- Approved: ")
    if approved:
        return "Approved" if approved.lower() == "true" else "Needs another pass"
    return "Unknown"


def extract_review_loop_status(markdown: str) -> str:
    return extract_prefixed_value(markdown, "- Loop Status: ") or extract_prefixed_value(markdown, "- Loop status: ") or "Unknown"


def extract_path_mentions(markdown: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", markdown):
        label = match.group(1).strip()
        if "/" in label or "\\" in label:
            candidates.append(label.replace("\\", "/"))
    for match in re.finditer(r"(?<![A-Za-z0-9_])((?:docs|design|src|tests|Source|Scripts|Validation|Checks)/[A-Za-z0-9_./-]+)", markdown):
        candidates.append(match.group(1).replace("\\", "/"))
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = candidate.strip().strip("`'\"")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def detect_artifact_flavor(artifact_dir: Path) -> str:
    if any(artifact_dir.glob("engineer_investigation_round_*.md")):
        return "gameplay-engineer"
    if any(artifact_dir.glob("investigation_round_*.md")):
        return "root-project-investigation"
    return "unknown"


def collect_investigation_round_docs(artifact_dir: Path) -> tuple[str, list[Path], dict[int, Path]]:
    flavor = detect_artifact_flavor(artifact_dir)
    if flavor == "gameplay-engineer":
        return (
            flavor,
            sorted(artifact_dir.glob("engineer_investigation_round_*.md"), key=parse_round_number),
            {parse_round_number(path): path for path in artifact_dir.glob("investigation_review_round_*.md")},
        )
    if flavor == "root-project-investigation":
        return (
            flavor,
            sorted(artifact_dir.glob("investigation_round_*.md"), key=parse_round_number),
            {parse_round_number(path): path for path in artifact_dir.glob("review_round_*.md")},
        )
    return flavor, [], {}


def summarize_changed_files(title: str, changed_files: list[str]) -> list[str]:
    lines = [title, ""]
    if changed_files:
        lines.extend(f"- {item}" for item in changed_files)
    else:
        lines.append("- No changed files detected.")
    return lines


def summarize_investigation_loop(artifact_dir: Path | None, direct_changed_files: list[str]) -> list[str]:
    lines = ["## AgentSwarm Investigation Loop vs Direct Codex CLI", ""]
    if artifact_dir is None or not artifact_dir.exists():
        lines.append("- No AgentSwarm workflow artifact directory was found.")
        return lines

    direct_changed = set(direct_changed_files)
    artifact_flavor, round_docs, review_docs = collect_investigation_round_docs(artifact_dir)
    if not round_docs:
        lines.append("- No investigation rounds were recorded.")
        return lines

    final_report_md = read_text_if_exists(artifact_dir / "final_report.md")
    lines.append(f"- Artifact flavor: {artifact_flavor}")
    if final_report_md:
        lines.append(f"- Final workflow status: {extract_prefixed_value(final_report_md, '- Status: ') or 'Unknown'}")
        lines.append(f"- Final workflow loop status: {extract_prefixed_value(final_report_md, '- Loop Status: ') or 'Unknown'}")
    lines.append(f"- Direct final changed files: {', '.join(direct_changed_files) if direct_changed_files else 'None.'}")
    for round_doc in round_docs:
        round_number = parse_round_number(round_doc)
        investigation_md = read_text_if_exists(round_doc)
        review_md = read_text_if_exists(review_docs.get(round_number))
        if artifact_flavor == "gameplay-engineer":
            runtime_paths = extract_section_bullets(investigation_md, "## Current Runtime Paths")
            source_hits = extract_section_bullets(investigation_md, "## Source Hits")
            test_hits = extract_section_bullets(investigation_md, "## Test Hits")
            ownership_summary = extract_section_text(investigation_md, "## Ownership Summary") or "None."
            implementation_medium = extract_prefixed_value(investigation_md, "Implementation Medium: ") or "Unknown"
        else:
            path_mentions = extract_path_mentions(investigation_md)
            runtime_paths = []
            source_hits = [path for path in path_mentions if path.startswith(("src/", "Source/", "Scripts/"))]
            test_hits = [path for path in path_mentions if path.startswith(("tests/", "Validation/", "Checks/"))]
            ownership_summary = extract_section_text(investigation_md, "## Candidate Ownership") or "None."
            implementation_medium = "investigation-only"
        overlap = sorted({*runtime_paths, *source_hits, *test_hits} & direct_changed)
        blocking_issues = extract_section_bullets(review_md, "## Blocking Issues")
        lines.extend(
            [
                "",
                f"### Investigation Round {round_number}",
                f"- Decision: {extract_review_decision(review_md)}",
                f"- Score: {extract_review_score(review_md)}",
                f"- Implementation medium: {implementation_medium}",
                f"- Current runtime paths: {', '.join(runtime_paths) if runtime_paths else 'None.'}",
                f"- Source hits: {', '.join(source_hits) if source_hits else 'None.'}",
                f"- Test hits: {', '.join(test_hits) if test_hits else 'None.'}",
                f"- Overlap with Direct final files: {', '.join(overlap) if overlap else 'None.'}",
                f"- Blocking issues: {' | '.join(blocking_issues) if blocking_issues else 'None.'}",
                f"- Ownership summary: {ownership_summary}",
            ]
        )
    return lines


def summarize_review_loop(artifact_dir: Path | None, direct_changed_files: list[str]) -> list[str]:
    lines = ["## AgentSwarm Architecture Review Loop vs Direct Codex CLI", ""]
    if artifact_dir is None or not artifact_dir.exists():
        lines.append("- No AgentSwarm workflow artifact directory was found.")
        return lines

    artifact_flavor = detect_artifact_flavor(artifact_dir)
    review_docs = sorted(artifact_dir.glob("review_round_*.md"), key=parse_round_number)
    if not review_docs:
        lines.append("- No architecture review rounds were recorded for this run.")
        return lines

    direct_has_source = any("/tests/" not in f"/{path}/" and path.startswith(("src/", "Source/", "Scripts/")) for path in direct_changed_files)
    direct_has_tests = any("/tests/" in f"/{path}/" or path.startswith(("tests/", "Validation/")) for path in direct_changed_files)
    final_report_md = read_text_if_exists(artifact_dir / "final_report.md")
    lines.append(f"- Artifact flavor: {artifact_flavor}")
    if final_report_md:
        lines.append(f"- Final workflow status: {extract_prefixed_value(final_report_md, '- Status: ') or 'Unknown'}")
        lines.append(f"- Final workflow loop status: {extract_prefixed_value(final_report_md, '- Loop Status: ') or 'Unknown'}")
    lines.append(f"- Direct final source changes present: {direct_has_source}")
    lines.append(f"- Direct final test changes present: {direct_has_tests}")
    for review_doc in review_docs:
        round_number = parse_round_number(review_doc)
        review_md = read_text_if_exists(review_doc)
        blocking_issues = extract_section_bullets(review_md, "## Blocking Issues")
        improvement_items = extract_section_bullets(review_md, "## Improvement Checklist")
        lines.extend(
            [
                "",
                f"### Review Round {round_number}",
                f"- Approved: {extract_prefixed_value(review_md, '- Approved: ') or ('True' if extract_review_decision(review_md).lower() == 'approve' else 'False' if extract_review_decision(review_md).lower() == 'revise' else 'Unknown')}",
                f"- Score: {extract_review_score(review_md)}",
                f"- Loop status: {extract_review_loop_status(review_md)}",
                f"- Direct final source changes present already: {direct_has_source}",
                f"- Direct final test changes present already: {direct_has_tests}",
                f"- Blocking issues: {' | '.join(blocking_issues) if blocking_issues else 'None.'}",
                f"- Improvement checklist: {' | '.join(improvement_items) if improvement_items else 'None.'}",
            ]
        )
    return lines


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
    direct_changed_files = list_changed_files(source_host_root, direct_host_root)
    agentswarm_changed_files = list_changed_files(source_host_root, agentswarm_host_root)
    artifact_dir = find_latest_workflow_artifact_dir(agentswarm_host_root)
    return "\n".join(
        [
            "# Codex CLI vs AgentSwarm Comparison",
            "",
            f"- Generated at: {generated_at}",
            f"- Source host root: `{source_host_root}`",
            f"- Direct Codex host copy: `{direct_host_root}`",
            f"- AgentSwarm host copy: `{agentswarm_host_root}`",
            f"- AgentSwarm root: `{agent_root}`",
            f"- Latest AgentSwarm workflow artifact: `{artifact_dir}`" if artifact_dir else "- Latest AgentSwarm workflow artifact: `None`",
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
            "",
            *summarize_changed_files("## Direct Codex CLI Final Changed Files", direct_changed_files),
            "",
            *summarize_changed_files("## AgentSwarm Final Changed Files", agentswarm_changed_files),
            "",
            *summarize_investigation_loop(artifact_dir, direct_changed_files),
            "",
            *summarize_review_loop(artifact_dir, direct_changed_files),
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
