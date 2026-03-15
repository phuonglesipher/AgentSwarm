from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest

from langgraph.graph import END, START, MessagesState, StateGraph
from typing_extensions import NotRequired, TypedDict

from core.graph_logging import log_graph_payload_event, trace_graph_node, trace_route_decision
from core.llm import LLMError
from core.models import WorkflowContext, WorkflowMetadata
from core.tool_graph import build_tool_call_message, find_latest_tool_message
from core.text_utils import normalize_text, slugify, tokenize


class EngineerState(MessagesState):
    prompt: str
    task_prompt: str
    task_id: str
    run_dir: str
    task_type: str
    task_type_reason: str
    doc_hits: list[str]
    doc_context: str
    design_doc: str
    plan_doc: str
    review_round: int
    review_score: int
    review_feedback: str
    missing_sections: list[str]
    code_attempt: int
    generated_code: str
    generated_tests: str
    implementation_notes: str
    compile_ok: bool
    tests_ok: bool
    test_output: str
    artifact_dir: str
    final_report: dict[str, Any]
    summary: str
    score: NotRequired[int]
    feedback: NotRequired[str]
    approved: NotRequired[bool]
    pending_tool_name: str
    pending_tool_call_id: str


REQUIRED_PLAN_SECTIONS = {
    "Task Type": "Describe whether this is a bug-fix or a new feature.",
    "Existing Docs": "Reference gameplay and design docs that informed the work.",
    "Implementation Steps": "Describe the implementation sequence and system touch points.",
    "Unit Tests": "List the unit tests that must exist before code is considered done.",
    "Risks": "Record the likely implementation risks and fallback plans.",
    "Acceptance Criteria": "State the observable gameplay outcome when the task is complete.",
}


def _fallback_task_classification(task_prompt: str) -> tuple[str, str]:
    bugfix_keywords = {"fix", "bug", "issue", "error", "crash"}
    normalized = normalize_text(task_prompt)
    task_type = "bugfix" if bugfix_keywords & tokenize(normalized) else "feature"
    reason = "Detected bug-fix vocabulary in the task prompt." if task_type == "bugfix" else "Defaulted to feature flow."
    return task_type, reason


def _compose_design_doc(task_prompt: str, task_type: str, doc_hits: list[str], doc_context: str) -> str:
    lines = [
        "# Gameplay Design Context",
        "",
        f"Task Prompt: {task_prompt}",
        f"Task Type: {task_type}",
        "",
        "## Existing References",
    ]
    if doc_hits:
        lines.extend([f"- {item}" for item in doc_hits])
    else:
        lines.extend(
            [
                "- No existing gameplay or design doc matched the task closely enough.",
                "- A fresh design baseline was created from the incoming prompt.",
            ]
        )

    lines.extend(
        [
            "",
            "## Design Intent",
            "- Keep the gameplay change readable for engineering and design partners.",
            "- Describe expected player-facing behavior first, then implementation notes.",
        ]
    )
    if doc_context:
        lines.extend(["", "## Reference Snippets", doc_context])
    return "\n".join(lines)


def _compose_initial_plan(task_prompt: str, task_type: str, doc_hits: list[str]) -> str:
    references = "\n".join(f"- {item}" for item in doc_hits) or "- No matching docs found."
    return "\n".join(
        [
            "# Gameplay Implementation Plan",
            "",
            "## Overview",
            f"- Deliver work for: {task_prompt}",
            "",
            "## Task Type",
            f"- {task_type}",
            "",
            "## Existing Docs",
            references,
            "",
            "## Implementation Steps",
            "- Inspect related gameplay systems and identify touch points.",
            "- Update the gameplay behavior and preserve predictable state transitions.",
            "- Capture expected debug signals for fast regression checks.",
        ]
    )


def _revise_plan(plan_doc: str, missing_sections: list[str], task_prompt: str, review_feedback: str) -> str:
    additions: list[str] = []
    for section in missing_sections:
        guidance = REQUIRED_PLAN_SECTIONS.get(section, "Add the missing section with concrete details.")
        additions.extend(
            [
                "",
                f"## {section}",
                f"- {guidance}",
                f"- Task context: {task_prompt}",
                f"- Reviewer note: {review_feedback}",
            ]
        )
    return plan_doc + "\n" + "\n".join(additions)


def _fallback_code_bundle(task_prompt: str, task_type: str, attempt: int) -> dict[str, str]:
    normalized_prompt = normalize_text(task_prompt)
    task_slug = slugify(task_prompt, fallback="gameplay-task").replace("-", "_")
    expected_unit_tests = ["compiles", "returns_task_metadata", "captures_task_type", "records_review_score"]

    if attempt == 1:
        source_code = "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def build_gameplay_change_summary() -> dict:",
                "    return {",
                f'        "task_id": "{task_slug}",',
                f'        "task_type": "{task_type}",',
                f'        "prompt": "{normalized_prompt}",',
                '        "implementation_status": "draft",',
                "    }",
            ]
        )
    else:
        source_code = "\n".join(
            [
                "from __future__ import annotations",
                "",
                "def build_gameplay_change_summary() -> dict:",
                "    return {",
                f'        "task_id": "{task_slug}",',
                f'        "task_type": "{task_type}",',
                f'        "prompt": "{normalized_prompt}",',
                '        "implementation_status": "ready-for-review",',
                f'        "unit_tests": {expected_unit_tests},',
                "    }",
            ]
        )

    test_code = "\n".join(
        [
            "from gameplay_change import build_gameplay_change_summary",
            "",
            "def test_build_gameplay_change_summary():",
            "    summary = build_gameplay_change_summary()",
            '    assert summary["task_type"] in {"bugfix", "feature"}',
            '    assert summary["implementation_status"] == "ready-for-review"',
            '    assert "unit_tests" in summary',
        ]
    )
    notes = "Deterministic fallback code bundle was generated."
    return {
        "source_code": source_code,
        "test_code": test_code,
        "implementation_notes": notes,
    }


def _build_tool_call_id(state: EngineerState, tool_name: str) -> str:
    message_count = len(state.get("messages", []))
    return f"{state['task_id']}-{tool_name}-{message_count + 1}"


def _extract_tool_artifact(state: EngineerState, tool_name: str) -> dict[str, Any]:
    tool_message = find_latest_tool_message(
        list(state.get("messages", [])),
        tool_name=tool_name,
        tool_call_id=state.get("pending_tool_call_id") or None,
    )
    if tool_message is None:
        raise RuntimeError(f"Expected ToolMessage for {tool_name}, but no matching tool result was found.")
    if isinstance(tool_message.artifact, dict):
        return tool_message.artifact
    return {}


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _run_compile_and_tests(source_code: str, test_code: str) -> tuple[bool, bool, str]:
    try:
        compile(source_code, "gameplay_change.py", "exec")
    except SyntaxError as exc:
        return False, False, f"Compile error in gameplay_change.py: {exc}"

    try:
        compile(test_code, "test_gameplay_change.py", "exec")
    except SyntaxError as exc:
        return False, False, f"Compile error in test_gameplay_change.py: {exc}"

    aliased_module_names = [
        "gameplay_change",
        "solution",
        "main",
        "gameplay_change_summary",
    ]
    previous_modules = {name: sys.modules.get(name) for name in aliased_module_names}
    generated_test_module_name = "_generated_test_gameplay_change"
    previous_generated_test_module = sys.modules.get(generated_test_module_name)
    original_sys_path = list(sys.path)

    try:
        with tempfile.TemporaryDirectory(prefix="gameplay-selftest-") as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "gameplay_change.py"
            test_path = temp_path / "test_gameplay_change.py"
            source_path.write_text(source_code, encoding="utf-8")
            test_path.write_text(test_code, encoding="utf-8")

            sys.path.insert(0, str(temp_path))

            source_module = _load_module_from_path("gameplay_change", source_path)
            for alias in aliased_module_names:
                sys.modules[alias] = source_module

            test_module = _load_module_from_path(generated_test_module_name, test_path)
            function_tests = [
                value
                for name, value in vars(test_module).items()
                if name.startswith("test_") and callable(value)
            ]
            for test_func in function_tests:
                test_func()

            suite = unittest.defaultTestLoader.loadTestsFromModule(test_module)
            unittest_cases = suite.countTestCases()
            if unittest_cases:
                stream = io.StringIO()
                result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
                if not result.wasSuccessful():
                    return True, False, stream.getvalue().strip()

            total_tests = len(function_tests) + unittest_cases
            if total_tests == 0:
                return True, False, "No generated test functions or unittest cases were found."

    except AssertionError as exc:
        message = str(exc) or "Generated assertion failed."
        return True, False, f"Unit test failed: {message}"
    except SyntaxError as exc:
        file_name = Path(exc.filename or "").name or "generated file"
        return False, False, f"Compile error in {file_name}: {exc}"
    except Exception as exc:
        return True, False, f"Unit test execution failed: {exc}"
    finally:
        sys.path[:] = original_sys_path
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module
        if previous_generated_test_module is None:
            sys.modules.pop(generated_test_module_name, None)
        else:
            sys.modules[generated_test_module_name] = previous_generated_test_module

    return True, True, f"Compile and {total_tests} generated test(s) passed."


def build_graph(context: WorkflowContext, metadata: WorkflowMetadata):
    default_llm = context.llm
    graph_name = metadata.name
    reviewer_graph = context.get_workflow_graph("gameplay-reviewer-workflow")
    tool_subgraphs = context.register_tools(metadata.tools, EngineerState)

    def classify_request(state: EngineerState) -> dict[str, Any]:
        artifact_dir = Path(state["run_dir"]) / "tasks" / state["task_id"] / metadata.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        task_type, task_type_reason = _fallback_task_classification(state["task_prompt"])

        if default_llm.is_enabled():
            schema = {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "enum": ["bugfix", "feature"]},
                    "reason": {"type": "string"},
                },
                "required": ["task_type", "reason"],
                "additionalProperties": False,
            }
            try:
                result = default_llm.generate_json(
                    instructions=(
                        "You are gameplay-engineer-workflow. Decide whether the task is a gameplay bugfix or a new gameplay feature. "
                        "Use bugfix only when the request is about fixing an issue, regression, crash, or unintended behavior."
                    ),
                    input_text=f"Task prompt:\n{state['task_prompt']}",
                    schema_name="gameplay_task_classification",
                    schema=schema,
                )
                task_type = result["task_type"]
                task_type_reason = result["reason"]
            except LLMError:
                pass

        return {
            "task_type": task_type,
            "task_type_reason": task_type_reason,
            "review_round": 0,
            "review_score": 0,
            "code_attempt": 0,
            "artifact_dir": str(artifact_dir),
        }

    def prepare_doc_search(state: EngineerState) -> dict[str, Any]:
        tool_name = "find-gameplay-docs"
        call_id = _build_tool_call_id(state, tool_name)
        return {
            "pending_tool_name": tool_name,
            "pending_tool_call_id": call_id,
            "messages": [
                build_tool_call_message(
                    tool_name,
                    {"task_prompt": state["task_prompt"]},
                    call_id,
                    content="Find the gameplay and design docs most relevant to this task.",
                )
            ],
        }

    def route_tool_request(state: EngineerState) -> str:
        tool_name = state.get("pending_tool_name", "").strip()
        if tool_name in tool_subgraphs:
            return tool_name
        return "tool_request_error"

    def tool_request_error(state: EngineerState) -> dict[str, Any]:
        raise RuntimeError(
            f"{metadata.name} requested an unregistered tool: {state.get('pending_tool_name') or '(empty)'}"
        )

    def capture_doc_hits(state: EngineerState) -> dict[str, Any]:
        artifact = _extract_tool_artifact(state, "find-gameplay-docs")
        raw_hits = artifact.get("doc_hits", [])
        doc_hits = [str(item) for item in raw_hits] if isinstance(raw_hits, list) else []
        return {
            "doc_hits": doc_hits,
            "pending_tool_name": "",
            "pending_tool_call_id": "",
        }

    def prepare_doc_context_lookup(state: EngineerState) -> dict[str, Any]:
        tool_name = "load-markdown-context"
        call_id = _build_tool_call_id(state, tool_name)
        return {
            "pending_tool_name": tool_name,
            "pending_tool_call_id": call_id,
            "messages": [
                build_tool_call_message(
                    tool_name,
                    {"doc_paths": state["doc_hits"], "max_chars": 2000},
                    call_id,
                    content="Load markdown snippets for the matched gameplay and design docs.",
                )
            ],
        }

    def capture_doc_context(state: EngineerState) -> dict[str, Any]:
        artifact = _extract_tool_artifact(state, "load-markdown-context")
        doc_context = str(artifact.get("doc_context") or "")
        return {
            "doc_context": doc_context,
            "pending_tool_name": "",
            "pending_tool_call_id": "",
        }

    def build_design_doc(state: EngineerState) -> dict[str, Any]:
        design_doc = _compose_design_doc(
            state["task_prompt"],
            state["task_type"],
            state["doc_hits"],
            state["doc_context"],
        )
        if default_llm.is_enabled():
            try:
                design_doc = default_llm.generate_text(
                    instructions=(
                        "You are gameplay-engineer-workflow. Write a concise markdown design context document for a gameplay task. "
                        "Include sections: Overview, Existing References, Player-Facing Behavior, Technical Notes, Risks. "
                        "Ground the design in the provided docs when they exist."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Task type: {state['task_type']}\n"
                        f"Classification reason: {state['task_type_reason']}\n\n"
                        f"Doc hits:\n{state['doc_hits']}\n\n"
                        f"Doc context:\n{state['doc_context'] or 'No matching docs.'}"
                    ),
                )
            except LLMError:
                pass

        artifact_dir = Path(state["artifact_dir"])
        (artifact_dir / "design_doc.md").write_text(design_doc, encoding="utf-8")
        return {"design_doc": design_doc}

    def plan_work(state: EngineerState) -> dict[str, Any]:
        plan_doc = _compose_initial_plan(state["task_prompt"], state["task_type"], state["doc_hits"])
        if default_llm.is_enabled():
            try:
                plan_doc = default_llm.generate_text(
                    instructions=(
                        "You are gameplay-engineer-workflow. Produce a markdown implementation plan for gameplay work. "
                        "The document must contain these exact sections: Overview, Task Type, Existing Docs, Implementation Steps, "
                        "Unit Tests, Risks, Acceptance Criteria. Each section must have concrete bullets."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Task type: {state['task_type']}\n"
                        f"Design doc:\n{state['design_doc']}\n\n"
                        f"Relevant docs:\n{state['doc_context'] or 'No matching docs.'}"
                    ),
                )
            except LLMError:
                pass

        artifact_dir = Path(state["artifact_dir"])
        (artifact_dir / "plan_doc.md").write_text(plan_doc, encoding="utf-8")
        return {"plan_doc": plan_doc}

    def request_review(state: EngineerState) -> dict[str, Any]:
        review_round = state["review_round"] + 1
        return {"review_round": review_round}

    def enter_review_subgraph(state: EngineerState) -> dict[str, Any]:
        review_request = {
            "task_prompt": state["task_prompt"],
            "plan_doc": state["plan_doc"],
            "review_round": state["review_round"],
            "task_id": state["task_id"],
            "run_dir": state["run_dir"],
        }
        log_graph_payload_event(
            state=state,
            graph_name=graph_name,
            node_name="gameplay-reviewer-workflow",
            phase="SUBGRAPH_ENTER",
            payload_label="input",
            payload=review_request,
        )
        return {}

    def capture_review_result(state: EngineerState) -> dict[str, Any]:
        review_result = {
            "score": state["score"],
            "feedback": state["feedback"],
            "missing_sections": state["missing_sections"],
            "approved": state["approved"],
            "summary": state["summary"],
        }
        log_graph_payload_event(
            state=state,
            graph_name=graph_name,
            node_name="gameplay-reviewer-workflow",
            phase="SUBGRAPH_EXIT",
            payload_label="output",
            payload=review_result,
        )
        artifact_dir = Path(state["artifact_dir"])
        feedback_lines = [
            f"# Review Round {state['review_round']}",
            "",
            f"Score: {state['score']}",
            "",
            "## Feedback",
            state["feedback"],
        ]
        (artifact_dir / f"review_round_{state['review_round']}.md").write_text(
            "\n".join(feedback_lines),
            encoding="utf-8",
        )
        return {
            "review_score": state["score"],
            "review_feedback": state["feedback"],
            "missing_sections": state["missing_sections"],
        }

    def review_gate(state: EngineerState) -> str:
        if state["review_score"] < 100:
            return "revise_plan"
        return "implement_code"

    def revise_plan(state: EngineerState) -> dict[str, Any]:
        revised_plan = _revise_plan(
            state["plan_doc"],
            state["missing_sections"],
            state["task_prompt"],
            state["review_feedback"],
        )
        if default_llm.is_enabled():
            try:
                revised_plan = default_llm.generate_text(
                    instructions=(
                        "You are gameplay-engineer-workflow. Rewrite the full markdown implementation plan after reviewer feedback. "
                        "Keep the exact sections Overview, Task Type, Existing Docs, Implementation Steps, Unit Tests, Risks, "
                        "Acceptance Criteria, and make sure all reviewer concerns are addressed."
                    ),
                    input_text=(
                        f"Task prompt:\n{state['task_prompt']}\n\n"
                        f"Current plan:\n{state['plan_doc']}\n\n"
                        f"Missing sections:\n{state['missing_sections']}\n\n"
                        f"Reviewer feedback:\n{state['review_feedback']}\n\n"
                        f"Design doc:\n{state['design_doc']}"
                    ),
                )
            except LLMError:
                pass

        artifact_dir = Path(state["artifact_dir"])
        (artifact_dir / "plan_doc.md").write_text(revised_plan, encoding="utf-8")
        return {"plan_doc": revised_plan}

    def _generate_code_bundle(state: EngineerState, error_context: str = "") -> dict[str, str]:
        fallback = _fallback_code_bundle(state["task_prompt"], state["task_type"], state["code_attempt"] + 1)
        codegen_llm = context.get_llm("codegen")
        if not codegen_llm.is_enabled():
            return fallback

        schema = {
            "type": "object",
            "properties": {
                "source_code": {"type": "string"},
                "test_code": {"type": "string"},
                "implementation_notes": {"type": "string"},
            },
            "required": ["source_code", "test_code", "implementation_notes"],
            "additionalProperties": False,
        }
        try:
            result = codegen_llm.generate_json(
                instructions=(
                    "You are gameplay-engineer-workflow code generation. Return Python source code and Python tests. "
                    "Do not use markdown fences. The source file must define build_gameplay_change_summary() -> dict. "
                    "The returned dict must include task_type, implementation_status set to 'ready-for-review', and unit_tests. "
                    "The test file must be plain Python with assert statements and no external dependencies."
                ),
                input_text=(
                    f"Task prompt:\n{state['task_prompt']}\n\n"
                    f"Task type: {state['task_type']}\n\n"
                    f"Design doc:\n{state['design_doc']}\n\n"
                    f"Implementation plan:\n{state['plan_doc']}\n\n"
                    f"Reviewer feedback:\n{state['review_feedback']}\n\n"
                    f"Previous self-test output:\n{error_context or 'None'}"
                ),
                schema_name="gameplay_code_bundle",
                schema=schema,
            )
        except LLMError:
            return fallback

        source_code = str(result["source_code"]).strip()
        test_code = str(result["test_code"]).strip()
        if not source_code or not test_code:
            return fallback
        return {
            "source_code": source_code,
            "test_code": test_code,
            "implementation_notes": str(result["implementation_notes"]).strip() or "Codex generated the code bundle.",
        }

    def implement_code(state: EngineerState) -> dict[str, Any]:
        bundle = _generate_code_bundle(state)
        code_attempt = state["code_attempt"] + 1
        artifact_dir = Path(state["artifact_dir"])
        (artifact_dir / "gameplay_change.py").write_text(bundle["source_code"], encoding="utf-8")
        (artifact_dir / "test_gameplay_change.py").write_text(bundle["test_code"], encoding="utf-8")
        return {
            "code_attempt": code_attempt,
            "generated_code": bundle["source_code"],
            "generated_tests": bundle["test_code"],
            "implementation_notes": bundle["implementation_notes"],
        }

    def self_test(state: EngineerState) -> dict[str, Any]:
        compile_ok, tests_ok, test_output = _run_compile_and_tests(state["generated_code"], state["generated_tests"])
        artifact_dir = Path(state["artifact_dir"])
        (artifact_dir / "self_test.txt").write_text(test_output, encoding="utf-8")
        return {
            "compile_ok": compile_ok,
            "tests_ok": tests_ok,
            "test_output": test_output,
        }

    def test_gate(state: EngineerState) -> str:
        if state["compile_ok"] and state["tests_ok"]:
            return "prepare_delivery"
        return "repair_code"

    def repair_code(state: EngineerState) -> dict[str, Any]:
        bundle = _generate_code_bundle(state, error_context=state["test_output"])
        code_attempt = state["code_attempt"] + 1
        artifact_dir = Path(state["artifact_dir"])
        (artifact_dir / "gameplay_change.py").write_text(bundle["source_code"], encoding="utf-8")
        (artifact_dir / "test_gameplay_change.py").write_text(bundle["test_code"], encoding="utf-8")
        return {
            "code_attempt": code_attempt,
            "generated_code": bundle["source_code"],
            "generated_tests": bundle["test_code"],
            "implementation_notes": bundle["implementation_notes"],
        }

    def prepare_delivery(state: EngineerState) -> dict[str, Any]:
        artifact_dir = Path(state["artifact_dir"])
        commit_message = f"feat(gameplay): deliver {state['task_id']}"
        if state["task_type"] == "bugfix":
            commit_message = f"fix(gameplay): resolve {state['task_id']}"
        pr_title = f"[Gameplay] {state['task_prompt']}"
        pr_body = "\n".join(
            [
                "# Pull Request Draft",
                "",
                f"Workflow: {metadata.name}",
                f"Task: {state['task_prompt']}",
                f"Review score: {state['review_score']}",
                f"Task type rationale: {state['task_type_reason']}",
                f"Implementation notes: {state['implementation_notes']}",
                "",
                "## Validation",
                f"- {state['test_output']}",
            ]
        )
        (artifact_dir / "commit_message.txt").write_text(commit_message, encoding="utf-8")
        (artifact_dir / "pull_request.md").write_text(pr_body, encoding="utf-8")

        final_report = {
            "task_type": state["task_type"],
            "review_rounds": state["review_round"],
            "review_score": state["review_score"],
            "compile_ok": state["compile_ok"],
            "tests_ok": state["tests_ok"],
            "commit_message": commit_message,
            "pr_title": pr_title,
            "artifact_dir": str(artifact_dir),
            "llm_profile": context.llm_manager.describe(metadata.llm_profile),
            "codegen_profile": context.llm_manager.describe("codegen"),
        }
        summary = (
            f"{metadata.name} completed {state['task_type']} flow in "
            f"{state['review_round']} review round(s) and {state['code_attempt']} code attempt(s)."
        )
        return {"final_report": final_report, "summary": summary}

    graph = StateGraph(EngineerState)
    graph.add_node(
        "classify_request",
        trace_graph_node(graph_name=graph_name, node_name="classify_request", node_fn=classify_request),
    )
    graph.add_node(
        "prepare_doc_search",
        trace_graph_node(graph_name=graph_name, node_name="prepare_doc_search", node_fn=prepare_doc_search),
    )
    graph.add_node(
        "tool_request_error",
        trace_graph_node(graph_name=graph_name, node_name="tool_request_error", node_fn=tool_request_error),
    )
    graph.add_node(
        "capture_doc_hits",
        trace_graph_node(graph_name=graph_name, node_name="capture_doc_hits", node_fn=capture_doc_hits),
    )
    graph.add_node(
        "prepare_doc_context_lookup",
        trace_graph_node(
            graph_name=graph_name,
            node_name="prepare_doc_context_lookup",
            node_fn=prepare_doc_context_lookup,
        ),
    )
    graph.add_node(
        "capture_doc_context",
        trace_graph_node(graph_name=graph_name, node_name="capture_doc_context", node_fn=capture_doc_context),
    )
    graph.add_node(
        "build_design_doc",
        trace_graph_node(graph_name=graph_name, node_name="build_design_doc", node_fn=build_design_doc),
    )
    graph.add_node(
        "plan_work",
        trace_graph_node(graph_name=graph_name, node_name="plan_work", node_fn=plan_work),
    )
    graph.add_node(
        "request_review",
        trace_graph_node(graph_name=graph_name, node_name="request_review", node_fn=request_review),
    )
    graph.add_node(
        "enter_review_subgraph",
        trace_graph_node(graph_name=graph_name, node_name="enter_review_subgraph", node_fn=enter_review_subgraph),
    )
    graph.add_node("gameplay-reviewer-workflow", reviewer_graph)
    graph.add_node(
        "capture_review_result",
        trace_graph_node(graph_name=graph_name, node_name="capture_review_result", node_fn=capture_review_result),
    )
    graph.add_node(
        "revise_plan",
        trace_graph_node(graph_name=graph_name, node_name="revise_plan", node_fn=revise_plan),
    )
    graph.add_node(
        "implement_code",
        trace_graph_node(graph_name=graph_name, node_name="implement_code", node_fn=implement_code),
    )
    graph.add_node(
        "self_test",
        trace_graph_node(graph_name=graph_name, node_name="self_test", node_fn=self_test),
    )
    graph.add_node(
        "repair_code",
        trace_graph_node(graph_name=graph_name, node_name="repair_code", node_fn=repair_code),
    )
    graph.add_node(
        "prepare_delivery",
        trace_graph_node(graph_name=graph_name, node_name="prepare_delivery", node_fn=prepare_delivery),
    )
    for tool_name, tool_subgraph in tool_subgraphs.items():
        graph.add_node(tool_name, tool_subgraph)

    graph.add_edge(START, "classify_request")
    graph.add_edge("classify_request", "prepare_doc_search")
    graph.add_conditional_edges(
        "prepare_doc_search",
        trace_route_decision(graph_name=graph_name, router_name="route_tool_request", route_fn=route_tool_request),
        {
            **{tool_name: tool_name for tool_name in tool_subgraphs},
            "tool_request_error": "tool_request_error",
        },
    )
    graph.add_edge("tool_request_error", END)
    graph.add_edge("find-gameplay-docs", "capture_doc_hits")
    graph.add_edge("capture_doc_hits", "prepare_doc_context_lookup")
    graph.add_conditional_edges(
        "prepare_doc_context_lookup",
        trace_route_decision(graph_name=graph_name, router_name="route_tool_request", route_fn=route_tool_request),
        {
            **{tool_name: tool_name for tool_name in tool_subgraphs},
            "tool_request_error": "tool_request_error",
        },
    )
    graph.add_edge("load-markdown-context", "capture_doc_context")
    graph.add_edge("capture_doc_context", "build_design_doc")
    graph.add_edge("build_design_doc", "plan_work")
    graph.add_edge("plan_work", "request_review")
    graph.add_edge("request_review", "enter_review_subgraph")
    graph.add_edge("enter_review_subgraph", "gameplay-reviewer-workflow")
    graph.add_edge("gameplay-reviewer-workflow", "capture_review_result")
    graph.add_conditional_edges(
        "capture_review_result",
        trace_route_decision(graph_name=graph_name, router_name="review_gate", route_fn=review_gate),
        {
            "revise_plan": "revise_plan",
            "implement_code": "implement_code",
        },
    )
    graph.add_edge("revise_plan", "request_review")
    graph.add_edge("implement_code", "self_test")
    graph.add_conditional_edges(
        "self_test",
        trace_route_decision(graph_name=graph_name, router_name="test_gate", route_fn=test_gate),
        {
            "repair_code": "repair_code",
            "prepare_delivery": "prepare_delivery",
        },
    )
    graph.add_edge("repair_code", "self_test")
    graph.add_edge("prepare_delivery", END)
    return graph
