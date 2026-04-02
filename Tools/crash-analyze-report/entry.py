from __future__ import annotations

from langchain_core.tools import tool

from core.models import ToolContext, ToolMetadata


REPORT_TEMPLATE = """\
# Crash Analysis Report

## Crash Summary
{crash_summary}

## Root Cause
{root_cause}

## Domain Classification
{domain}

## Recommended Workflow
{recommended_workflow}

## Reproduction Steps
{reproduction}

## Fix Guidance
{fix_guidance}

## Affected Files
{affected_files}

## Verification Criteria
{verification}
"""

DOMAIN_TO_WORKFLOW = {
    "gameplay": "gameplay-engineer-workflow",
    "graphics": "graphics-engineer-workflow",
    "engine": "engine-engineer-workflow",
    "platform": "engine-engineer-workflow",
    "memory": "engine-engineer-workflow",
}


def build_tool(context: ToolContext, metadata: ToolMetadata):
    @tool(metadata.qualified_name, response_format="content_and_artifact")
    def crash_analyze_report(
        crash_summary: str,
        root_cause: str,
        domain: str,
        reproduction: str = "Not determined.",
        fix_guidance: str = "Not determined.",
        affected_files: str = "Not determined.",
        verification: str = "Not determined.",
    ) -> tuple[str, dict[str, str]]:
        """Generate a structured crash analysis report for handoff.

        Takes investigation findings and formats them into a standardized
        report that downstream engineer workflows can parse and act on
        immediately.

        Args:
            crash_summary: One-line crash description with type and severity.
            root_cause: Root cause with file:line references.
            domain: One of: gameplay, graphics, engine, platform, memory.
            reproduction: Steps to reproduce the crash.
            fix_guidance: Concrete fix recommendations.
            affected_files: File paths that need modification.
            verification: How to verify the fix works.
        """
        domain_lower = domain.strip().lower()
        recommended = DOMAIN_TO_WORKFLOW.get(domain_lower, "template-investigation-workflow")

        report = REPORT_TEMPLATE.format(
            crash_summary=crash_summary.strip(),
            root_cause=root_cause.strip(),
            domain=domain.strip(),
            recommended_workflow=recommended,
            reproduction=reproduction.strip(),
            fix_guidance=fix_guidance.strip(),
            affected_files=affected_files.strip(),
            verification=verification.strip(),
        )

        return (
            f"Crash analysis report generated. Domain: {domain_lower}. "
            f"Recommended workflow: {recommended}.",
            {"report": report, "domain": domain_lower, "recommended_workflow": recommended},
        )

    return crash_analyze_report
