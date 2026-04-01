from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .profile import ReviewCriterion, ReviewProfile

# Type alias matching the existing CriterionAssessment TypedDict shape
CriterionDict = dict[str, Any]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in items:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _extract_heading_block(document: str, heading: str) -> list[str]:
    lines = document.splitlines()
    active = False
    collected: list[str] = []
    for line in lines:
        if line.strip() == f"## {heading}":
            active = True
            continue
        if active and line.startswith("## "):
            break
        if active and line.strip():
            collected.append(line.rstrip())
    return collected


def parse_review_json(
    data: dict[str, Any],
    criteria: tuple[ReviewCriterion, ...],
    approval_threshold: int,
    *,
    filter_fn: Any = None,
) -> dict[str, Any]:
    """Parse structured JSON output from the reviewer LLM.

    Handles both schema shapes:
    - criterion_scores (template/optimization): [{criterion, score, max_score, rationale}]
    - section_reviews (gameplay): [{section, score, status, rationale, action_items}]
    """
    weight_map = {c.name: c.weight for c in criteria}

    # Detect schema shape
    raw_items = data.get("criterion_scores") or data.get("section_reviews") or []
    name_key = "criterion" if "criterion_scores" in data else "section"

    parsed: list[CriterionDict] = []
    seen: set[str] = set()
    for item in raw_items:
        name = str(item.get(name_key, item.get("criterion", item.get("section", "")))).strip()
        if name not in weight_map or name in seen:
            continue
        seen.add(name)
        max_score = weight_map[name]
        score = max(0, min(int(item.get("score", 0)), max_score))
        raw_status = str(item.get("status", "")).strip().lower()
        if raw_status == "missing" or score == 0:
            status = "missing"
        elif (raw_status in {"pass", "approved"}) and score >= max_score:
            status = "pass"
        elif score >= max_score:
            status = "pass"
        else:
            status = "needs-work"
        parsed.append({
            "criterion": name,
            "score": score,
            "max_score": max_score,
            "status": status,
            "rationale": str(item.get("rationale", "")).strip(),
            "action_items": (
                []
                if status == "pass"
                else _dedupe([str(a).strip() for a in item.get("action_items", []) if str(a).strip()])
            ),
        })

    # Fill in missing criteria with zero scores
    for c in criteria:
        if c.name not in seen:
            parsed.append({
                "criterion": c.name,
                "score": 0,
                "max_score": c.weight,
                "status": "missing",
                "rationale": "Criterion not assessed by reviewer.",
                "action_items": [],
            })

    # Order by profile criteria order
    ordered = [{**next(item for item in parsed if item["criterion"] == c.name)} for c in criteria]

    # Parse blocking issues and improvement actions, filtering process-only items
    raw_blocking = [
        str(item).strip() for item in data.get("blocking_issues", [])
        if str(item).strip() and not str(item).strip().lower().startswith("none")
    ]
    raw_improvements = [
        str(item).strip() for item in data.get("improvement_actions", [])
        if str(item).strip() and not str(item).strip().lower().startswith("no further")
    ]
    if filter_fn:
        blocking = _dedupe([item for item in raw_blocking if not filter_fn(item)])
        improvements = _dedupe([item for item in raw_improvements if not filter_fn(item)])
    else:
        blocking = _dedupe(raw_blocking)
        improvements = _dedupe(raw_improvements)

    score = sum(item["score"] for item in ordered)
    decision = str(data.get("decision", "REVISE")).strip().upper()
    explicit_approval = decision == "APPROVE" or bool(data.get("approved", False))
    approved = explicit_approval or (score >= approval_threshold and not blocking)

    # Extract missing sections from LLM output if present (gameplay shape)
    missing_sections = _dedupe([
        str(item).strip() for item in data.get("missing_sections", [])
        if str(item).strip()
    ])

    return {
        "score": score,
        "approved": approved,
        "criterion_scores": ordered,
        "blocking_issues": blocking,
        "improvement_actions": improvements,
        "missing_sections": missing_sections,
        "senior_notes": str(data.get("senior_notes", data.get("feedback", ""))).strip(),
    }


def parse_review_markdown(
    review_doc: str,
    criteria: tuple[ReviewCriterion, ...],
    approval_threshold: int,
    *,
    filter_fn: Any = None,
) -> dict[str, Any]:
    """Parse markdown-formatted reviewer output (fallback when JSON fails)."""
    decision_match = re.search(r"Decision:\s*(APPROVE|REVISE)", review_doc, flags=re.IGNORECASE)
    score_match = re.search(r"Overall Score:\s*(\d{1,3})\s*/\s*100", review_doc, flags=re.IGNORECASE)
    if decision_match is None or score_match is None:
        raise ValueError("Reviewer LLM response is missing the required Decision or Overall Score fields.")

    weight_map = {c.name: c.weight for c in criteria}
    parsed: list[CriterionDict] = []
    seen: set[str] = set()
    for line in _extract_heading_block(review_doc, "Criterion Scores"):
        match = re.match(
            r"-\s*(?P<criterion>[^:]+):\s*(?P<score>\d{1,3})\s*/\s*(?P<max>\d{1,3})\s*-\s*(?P<rationale>.+)",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        name = match.group("criterion").strip()
        if name not in weight_map or name in seen:
            continue
        seen.add(name)
        max_score = weight_map[name]
        item_score = max(0, min(int(match.group("score")), max_score))
        status = "pass" if item_score >= max_score else "needs-work"
        if item_score == 0:
            status = "missing"
        parsed.append({
            "criterion": name,
            "score": item_score,
            "max_score": max_score,
            "status": status,
            "rationale": match.group("rationale").strip(),
            "action_items": [],
        })

    if len(parsed) != len(criteria):
        raise ValueError("Reviewer LLM did not return the full criterion assessment set.")

    ordered = [{**next(item for item in parsed if item["criterion"] == c.name)} for c in criteria]

    # Parse blocking issues
    raw_blocking: list[str] = []
    for line in _extract_heading_block(review_doc, "Blocking Issues"):
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if not item.lower().startswith("none."):
                raw_blocking.append(item)
    if filter_fn:
        blocking = _dedupe([item for item in raw_blocking if not filter_fn(item)])
    else:
        blocking = _dedupe(raw_blocking)

    # Parse improvement actions
    raw_improvements: list[str] = []
    for line in _extract_heading_block(review_doc, "Improvement Checklist"):
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        item = re.sub(r"^-\s*\[[ xX]\]\s*", "", stripped).strip()
        if not item.lower().startswith("no further"):
            raw_improvements.append(item)
    if filter_fn:
        improvements = _dedupe([item for item in raw_improvements if not filter_fn(item)])
    else:
        improvements = _dedupe(raw_improvements)

    score = sum(item["score"] for item in ordered)
    explicit_approval = decision_match.group(1).strip().lower() == "approve"
    approved = explicit_approval or (score >= approval_threshold and not blocking)

    senior_notes = "\n".join(_extract_heading_block(review_doc, "Senior Engineer Notes")).strip()

    return {
        "score": score,
        "approved": approved,
        "criterion_scores": ordered,
        "blocking_issues": blocking,
        "improvement_actions": improvements,
        "missing_sections": [],
        "senior_notes": senior_notes,
    }


def build_json_schema(criteria: tuple[ReviewCriterion, ...], *, gameplay_shape: bool = False) -> dict[str, Any]:
    """Build the JSON schema for the reviewer LLM output."""
    if gameplay_shape:
        return {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "feedback": {"type": "string"},
                "missing_sections": {"type": "array", "items": {"type": "string"}},
                "section_reviews": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section": {"type": "string"},
                            "score": {"type": "integer"},
                            "status": {"type": "string"},
                            "rationale": {"type": "string"},
                            "action_items": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["section", "score", "status", "rationale", "action_items"],
                        "additionalProperties": False,
                    },
                },
                "blocking_issues": {"type": "array", "items": {"type": "string"}},
                "improvement_actions": {"type": "array", "items": {"type": "string"}},
                "approved": {"type": "boolean"},
            },
            "required": [
                "score", "feedback", "missing_sections", "section_reviews",
                "blocking_issues", "improvement_actions", "approved",
            ],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["APPROVE", "REVISE"]},
            "overall_score": {"type": "integer"},
            "criterion_scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion": {"type": "string"},
                        "score": {"type": "integer"},
                        "max_score": {"type": "integer"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["criterion", "score", "max_score", "rationale"],
                    "additionalProperties": False,
                },
            },
            "blocking_issues": {"type": "array", "items": {"type": "string"}},
            "improvement_actions": {"type": "array", "items": {"type": "string"}},
            "senior_notes": {"type": "string"},
        },
        "required": [
            "decision", "overall_score", "criterion_scores",
            "blocking_issues", "improvement_actions", "senior_notes",
        ],
        "additionalProperties": False,
    }
