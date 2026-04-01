from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from core.llm import LLMError
from core.models import WorkflowContext, WorkflowMetadata
from core.quality_loop import QualityLoopSpec, evaluate_quality_loop
from core.scoring import ScoreAssessment, ScorePolicy, evaluate_score_decision
from core.text_utils import normalize_text, slugify

from .parsing import _dedupe, _extract_heading_block, build_json_schema, parse_review_json, parse_review_markdown
from .process_filter import is_process_only_feedback
from .profile import ReviewCriterion, ReviewProfile
from .prompt_builder import build_markdown_fallback_instructions, build_review_input, build_review_instructions
from .state import apply_field_aliases


class ReviewEngine:
    """Unified review engine. One implementation for all review types.

    Each review type provides a ``ReviewProfile`` that configures criteria,
    thresholds, prompts, and domain-specific extensions. The engine handles
    the entire review flow: LLM call, parsing, filtering, scoring, loop
    evaluation, artifact writing, and state output.
    """

    def __init__(
        self,
        profile: ReviewProfile,
        context: WorkflowContext,
        metadata: WorkflowMetadata,
    ) -> None:
        self._profile = profile
        self._context = context
        self._metadata = metadata

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #

    def review(self, state: dict[str, Any]) -> dict[str, Any]:
        """Main review node function. Pass to ``StateGraph.add_node()``."""
        p = self._profile
        review_round = int(state.get("review_round", 0)) + 1
        artifact_path = self._artifact_dir(state)

        # Check LLM availability
        reviewer_llm = self._context.get_llm("reviewer")
        if not reviewer_llm.is_enabled():
            return self._blocked_response(
                artifact_path=artifact_path,
                review_round=review_round,
                reason=f"Reviewer LLM is unavailable, so {p.domain_noun} assessments cannot be generated.",
                loop_status="llm-unavailable",
            )

        # Resolve criteria (may be dynamic for optimization)
        criteria = self._resolve_criteria(state)
        optimization_domain = self._resolve_domain(state)

        # Build LLM instructions and input
        instructions = build_review_instructions(
            p, criteria, review_round, optimization_domain=optimization_domain,
        )
        extra_context = self._build_extra_context(state)
        input_text = build_review_input(
            p,
            task_prompt=state["task_prompt"],
            doc_text=state[p.doc_field_name],
            review_round=review_round,
            optimization_domain=optimization_domain,
            extra_context_lines=extra_context,
        )

        # Call LLM (JSON first, markdown fallback if supported)
        schema = build_json_schema(criteria, gameplay_shape=not p.supports_markdown_fallback)
        try:
            review_result = self._call_llm(
                reviewer_llm, instructions, input_text, schema, criteria,
            )
        except (LLMError, ValueError) as exc:
            return self._blocked_response(
                artifact_path=artifact_path,
                review_round=review_round,
                reason=f"Reviewer LLM failed to produce usable {p.domain_noun} assessments: {exc}",
                loop_status="llm-error",
            )

        # Apply hard blocker guardrails (gameplay: dismiss prefixed blockers when all pass)
        review_result = self._apply_hard_blocker_guardrails(review_result)

        # Enforce minimum rounds
        if review_round < p.min_rounds:
            review_result = self._enforce_min_rounds(review_result, review_round)

        # Detect missing sections (only if profile requires it)
        missing_sections = self._detect_missing_sections(state[p.doc_field_name], criteria)
        # Merge with LLM-reported missing sections
        llm_missing = review_result.get("missing_sections", [])
        if llm_missing:
            combined = list(llm_missing)
            for s in missing_sections:
                if s not in combined:
                    combined.append(s)
            missing_sections = combined

        # Score decision
        score_policy = ScorePolicy(
            system_id=p.system_id,
            threshold=p.approval_threshold,
            require_blocker_free=True,
            require_missing_section_free=p.require_missing_section_free,
            require_explicit_approval=True,
        )
        score_decision = evaluate_score_decision(
            score_policy,
            round_index=review_round,
            assessments=self._to_score_assessments(review_result["criterion_scores"]),
            explicit_approval=bool(review_result["approved"]),
            blocking_issues=review_result["blocking_issues"],
            improvement_actions=review_result["improvement_actions"],
            missing_sections=missing_sections,
            artifact_dir=artifact_path,
        )

        # Quality loop evaluation
        loop_spec = QualityLoopSpec(
            loop_id=p.loop_id,
            threshold=p.approval_threshold,
            max_rounds=p.max_rounds,
            min_rounds=p.min_rounds,
            require_blocker_free=True,
            require_missing_section_free=p.require_missing_section_free,
            require_explicit_approval=True,
            min_score_delta=1,
            stagnation_limit=p.stagnation_limit,
        )
        previous_score = int(state.get("review_score", state.get("score", 0))) if review_round > 1 else None
        prior_stagnated = int(state.get("loop_stagnated_rounds", 0)) if review_round > 1 else 0
        progress = evaluate_quality_loop(
            loop_spec,
            round_index=review_round,
            score=score_decision.score,
            approved=score_decision.approved,
            blocking_issues=score_decision.blocking_issues,
            missing_sections=score_decision.missing_sections,
            improvement_actions=score_decision.improvement_actions,
            previous_score=previous_score,
            prior_stagnated_rounds=prior_stagnated,
        )

        # Compose final review document
        senior_notes = review_result.get("senior_notes", "")
        final_doc = self._compose_review_doc(
            score=progress.score,
            approved=progress.approved,
            criterion_scores=review_result["criterion_scores"],
            blocking_issues=list(score_decision.blocking_issues),
            improvement_actions=list(score_decision.improvement_actions),
            senior_notes=senior_notes,
            confidence=score_decision.confidence,
            confidence_label=score_decision.confidence_label,
            confidence_reason=score_decision.confidence_reason,
            review_round=review_round,
        )
        (artifact_path / f"review_round_{review_round}.md").write_text(final_doc, encoding="utf-8")

        # Build summary
        summary = self._build_summary(review_round, progress.score, progress.approved, progress.completed, progress.status)

        # Build canonical output
        output: dict[str, Any] = {
            "review_round": review_round,
            "artifact_dir": str(artifact_path),
            "review_doc": final_doc,
            "review_score": progress.score,
            "review_feedback": final_doc,
            "review_blocking_issues": list(score_decision.blocking_issues),
            "review_improvement_actions": list(score_decision.improvement_actions),
            "review_missing_sections": list(progress.missing_sections),
            "review_criterion_scores": list(review_result["criterion_scores"]),
            "review_approved": progress.approved,
            "loop_status": progress.status,
            "loop_reason": progress.reason,
            "loop_should_continue": progress.should_continue,
            "loop_completed": progress.completed,
            "loop_stagnated_rounds": progress.stagnated_rounds,
            "review_score_confidence": score_decision.confidence,
            "review_score_confidence_label": score_decision.confidence_label,
            "review_score_confidence_reason": score_decision.confidence_reason,
            "summary": summary,
        }
        return apply_field_aliases(output, p.state_field_aliases)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _resolve_criteria(self, state: dict[str, Any]) -> tuple[ReviewCriterion, ...]:
        p = self._profile
        if p.dynamic_criteria_field:
            override = state.get(p.dynamic_criteria_field)
            if override:
                return tuple(
                    ReviewCriterion(
                        name=c[0],
                        weight=c[1],
                        expected_sections=tuple(c[2:]) if len(c) > 2 else (),
                    )
                    for c in override
                )
        return p.criteria

    def _resolve_domain(self, state: dict[str, Any]) -> str | None:
        p = self._profile
        if p.dynamic_domain_field:
            return str(state.get(p.dynamic_domain_field, "optimization"))
        return None

    def _build_extra_context(self, state: dict[str, Any]) -> list[str] | None:
        """Build extra context lines from state for gameplay-specific fields."""
        p = self._profile
        if not p.state_field_aliases:
            return None
        # Gameplay needs task_type and execution_track in context
        lines: list[str] = []
        task_type = state.get("task_type")
        if task_type:
            lines.append(f"- Task type: {task_type}")
        execution_track = state.get("execution_track")
        if execution_track:
            lines.append(f"- Execution track: {execution_track}")
        return lines if lines else None

    def _filter_fn(self, text: str) -> bool:
        return is_process_only_feedback(
            text,
            extra_keywords=self._profile.extra_process_keywords,
            extra_patterns=self._profile.extra_process_patterns,
        )

    def _call_llm(
        self,
        reviewer_llm: Any,
        instructions: str,
        input_text: str,
        schema: dict[str, Any],
        criteria: tuple[ReviewCriterion, ...],
    ) -> dict[str, Any]:
        p = self._profile
        try:
            data = reviewer_llm.generate_json(
                instructions=instructions,
                input_text=input_text,
                schema_name=p.schema_name,
                schema=schema,
            )
            # Validate section count for gameplay shape
            if not p.supports_markdown_fallback:
                raw_items = data.get("section_reviews", [])
                expected = {c.name for c in criteria}
                received = {str(item.get("section", "")).strip() for item in raw_items}
                if len(raw_items) != len(criteria) or received != expected:
                    raise ValueError("Reviewer LLM did not return the full assessment set.")
            return parse_review_json(data, criteria, p.approval_threshold, filter_fn=self._filter_fn)
        except (LLMError, ValueError, KeyError, TypeError, AttributeError):
            if not p.supports_markdown_fallback:
                raise
            # Fallback to markdown
            fallback_instructions = instructions + build_markdown_fallback_instructions(p)
            generated = reviewer_llm.generate_text(
                instructions=fallback_instructions,
                input_text=input_text,
            )
            return parse_review_markdown(generated, criteria, p.approval_threshold, filter_fn=self._filter_fn)

    def _apply_hard_blocker_guardrails(self, review_result: dict[str, Any]) -> dict[str, Any]:
        """If all criteria pass and profile has hard_blockers, dismiss prefixed blockers."""
        p = self._profile
        if not p.hard_blockers:
            return review_result
        criterion_scores = review_result["criterion_scores"]
        if not all(item["status"] == "pass" for item in criterion_scores):
            return review_result
        prefixes = tuple(hb.label for hb in p.hard_blockers)
        filtered_blockers = _dedupe([
            item for item in review_result["blocking_issues"]
            if not any(item.startswith(prefix) for prefix in prefixes)
        ])
        return {**review_result, "blocking_issues": filtered_blockers}

    def _enforce_min_rounds(self, review_result: dict[str, Any], review_round: int) -> dict[str, Any]:
        p = self._profile
        mandatory = p.mandatory_action
        if callable(mandatory):
            action_text = mandatory(review_result["improvement_actions"], review_result["blocking_issues"])
        else:
            action_text = mandatory
        if not action_text:
            action_text = (
                f"Run one more {p.domain_noun} pass that independently re-validates "
                "the findings with fresh source code evidence before final handoff."
            )
        enforced_actions = _dedupe([*review_result["improvement_actions"], action_text])
        enforced_notes = "\n".join(
            item for item in [
                review_result.get("senior_notes", ""),
                f"Minimum verification depth is {p.min_rounds} rounds, so this brief still needs one more independent pass.",
            ] if item
        ).strip()
        return {
            **review_result,
            "approved": False,
            "improvement_actions": enforced_actions,
            "senior_notes": enforced_notes,
        }

    def _detect_missing_sections(
        self, doc: str, criteria: tuple[ReviewCriterion, ...],
    ) -> list[str]:
        if not self._profile.require_missing_section_free:
            return []
        expected = tuple(s for c in criteria for s in c.expected_sections)
        if not expected:
            return []
        doc_lower = doc.lower()
        missing: list[str] = []
        for section in expected:
            if f"## {section.lower()}" not in doc_lower:
                missing.append(section)
                continue
            block = _extract_heading_block(doc, section)
            content = "\n".join(block).strip()
            if len(content) < 50:
                missing.append(section)
        return missing

    def _compose_review_doc(
        self,
        *,
        score: int,
        approved: bool,
        criterion_scores: list[dict[str, Any]],
        blocking_issues: list[str],
        improvement_actions: list[str],
        senior_notes: str,
        confidence: float | None = None,
        confidence_label: str = "unmeasured",
        confidence_reason: str = "",
        review_round: int = 0,
    ) -> str:
        p = self._profile
        lines = [
            f"# {p.review_doc_title}",
            "",
            f"Decision: {'APPROVE' if approved else 'REVISE'}",
            f"Overall Score: {score}/100",
            f"Scoring Confidence: {confidence_label}{f' ({confidence:.1f}x noise floor)' if confidence is not None else ''}",
            f"Confidence Reason: {confidence_reason}",
            "",
            "## Criterion Scores",
        ]
        for item in criterion_scores:
            lines.append(f"- {item['criterion']}: {item['score']}/{item['max_score']} - {item['rationale']}")
        lines.extend(["", "## Blocking Issues"])
        lines.extend([f"- {item}" for item in blocking_issues] or ["- None."])
        lines.extend(["", "## Improvement Checklist"])
        lines.extend(
            [f"- [ ] {item}" for item in improvement_actions]
            or [f"- [x] {p.no_action_text}"]
        )
        lines.extend([
            "",
            f"## {p.notes_heading}",
            senior_notes.strip() or f"The {p.domain_noun} is ready for handoff.",
        ])
        return "\n".join(lines)

    def _blocked_response(
        self,
        *,
        artifact_path: Path,
        review_round: int,
        reason: str,
        loop_status: str,
    ) -> dict[str, Any]:
        p = self._profile
        improvement_action = (
            f"Enable the reviewer LLM and rerun {p.domain_noun} review so the "
            "scoring assessments come from LLM output."
        )
        review_doc = self._compose_review_doc(
            score=0,
            approved=False,
            criterion_scores=[],
            blocking_issues=[reason],
            improvement_actions=[improvement_action],
            senior_notes=reason,
            confidence_label="unmeasured",
            confidence_reason="MAD confidence is unavailable because no LLM-generated assessments were produced.",
        )
        (artifact_path / f"review_round_{review_round}.md").write_text(review_doc, encoding="utf-8")
        output: dict[str, Any] = {
            "review_round": review_round,
            "artifact_dir": str(artifact_path),
            "review_doc": review_doc,
            "review_score": 0,
            "review_feedback": review_doc,
            "review_blocking_issues": [reason],
            "review_improvement_actions": [improvement_action],
            "review_missing_sections": [],
            "review_criterion_scores": [],
            "review_approved": False,
            "loop_status": loop_status,
            "loop_reason": reason,
            "loop_should_continue": False,
            "loop_completed": True,
            "loop_stagnated_rounds": 0,
            "review_score_confidence": None,
            "review_score_confidence_label": "unmeasured",
            "review_score_confidence_reason": "MAD confidence is unavailable because no LLM-generated assessments were produced.",
            "summary": f"{self._metadata.name} stopped in review round {review_round} because {p.domain_noun} assessments were unavailable.",
        }
        return apply_field_aliases(output, p.state_field_aliases)

    def _build_summary(
        self, review_round: int, score: int, approved: bool, completed: bool, status: str,
    ) -> str:
        name = self._metadata.name
        noun = self._profile.domain_noun
        if approved:
            return f"{name} approved {noun} review round {review_round} with score {score}/100."
        if completed:
            return f"{name} stopped after review round {review_round} with score {score}/100. Loop status: {status}."
        return f"{name} scored {score}/100 in review round {review_round} and requested another {noun} pass."

    def _artifact_dir(self, state: dict[str, Any]) -> Path:
        existing = str(state.get("artifact_dir", "")).strip()
        if existing:
            path = Path(existing)
            path.mkdir(parents=True, exist_ok=True)
            return path
        run_dir = str(state.get("run_dir", "")).strip()
        base_dir = Path(run_dir) if run_dir else Path(self._context.artifact_root) / "adhoc"
        task_id = str(state.get("task_id", "")).strip() or state["task_prompt"]
        digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:6]
        task_dir = f"{_short_slug(task_id, fallback='task')}-{digest}"
        path = base_dir / "tasks" / task_dir / self._metadata.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _to_score_assessments(criterion_scores: list[dict[str, Any]]) -> list[ScoreAssessment]:
        return [
            ScoreAssessment(
                label=item["criterion"],
                score=int(item["score"]),
                max_score=int(item["max_score"]),
                status=item["status"],
                rationale=item.get("rationale", ""),
                action_items=tuple(item.get("action_items", ())),
            )
            for item in criterion_scores
        ]


def _short_slug(value: str, *, fallback: str, max_length: int = 18) -> str:
    slug = slugify(value, fallback=fallback)
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(normalize_text(value).encode("utf-8")).hexdigest()[:6]
    keep = max(1, max_length - len(digest) - 1)
    return f"{slug[:keep].rstrip('-') or fallback}-{digest}"
