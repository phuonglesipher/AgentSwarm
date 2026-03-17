from __future__ import annotations

import unittest

from core.quality_loop import QualityLoopSpec, evaluate_quality_loop


class QualityLoopTests(unittest.TestCase):
    def test_loop_passes_when_threshold_and_blockers_are_clear(self) -> None:
        spec = QualityLoopSpec(loop_id="plan-review", threshold=90, max_rounds=3, stagnation_limit=2)

        progress = evaluate_quality_loop(
            spec,
            round_index=1,
            score=95,
            approved=True,
            missing_sections=[],
            blocking_issues=[],
            improvement_actions=["Tighten task type rationale."],
        )

        self.assertEqual(progress.status, "passed")
        self.assertTrue(progress.approved)
        self.assertFalse(progress.should_continue)
        self.assertTrue(progress.completed)
        self.assertEqual(progress.improvement_actions, ("Tighten task type rationale.",))

    def test_loop_retries_when_score_is_below_threshold(self) -> None:
        spec = QualityLoopSpec(loop_id="plan-review", threshold=90, max_rounds=3, stagnation_limit=2)

        progress = evaluate_quality_loop(
            spec,
            round_index=1,
            score=72,
            approved=False,
            blocking_issues=["Implementation Steps: add ordered gameplay changes."],
        )

        self.assertEqual(progress.status, "retry")
        self.assertTrue(progress.should_continue)
        self.assertFalse(progress.completed)
        self.assertIn("score 72/90", progress.reason)

    def test_loop_stops_at_max_rounds(self) -> None:
        spec = QualityLoopSpec(loop_id="plan-review", threshold=90, max_rounds=2, stagnation_limit=0)

        progress = evaluate_quality_loop(
            spec,
            round_index=2,
            score=88,
            approved=False,
            blocking_issues=["Unit Tests: add regression checks."],
        )

        self.assertEqual(progress.status, "max-rounds")
        self.assertFalse(progress.should_continue)
        self.assertTrue(progress.completed)

    def test_loop_stops_on_stagnation_after_low_progress_rounds(self) -> None:
        spec = QualityLoopSpec(
            loop_id="plan-review",
            threshold=90,
            max_rounds=4,
            min_score_delta=2,
            stagnation_limit=2,
        )

        second_round = evaluate_quality_loop(
            spec,
            round_index=2,
            score=81,
            approved=False,
            previous_score=80,
            prior_stagnated_rounds=0,
            blocking_issues=["Acceptance Criteria: add player-visible checks."],
        )
        third_round = evaluate_quality_loop(
            spec,
            round_index=3,
            score=82,
            approved=False,
            previous_score=81,
            prior_stagnated_rounds=second_round.stagnated_rounds,
            blocking_issues=["Acceptance Criteria: add player-visible checks."],
        )

        self.assertEqual(second_round.status, "retry")
        self.assertEqual(second_round.stagnated_rounds, 1)
        self.assertEqual(third_round.status, "stagnated")
        self.assertEqual(third_round.stagnated_rounds, 2)
        self.assertTrue(third_round.completed)
        self.assertFalse(third_round.should_continue)


if __name__ == "__main__":
    unittest.main()
