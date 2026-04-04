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

    def test_loop_requires_minimum_rounds_before_passing(self) -> None:
        spec = QualityLoopSpec(loop_id="plan-review", threshold=90, max_rounds=3, min_rounds=2)

        first_round = evaluate_quality_loop(
            spec,
            round_index=1,
            score=96,
            approved=True,
            blocking_issues=[],
            improvement_actions=[],
        )
        second_round = evaluate_quality_loop(
            spec,
            round_index=2,
            score=96,
            approved=True,
            blocking_issues=[],
            improvement_actions=[],
            previous_score=96,
            prior_stagnated_rounds=first_round.stagnated_rounds,
        )

        self.assertEqual(first_round.status, "retry")
        self.assertIn("needs at least 2 round(s)", first_round.reason)
        self.assertTrue(first_round.should_continue)
        self.assertFalse(first_round.completed)
        self.assertEqual(second_round.status, "passed")
        self.assertTrue(second_round.approved)
        self.assertFalse(second_round.should_continue)
        self.assertTrue(second_round.completed)


    def test_loop_does_not_stagnate_when_score_meets_threshold(self) -> None:
        spec = QualityLoopSpec(
            loop_id="plan-review",
            threshold=90,
            max_rounds=4,
            min_score_delta=1,
            stagnation_limit=2,
        )

        second_round = evaluate_quality_loop(
            spec,
            round_index=2,
            score=90,
            approved=True,
            previous_score=90,
            prior_stagnated_rounds=0,
            blocking_issues=[],
        )

        self.assertEqual(second_round.stagnated_rounds, 0)
        self.assertEqual(second_round.status, "passed")
        self.assertTrue(second_round.approved)

    def test_loop_still_stagnates_when_score_is_below_threshold(self) -> None:
        spec = QualityLoopSpec(
            loop_id="plan-review",
            threshold=90,
            max_rounds=4,
            min_score_delta=1,
            stagnation_limit=2,
        )

        second_round = evaluate_quality_loop(
            spec,
            round_index=2,
            score=70,
            approved=False,
            previous_score=70,
            prior_stagnated_rounds=0,
            blocking_issues=["Still needs work."],
        )
        third_round = evaluate_quality_loop(
            spec,
            round_index=3,
            score=70,
            approved=False,
            previous_score=70,
            prior_stagnated_rounds=second_round.stagnated_rounds,
            blocking_issues=["Still needs work."],
        )

        self.assertEqual(second_round.stagnated_rounds, 1)
        self.assertEqual(second_round.status, "retry")
        self.assertEqual(third_round.stagnated_rounds, 2)
        self.assertEqual(third_round.status, "stagnated")
        self.assertTrue(third_round.completed)


    def test_recurring_blockers_detected(self) -> None:
        spec = QualityLoopSpec(
            loop_id="plan-review",
            threshold=90,
            max_rounds=4,
            min_score_delta=5,
            stagnation_limit=2,
        )

        progress = evaluate_quality_loop(
            spec,
            round_index=2,
            score=75,
            approved=False,
            previous_score=70,
            prior_stagnated_rounds=0,
            blocking_issues=["Missing section X", "No test coverage"],
            previous_blocking_issues=["Missing section X", "Vague description"],
        )

        self.assertIn("Missing section X", progress.recurring_blockers)
        self.assertNotIn("No test coverage", progress.recurring_blockers)
        self.assertNotIn("Vague description", progress.recurring_blockers)

    def test_no_recurring_blockers_on_first_round(self) -> None:
        spec = QualityLoopSpec(loop_id="test", threshold=90, max_rounds=3)

        progress = evaluate_quality_loop(
            spec,
            round_index=1,
            score=70,
            approved=False,
            blocking_issues=["Missing section X"],
        )

        self.assertEqual(progress.recurring_blockers, ())

    def test_recurring_blockers_trigger_stagnation(self) -> None:
        """Same blockers in consecutive rounds should count as stagnation
        even if the score delta is above min_score_delta."""
        spec = QualityLoopSpec(
            loop_id="test",
            threshold=90,
            max_rounds=5,
            min_score_delta=1,
            stagnation_limit=2,
        )

        # Round 2: score improved (70→75, delta=5), but same single blocker
        round2 = evaluate_quality_loop(
            spec,
            round_index=2,
            score=75,
            approved=False,
            previous_score=70,
            prior_stagnated_rounds=0,
            blocking_issues=["Blocker A"],
            previous_blocking_issues=["Blocker A"],
        )
        # All current blockers are recurring → blocker-stuck → stagnation increments
        self.assertEqual(round2.stagnated_rounds, 1)
        self.assertEqual(round2.recurring_blockers, ("Blocker A",))

        # Round 3: still same blocker
        round3 = evaluate_quality_loop(
            spec,
            round_index=3,
            score=80,
            approved=False,
            previous_score=75,
            prior_stagnated_rounds=round2.stagnated_rounds,
            blocking_issues=["Blocker A"],
            previous_blocking_issues=["Blocker A"],
        )
        self.assertEqual(round3.stagnated_rounds, 2)
        self.assertEqual(round3.status, "stagnated")


if __name__ == "__main__":
    unittest.main()
