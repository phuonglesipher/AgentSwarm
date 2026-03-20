from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.scoring import ScoreAssessment, ScorePolicy, evaluate_score_decision, mad_confidence


class ScoringStatsTests(unittest.TestCase):
    def test_mad_confidence_requires_minimum_samples(self) -> None:
        self.assertIsNone(mad_confidence([90, 95], baseline=90, current=95, min_samples=3))

    def test_mad_confidence_matches_expected_ratio(self) -> None:
        confidence = mad_confidence([70, 85, 90], baseline=70, current=90, min_samples=3)

        self.assertEqual(confidence, 4.0)


class ScoringEngineTests(unittest.TestCase):
    def test_engine_passes_before_confidence_is_measured(self) -> None:
        policy = ScorePolicy(
            system_id="review-score",
            threshold=90,
            require_blocker_free=True,
            require_missing_section_free=True,
            require_explicit_approval=True,
        )

        with tempfile.TemporaryDirectory(prefix="agentswarm-score-pass-") as temp_dir:
            decision = evaluate_score_decision(
                policy,
                round_index=1,
                assessments=[
                    ScoreAssessment(
                        label="Overall",
                        score=95,
                        max_score=100,
                        status="pass",
                        rationale="The artifact is technically ready.",
                    )
                ],
                explicit_approval=True,
                artifact_dir=Path(temp_dir),
            )

        self.assertTrue(decision.approved)
        self.assertEqual(decision.score, 95)
        self.assertIsNone(decision.confidence)
        self.assertEqual(decision.confidence_label, "unmeasured")

    def test_engine_blocks_low_confidence_when_noise_is_high(self) -> None:
        policy = ScorePolicy(
            system_id="review-score",
            threshold=90,
            require_blocker_free=True,
            require_missing_section_free=False,
            require_explicit_approval=True,
        )

        with tempfile.TemporaryDirectory(prefix="agentswarm-score-confidence-") as temp_dir:
            artifact_dir = Path(temp_dir)
            final_decision = None
            for round_index, score in enumerate((90, 100, 80, 91), start=1):
                final_decision = evaluate_score_decision(
                    policy,
                    round_index=round_index,
                    assessments=[
                        ScoreAssessment(
                            label="Overall",
                            score=score,
                            max_score=100,
                            status="pass",
                            rationale="Round-level aggregate score.",
                        )
                    ],
                    explicit_approval=True,
                    artifact_dir=artifact_dir,
                )

        self.assertIsNotNone(final_decision)
        assert final_decision is not None
        self.assertFalse(final_decision.approved)
        self.assertLess(final_decision.confidence or 0.0, 1.0)
        self.assertEqual(final_decision.confidence_label, "weak")
        self.assertTrue(
            any(item.startswith("Scoring Confidence:") for item in final_decision.blocking_issues),
        )
        self.assertEqual(len(final_decision.history), 4)


if __name__ == "__main__":
    unittest.main()
