# Scoring Engine

## Rules

- `evaluate_score_decision()` in `core/scoring/engine.py` is the single entry point for all scoring
- `ScorePolicy` configures threshold, blocker requirements, and MAD confidence settings
- `ScoreAssessment` tuples feed from ReviewEngine criterion parsing
- MAD (Median Absolute Deviation) confidence from `core/scoring/stats.py` detects convergence across rounds
- `confidence_override_score` (default 95) bypasses weak confidence when score is very high
- `artifact_dir` enables score history persistence across rounds (JSONL files)
- Never implement custom scoring logic; extend `ScorePolicy` fields instead
