# Quality Loop Wiring

## Rules

- Always use `evaluate_quality_loop()` from `core/quality_loop.py` as the loop gate — never hand-code threshold checks in conditional edges
- Construct `QualityLoopSpec` with `loop_id` matching the profile's `system_id`
- Pass `previous_score` and `prior_stagnated_rounds` from state for delta and stagnation tracking
- Check `progress.should_continue` to decide the conditional edge (loop back vs proceed)
- `ReviewEngine.review()` calls `evaluate_quality_loop()` internally — do not double-call it from the workflow node
- Default guardrails: `threshold=90`, `min_rounds=2`, `stagnation_limit=2`

## Pattern

```python
spec = QualityLoopSpec(
    loop_id=profile.system_id,
    threshold=profile.approval_threshold,
    max_rounds=profile.max_rounds,
    min_rounds=profile.min_rounds,
    stagnation_limit=profile.stagnation_limit,
)
progress = evaluate_quality_loop(spec, round_index=..., score=..., approved=..., ...)
if progress.should_continue:
    # loop back to work node
```
