from __future__ import annotations

from collections.abc import Iterable


def median(values: Iterable[float]) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2.0
    return ordered[middle]


def median_absolute_deviation(values: Iterable[float]) -> float | None:
    ordered = [float(value) for value in values]
    midpoint = median(ordered)
    if midpoint is None:
        return None
    deviations = [abs(value - midpoint) for value in ordered]
    return median(deviations)


def mad_confidence(
    values: Iterable[float],
    *,
    baseline: float | None = None,
    current: float | None = None,
    min_samples: int = 3,
) -> float | None:
    observations = [float(value) for value in values]
    if len(observations) < max(1, int(min_samples)):
        return None

    mad = median_absolute_deviation(observations)
    if mad is None:
        return None
    if mad == 0:
        return float('inf')

    baseline_value = float(observations[0] if baseline is None else baseline)
    current_value = float(observations[-1] if current is None else current)
    return abs(current_value - baseline_value) / mad
