"""Small, dependency-light drift calculation for stored probabilities."""

from collections.abc import Sequence


def population_stability_index(
    baseline: Sequence[float], current: Sequence[float], bins: int = 10
) -> float:
    if not baseline or not current:
        raise ValueError("Both baseline and current populations are required")
    def proportions(values: Sequence[float]) -> list[float]:
        counts = [0] * bins
        for value in values:
            index = min(int(max(0.0, min(0.999999, value)) * bins), bins - 1)
            counts[index] += 1
        total = len(values)
        return [(count / total) + 1e-6 for count in counts]

    baseline_proportions = proportions(baseline)
    current_proportions = proportions(current)
    return sum(
        (current_value - baseline_value)
        * __import__("math").log(current_value / baseline_value)
        for baseline_value, current_value in zip(
            baseline_proportions, current_proportions, strict=True
        )
    )