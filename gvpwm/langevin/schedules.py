from __future__ import annotations


def exponential_schedule(
    start: float,
    end: float,
    step: int,
    total_steps: int,
) -> float:
    """Exponentially interpolate from start to end, including both endpoints."""

    if total_steps <= 1:
        return end
    if not 0 <= step < total_steps:
        raise ValueError("step must satisfy 0 <= step < total_steps")

    progress = step / (total_steps - 1)
    if start <= 0.0 or end <= 0.0:
        return linear_schedule(start, end, step, total_steps)
    return start * (end / start) ** progress


def linear_schedule(
    start: float,
    end: float,
    step: int,
    total_steps: int,
) -> float:
    """Linearly interpolate from start to end, including both endpoints."""

    if total_steps <= 1:
        return end
    if not 0 <= step < total_steps:
        raise ValueError("step must satisfy 0 <= step < total_steps")

    progress = step / (total_steps - 1)
    return start + progress * (end - start)
