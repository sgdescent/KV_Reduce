"""Shared proposal-weighted statistics for speculative-decoding experiments."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple


def acceptance_ratio(rows: Sequence[Dict[str, str]]) -> float:
    proposed = sum(float(row["proposed_tokens"]) for row in rows)
    accepted = sum(float(row["accepted_tokens"]) for row in rows)
    return accepted / proposed if proposed > 0 else 0.0


def acceptance_contrast(
    prompt_rows: Sequence[Tuple[Dict[str, str], ...]],
    coefficients: Sequence[float],
) -> float:
    if not prompt_rows:
        return float("nan")
    if any(len(rows) != len(coefficients) for rows in prompt_rows):
        raise ValueError("Every prompt tuple must match the number of contrast coefficients.")
    columns = list(zip(*prompt_rows))
    return sum(
        float(coefficient) * acceptance_ratio(list(rows))
        for coefficient, rows in zip(coefficients, columns)
    )


def bootstrap_acceptance_contrast(
    prompt_rows: List[Tuple[Dict[str, str], ...]],
    coefficients: Sequence[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> Dict[str, float]:
    point = acceptance_contrast(prompt_rows, coefficients)
    if len(prompt_rows) <= 1 or samples <= 0:
        return {"mean": point, "ci_low": point, "ci_high": point}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resampled = [prompt_rows[rng.randrange(len(prompt_rows))] for _ in prompt_rows]
        estimates.append(acceptance_contrast(resampled, coefficients))
    estimates.sort()
    return {
        "mean": point,
        "ci_low": estimates[int(0.025 * samples)],
        "ci_high": estimates[min(samples - 1, int(0.975 * samples))],
    }
