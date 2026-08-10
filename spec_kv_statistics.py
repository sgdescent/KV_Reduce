"""Shared proposal-weighted statistics for speculative-decoding experiments."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PromptRow = Dict[str, str]
PromptEffect = Tuple[PromptRow, PromptRow]


def sample_count_status(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
) -> Dict[str, Any]:
    """Compare a run's requested prompt count with unique observed prompts."""
    observed = len({str(row["prompt_idx"]) for row in rows})
    requested = int(summary.get("config", {}).get("num_prompts", observed))
    return {
        "requested_num_prompts": requested,
        "observed_num_prompts": observed,
        "underfilled": observed < requested,
        "prompt_shortfall": max(0, requested - observed),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> Dict[str, float]:
    if not values:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    point = sum(float(value) for value in values) / len(values)
    if len(values) <= 1 or samples <= 0:
        return {"mean": point, "ci_low": point, "ci_high": point}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(
            sum(float(values[rng.randrange(len(values))]) for _ in values) / len(values)
        )
    estimates.sort()
    return {
        "mean": point,
        "ci_low": estimates[int(0.025 * samples)],
        "ci_high": estimates[min(samples - 1, int(0.975 * samples))],
    }


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


def align_config_rows_by_prompt(
    effects: Mapping[str, Sequence[PromptEffect]],
    config_a: str,
    config_b: str,
) -> List[Tuple[PromptRow, PromptRow]]:
    """Align two quantized configurations using their shared baseline prompt IDs."""
    rows_a = {
        baseline["prompt_idx"]: quantized
        for quantized, baseline in effects.get(config_a, ())
    }
    rows_b = {
        baseline["prompt_idx"]: quantized
        for quantized, baseline in effects.get(config_b, ())
    }
    return [
        (rows_a[prompt], rows_b[prompt])
        for prompt in sorted(rows_a.keys() & rows_b.keys())
    ]
