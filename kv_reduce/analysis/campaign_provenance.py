"""Dependency-light provenance checks shared by paper aggregation tests."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


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
