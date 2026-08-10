import math

from aggregate_objective_kv_results import audit_acceptance_rows


def make_row(prompt_idx, config, accept_rate, *, match=1, margin="nan"):
    return {
        "prompt_idx": str(prompt_idx),
        "config": config,
        "accept_rate": str(accept_rate),
        "matches_target_greedy": str(match),
        "mismatch_min_top1_margin": str(margin),
    }


def test_exactness_audit_keeps_ties_and_excludes_non_ties():
    rows = []
    values = {
        0: {"none": 0.5, "quality": 0.5, "acceptance": 0.6},
        1: {"none": 0.5, "quality": 0.4, "acceptance": 0.45},
        2: {"none": 0.5, "quality": 0.2, "acceptance": 0.9},
    }
    for prompt_idx, configs in values.items():
        for config, accept_rate in configs.items():
            if prompt_idx == 0:
                rows.append(make_row(prompt_idx, config, accept_rate))
            elif prompt_idx == 1:
                rows.append(make_row(prompt_idx, config, accept_rate, match=0, margin=0.0))
            else:
                rows.append(make_row(prompt_idx, config, accept_rate, match=0, margin=0.1))

    audit = audit_acceptance_rows(rows, quality_name="quality", acceptance_name="acceptance")

    assert audit["candidate_prompts"] == 3
    assert audit["valid_prompts"] == 2
    assert audit["excluded_non_tie_prompts"] == 1
    assert audit["row_counts"] == {
        "exact": 3,
        "numerical_tie": 3,
        "non_tie": 3,
        "invalid": 0,
    }
    assert math.isclose(audit["effects"]["quality_vs_native"]["mean"], -0.05)
    assert math.isclose(audit["effects"]["acceptance_vs_native"]["mean"], 0.025)
    assert math.isclose(audit["effects"]["acceptance_vs_quality"]["mean"], 0.075)


def test_exactness_audit_compares_optimized_policies_to_uniform():
    rows = []
    values = {
        0: {"none": 0.70, "k8v8": 0.65, "quality": 0.66, "acceptance": 0.68},
        1: {"none": 0.60, "k8v8": 0.55, "quality": 0.54, "acceptance": 0.57},
    }
    for prompt_idx, configs in values.items():
        for config, accept_rate in configs.items():
            rows.append(make_row(prompt_idx, config, accept_rate))

    audit = audit_acceptance_rows(
        rows,
        quality_name="quality",
        acceptance_name="acceptance",
        uniform_name="k8v8",
    )

    assert audit["candidate_prompts"] == 2
    assert audit["valid_prompts"] == 2
    assert math.isclose(audit["effects"]["uniform_vs_native"]["mean"], -0.05)
    assert math.isclose(audit["effects"]["quality_vs_uniform"]["mean"], 0.0, abs_tol=1e-12)
    assert math.isclose(audit["effects"]["acceptance_vs_uniform"]["mean"], 0.025)
