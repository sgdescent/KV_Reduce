import math

from resample_kv_sensitivity_profile import (
    _acceptance_profile,
    _invalid_acceptance_ids,
    _quality_profile,
)


def test_quality_profile_recomputes_prefix_mean_risk():
    profile = [
        {"candidate": "baseline_none", "cache_bytes_saved": "0"},
        {"candidate": "layer0_k4", "cache_bytes_saved": "12"},
    ]
    raw = [
        {"sequence_idx": 0, "candidate": "baseline_none", "kl_p_to_q": 0.0, "delta_nll": 0.0},
        {"sequence_idx": 1, "candidate": "baseline_none", "kl_p_to_q": 0.0, "delta_nll": 0.0},
        {"sequence_idx": 2, "candidate": "baseline_none", "kl_p_to_q": 0.0, "delta_nll": 0.0},
        {"sequence_idx": 0, "candidate": "layer0_k4", "kl_p_to_q": 0.1, "delta_nll": 0.2},
        {"sequence_idx": 1, "candidate": "layer0_k4", "kl_p_to_q": 0.3, "delta_nll": 0.4},
        {"sequence_idx": 2, "candidate": "layer0_k4", "kl_p_to_q": 9.0, "delta_nll": 9.0},
    ]

    rebuilt = _quality_profile(profile, raw, selected_ids={0, 1}, risk_metric="kl")

    assert rebuilt[1]["cache_bytes_saved"] == "12"
    assert math.isclose(rebuilt[1]["quality_risk"], 0.2)
    assert math.isclose(rebuilt[1]["delta_nll"], 0.3)


def test_acceptance_profile_uses_valid_paired_prompt_rows():
    profile = [
        {"candidate": "baseline_none", "cache_bytes_saved": "0"},
        {"candidate": "layer0_v4", "cache_bytes_saved": "10"},
    ]
    raw = [
        {
            "prompt_idx": idx,
            "config": config,
            "accept_rate": rate,
            "round_accept_mass": mass,
            "accepted_per_round": 1.0,
            "round_js": 0.1,
            "round_top1_match": 0.5,
        }
        for idx, baseline, candidate in ((0, 0.8, 0.6), (1, 0.6, 0.5))
        for config, rate, mass in (
            ("baseline_none", baseline, baseline),
            ("layer0_v4", candidate, candidate),
        )
    ]

    rebuilt = _acceptance_profile(
        profile,
        raw,
        valid_ids={0, 1},
        baseline_config="baseline_none",
        z_score=1.96,
    )

    candidate = rebuilt[1]
    assert candidate["cache_bytes_saved"] == "10"
    assert math.isclose(candidate["accept_rate_drop"], 0.15)
    assert math.isclose(candidate["accept_rate_drop_prompt_mean"], 0.15)
    assert candidate["paired_prompt_count"] == 2.0


def test_acceptance_exactness_filter_keeps_ties_and_rejects_non_ties():
    rows = [
        {"prompt_idx": 0, "matches_target_greedy": 1, "mismatch_min_top1_margin": "nan"},
        {"prompt_idx": 1, "matches_target_greedy": 0, "mismatch_min_top1_margin": 0.0005},
        {"prompt_idx": 2, "matches_target_greedy": 0, "mismatch_min_top1_margin": 0.1},
        {"prompt_idx": 3, "matches_target_greedy": 0, "mismatch_min_top1_margin": "nan"},
    ]

    assert _invalid_acceptance_ids(rows, selected_ids={0, 1, 2, 3}, tie_margin=1e-3) == {2, 3}
