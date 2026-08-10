from paper.compare_calibration_profiles import component_summary, paired_stability


def candidate(name: str, component: str, bits: int, rate_risk: float, mass_risk: float):
    return {
        "candidate": name,
        "component": component,
        "bits": str(bits),
        "paired_prompt_count": "16",
        "accept_rate_drop_prompt_mean": str(rate_risk / 2),
        "accept_rate_drop_ucb95_clipped": str(rate_risk),
        "accept_mass_drop_prompt_mean": str(mass_risk / 2),
        "accept_mass_drop_ucb95_clipped": str(mass_risk),
    }


def test_calibration_summary_and_ranking_stability():
    small = {
        "layer0_k4": candidate("layer0_k4", "k", 4, 0.4, 0.2),
        "layer0_v4": candidate("layer0_v4", "v", 4, 0.1, 0.05),
        "layer1_k4": candidate("layer1_k4", "k", 4, 0.3, 0.15),
    }
    large = {
        "layer0_k4": candidate("layer0_k4", "k", 4, 0.5, 0.25),
        "layer0_v4": candidate("layer0_v4", "v", 4, 0.05, 0.02),
        "layer1_k4": candidate("layer1_k4", "k", 4, 0.4, 0.20),
    }

    component_rows = component_summary([("n16", small), ("n64", large)])
    large_k = next(
        row for row in component_rows if row["profile"] == "n64" and row["component"] == "k"
    )
    assert large_k["accept_rate_drop_ucb95_clipped"] == 0.45

    stability = paired_stability(
        [("n16", small), ("n64", large)], "accept_rate_drop_ucb95_clipped", top_k=2
    )[0]
    assert stability["num_common_candidates"] == 3
    assert stability["spearman"] == 1.0
    assert stability["top_k_overlap"] == 2
