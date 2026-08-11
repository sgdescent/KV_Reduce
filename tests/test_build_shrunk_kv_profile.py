from build_shrunk_kv_profile import (
    calibrate_prior_scale,
    estimate_residual_variance,
    shrink_rows,
)


def row(layer, component, bits, **values):
    return {
        "candidate": f"layer{layer}_{component}{bits}",
        "layer": str(layer),
        "component": component,
        "bits": str(bits),
        **{key: str(value) for key, value in values.items()},
    }


def test_calibrates_nonnegative_quality_prior_scale():
    assert calibrate_prior_scale([1.0, 2.0], [2.0, 4.0]) == 2.0
    assert calibrate_prior_scale([1.0], [-3.0]) == 0.0


def test_residual_variance_subtracts_measurement_noise():
    variance = estimate_residual_variance([1.0, 3.0], [1.0, 1.0], [0.5, 0.5])
    assert variance == 1.75


def test_noisy_acceptance_effect_shrinks_toward_quality_prior():
    quality = [
        row(0, "k", 4, quality_risk=0.1),
        row(0, "v", 4, quality_risk=0.2),
        row(1, "k", 4, quality_risk=0.3),
    ]
    acceptance = [
        row(0, "k", 4, accept_rate_drop_prompt_mean=0.1, accept_rate_drop_prompt_se=0.01),
        row(0, "v", 4, accept_rate_drop_prompt_mean=0.2, accept_rate_drop_prompt_se=0.01),
        row(1, "k", 4, accept_rate_drop_prompt_mean=1.2, accept_rate_drop_prompt_se=1.0),
    ]
    rows, diagnostics = shrink_rows(
        quality,
        acceptance,
        quality_field="quality_risk",
        acceptance_field="accept_rate_drop_prompt_mean",
        acceptance_se_field="accept_rate_drop_prompt_se",
        prior_strength=1.0,
        ucb_z=1.96,
        variance_mode="empirical",
    )
    noisy = next(row for row in rows if int(row["layer"]) == 1)
    assert 0.0 <= noisy["acceptance_reliability"] < 0.5
    assert noisy["shrunk_acceptance_risk"] < noisy["observed_acceptance_risk"]
    assert noisy["shrunk_acceptance_ucb"] >= noisy["shrunk_acceptance_risk"]
    assert diagnostics["num_common_candidates"] == 3.0
