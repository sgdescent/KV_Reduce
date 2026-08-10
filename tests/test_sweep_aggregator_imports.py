import unittest

from aggregate_value_precision_gamma_sweep import bootstrap_acceptance_contrast
from aggregate_value_precision_quality_sweep import (
    bootstrap_mean_ci,
    paired_sequence_metric_differences,
)
from spec_kv_statistics import align_config_rows_by_prompt


class SweepAggregatorImportsTest(unittest.TestCase):
    def test_quality_aggregator_uses_mean_bootstrap(self) -> None:
        result = bootstrap_mean_ci([1.0, 3.0], seed=0, samples=20)
        self.assertEqual(result["mean"], 2.0)

    def test_gamma_aggregator_uses_acceptance_bootstrap(self) -> None:
        candidate = {"accepted_tokens": "8", "proposed_tokens": "10"}
        baseline = {"accepted_tokens": "5", "proposed_tokens": "10"}
        result = bootstrap_acceptance_contrast(
            [(candidate, baseline)],
            (1.0, -1.0),
            seed=0,
            samples=20,
        )
        self.assertAlmostEqual(result["mean"], 0.3)

    def test_config_pairing_aligns_rows_by_prompt(self) -> None:
        effects = {
            "k4v2": [
                (
                    {"prompt_idx": "1", "accepted_tokens": "1", "proposed_tokens": "2"},
                    {"prompt_idx": "1"},
                ),
                (
                    {"prompt_idx": "0", "accepted_tokens": "2", "proposed_tokens": "2"},
                    {"prompt_idx": "0"},
                ),
            ],
            "k2v4": [
                (
                    {"prompt_idx": "0", "accepted_tokens": "1", "proposed_tokens": "2"},
                    {"prompt_idx": "0"},
                ),
                (
                    {"prompt_idx": "2", "accepted_tokens": "2", "proposed_tokens": "2"},
                    {"prompt_idx": "2"},
                ),
                (
                    {"prompt_idx": "1", "accepted_tokens": "2", "proposed_tokens": "2"},
                    {"prompt_idx": "1"},
                ),
            ],
        }

        paired = align_config_rows_by_prompt(effects, "k4v2", "k2v4")

        self.assertEqual(
            [(left["prompt_idx"], right["prompt_idx"]) for left, right in paired],
            [("0", "0"), ("1", "1")],
        )

    def test_quality_pairing_aligns_rows_by_sequence(self) -> None:
        rows = [
            {"sequence_idx": "1", "candidate": "k8v4", "kl_p_to_q": "0.4"},
            {"sequence_idx": "0", "candidate": "k4v8", "kl_p_to_q": "0.2"},
            {"sequence_idx": "0", "candidate": "k8v4", "kl_p_to_q": "0.3"},
            {"sequence_idx": "2", "candidate": "k4v8", "kl_p_to_q": "0.1"},
            {"sequence_idx": "1", "candidate": "k4v8", "kl_p_to_q": "0.1"},
        ]

        differences = paired_sequence_metric_differences(
            rows,
            config_a="k8v4",
            config_b="k4v8",
            metric="kl_p_to_q",
        )

        self.assertEqual(len(differences), 2)
        self.assertAlmostEqual(differences[0], 0.3)
        self.assertAlmostEqual(differences[1], 0.1)


if __name__ == "__main__":
    unittest.main()
