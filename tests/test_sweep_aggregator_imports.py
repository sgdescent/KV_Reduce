import unittest

from aggregate_value_precision_gamma_sweep import bootstrap_acceptance_contrast
from aggregate_value_precision_quality_sweep import bootstrap_mean_ci


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


if __name__ == "__main__":
    unittest.main()
