import unittest

from spec_kv_statistics import acceptance_contrast, acceptance_ratio, bootstrap_mean_ci


def row(accepted: int, proposed: int):
    return {"accepted_tokens": str(accepted), "proposed_tokens": str(proposed)}


class SpecKvStatisticsTest(unittest.TestCase):
    def test_bootstrap_mean_reports_point_estimate(self) -> None:
        result = bootstrap_mean_ci([1.0, 2.0, 3.0], seed=7, samples=100)
        self.assertEqual(result["mean"], 2.0)
        self.assertLessEqual(result["ci_low"], result["mean"])
        self.assertGreaterEqual(result["ci_high"], result["mean"])

    def test_acceptance_ratio_weights_by_proposals(self) -> None:
        self.assertAlmostEqual(acceptance_ratio([row(10, 10), row(0, 90)]), 0.1)

    def test_contrast_uses_ratio_of_totals_for_each_arm(self) -> None:
        pairs = [(row(10, 10), row(0, 10)), (row(0, 90), row(45, 90))]
        self.assertAlmostEqual(acceptance_contrast(pairs, (1.0, -1.0)), -0.35)

    def test_four_arm_axis_contrast(self) -> None:
        rows = [(row(80, 100), row(60, 100), row(50, 100), row(40, 100))]
        self.assertAlmostEqual(acceptance_contrast(rows, (1.0, -1.0, -1.0, 1.0)), 0.1)


if __name__ == "__main__":
    unittest.main()
