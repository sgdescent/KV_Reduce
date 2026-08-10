import unittest

from spec_kv_statistics import acceptance_contrast, acceptance_ratio


def row(accepted: int, proposed: int):
    return {"accepted_tokens": str(accepted), "proposed_tokens": str(proposed)}


class SpecKvStatisticsTest(unittest.TestCase):
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
