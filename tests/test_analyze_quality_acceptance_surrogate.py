import unittest

from analyze_quality_acceptance_surrogate import (
    average_ranks,
    quality_gate_metrics,
)


class QualityAcceptanceSurrogateTest(unittest.TestCase):
    def test_average_ranks_ties(self):
        self.assertEqual(average_ranks([3.0, 1.0, 1.0]).tolist(), [2.0, 0.5, 0.5])

    def test_quality_gate_reports_false_safe_cells(self):
        rows = [
            {"quality_kl_mean": 0.005, "acceptance_delta_mean": -0.01},
            {"quality_kl_mean": 0.008, "acceptance_delta_mean": -0.04},
            {"quality_kl_mean": 0.020, "acceptance_delta_mean": -0.01},
        ]
        result = quality_gate_metrics(
            rows,
            quality_kl_budget=0.01,
            acceptance_drop_budget=0.02,
        )
        self.assertEqual(result["num_selected"], 2)
        self.assertEqual(result["num_selected_and_safe"], 1)
        self.assertEqual(result["false_safe_count"], 1)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
