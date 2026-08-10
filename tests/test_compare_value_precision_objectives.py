import math
import unittest

from compare_value_precision_objectives import (
    pareto_configs,
    rankdata,
    select_max_savings,
    select_objective_choices,
    spearman,
)


class CompareValuePrecisionObjectivesTest(unittest.TestCase):
    def test_rankdata_averages_ties(self):
        self.assertEqual(rankdata([3.0, 1.0, 1.0, 2.0]), [4.0, 1.5, 1.5, 3.0])

    def test_spearman_detects_matching_order(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertTrue(math.isnan(spearman([1, 1], [2, 3])))

    def test_selection_distinguishes_mean_and_conservative_feasibility(self):
        rows = [
            {
                "config": "k8v4",
                "total_cache_saved_fraction": 0.27,
                "acceptance_delta_mean": -0.01,
                "acceptance_delta_ci_low": -0.019,
                "acceptance_delta_ci_high": 0.0,
                "quality_kl_mean": 0.005,
                "quality_kl_ci_high": 0.009,
            },
            {
                "config": "k8v3",
                "total_cache_saved_fraction": 0.283,
                "acceptance_delta_mean": -0.015,
                "acceptance_delta_ci_low": -0.04,
                "acceptance_delta_ci_high": 0.0,
                "quality_kl_mean": 0.008,
                "quality_kl_ci_high": 0.012,
            },
        ]
        selected = select_max_savings(rows, acceptance_drop_budget=0.02, quality_kl_budget=0.01)
        self.assertEqual(selected["best_mean_feasible"]["config"], "k8v3")
        self.assertEqual(selected["best_conservative_feasible"]["config"], "k8v4")

    def test_objective_choices_report_cross_regret(self):
        rows = [
            {
                "config": "k8v4",
                "total_cache_saved_fraction": 0.27,
                "acceptance_delta_mean": -0.005,
                "quality_kl_mean": 0.02,
            },
            {
                "config": "k4v8",
                "total_cache_saved_fraction": 0.27,
                "acceptance_delta_mean": -0.02,
                "quality_kl_mean": 0.005,
            },
        ]

        choice = select_objective_choices(rows, minimum_savings=0.25)

        self.assertEqual(choice["spec_choice"], "k8v4")
        self.assertEqual(choice["quality_choice"], "k4v8")
        self.assertTrue(choice["objective_disagreement"])
        self.assertAlmostEqual(choice["acceptance_regret_of_quality_choice"], 0.015)
        self.assertAlmostEqual(choice["quality_kl_regret_of_spec_choice"], 0.015)

    def test_pareto_front_removes_dominated_configs(self):
        rows = [
            {"config": "a", "total_cache_saved_fraction": 0.2, "harm": 0.01},
            {"config": "b", "total_cache_saved_fraction": 0.3, "harm": 0.02},
            {"config": "dominated", "total_cache_saved_fraction": 0.1, "harm": 0.03},
        ]

        self.assertEqual(pareto_configs(rows, harm_key="harm"), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
