import unittest

from aggregate_longbench_cross_family import hierarchical_macro_ci, pair_effects


class LongBenchCrossFamilyAggregationTest(unittest.TestCase):
    def test_pairs_normalized_accuracy_by_seed_and_source(self):
        rows = [
            {"seed": "0", "source_idx": "4", "config": "none", "normalized_correct": "1"},
            {"seed": "0", "source_idx": "4", "config": "k4v4", "normalized_correct": "0"},
            {"seed": "1", "source_idx": "4", "config": "none", "normalized_correct": "0"},
            {"seed": "1", "source_idx": "4", "config": "k4v4", "normalized_correct": "1"},
        ]
        self.assertEqual(
            pair_effects(rows, config_a="k4v4", config_b="none"),
            [-1.0, 1.0],
        )

    def test_hierarchical_macro_uses_equal_model_weight(self):
        estimate, low, high = hierarchical_macro_ci(
            [[1.0, 1.0], [-1.0]],
            samples=500,
            seed=7,
        )
        self.assertEqual(estimate, 0.0)
        self.assertLessEqual(low, estimate)
        self.assertGreaterEqual(high, estimate)

    def test_rejects_empty_model_group(self):
        with self.assertRaises(ValueError):
            hierarchical_macro_ci([[1.0], []], samples=10, seed=7)


if __name__ == "__main__":
    unittest.main()
