import unittest

from compare_kv_quantizer_geometry import join_geometry_rows


class QuantizerGeometryComparisonTest(unittest.TestCase):
    def test_joins_pairs_and_computes_shift(self):
        naive = {
            "pair_a": {
                "effect": 0.2,
                "ci_low": 0.1,
                "ci_high": 0.3,
                "memory_saved_fraction": 0.25,
            }
        }
        kivi = {
            "pair_a": {
                "effect": -0.05,
                "ci_low": -0.1,
                "ci_high": 0.0,
                "memory_saved_fraction": 0.24,
                "memory_gap": 0.001,
            },
            "unmatched": {
                "effect": 1.0,
                "ci_low": 1.0,
                "ci_high": 1.0,
                "memory_saved_fraction": 0.2,
            },
        }
        rows = join_geometry_rows(naive, kivi)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pair"], "pair_a")
        self.assertAlmostEqual(rows[0]["geometry_shift_kivi_minus_naive"], -0.25)


if __name__ == "__main__":
    unittest.main()
