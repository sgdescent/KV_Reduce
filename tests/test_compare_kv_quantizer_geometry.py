import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from compare_kv_quantizer_geometry import join_geometry_rows, load_kivi_rows


class QuantizerGeometryComparisonTest(unittest.TestCase):
    def test_loads_complete_preference_table(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "preference_rows.csv"
            path.write_text(
                "pair,context,config_a,config_b,spec_acceptance_a_minus_b_mean,"
                "spec_acceptance_a_minus_b_ci_low,spec_acceptance_a_minus_b_ci_high,"
                "config_a_total_saved_fraction,config_b_total_saved_fraction,"
                "absolute_total_saved_fraction_gap\n"
                "pair_a,1024,k8v4,k4v8,0.01,-0.02,0.03,0.20,0.19,0.01\n"
                "pair_a,4096,k8v4,k4v8,0.50,0.40,0.60,0.30,0.29,0.01\n",
                encoding="utf-8",
            )
            rows = load_kivi_rows(path, context=1024)
        self.assertEqual(set(rows), {"pair_a"})
        self.assertAlmostEqual(rows["pair_a"]["effect"], 0.01)
        self.assertAlmostEqual(rows["pair_a"]["memory_saved_fraction"], 0.195)

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
