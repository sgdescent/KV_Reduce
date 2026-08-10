import unittest

from eval_kv_quantized_generation import mean_numeric_rows, rollout_comparison


class QuantizedGenerationMetricsTest(unittest.TestCase):
    def test_exact_rollout_retains_full_prefix(self) -> None:
        metrics = rollout_comparison([1, 2, 3], [1, 2, 3], [1.0, 0.5, 0.2], tie_margin=1e-3)
        self.assertEqual(metrics["exact_sequence_match"], 1.0)
        self.assertEqual(metrics["token_match_fraction"], 1.0)
        self.assertEqual(metrics["prefix_retained_fraction"], 1.0)
        self.assertEqual(metrics["first_divergence_position"], -1)
        self.assertIsNone(metrics["reference_margin_at_first_divergence"])

    def test_divergence_records_margin_and_later_recovery(self) -> None:
        metrics = rollout_comparison(
            [1, 2, 3, 4],
            [1, 9, 3, 4],
            [2.0, 5e-4, 0.4, 0.1],
            tie_margin=1e-3,
        )
        self.assertEqual(metrics["exact_sequence_match"], 0.0)
        self.assertEqual(metrics["token_match_fraction"], 0.75)
        self.assertEqual(metrics["prefix_match_tokens"], 1.0)
        self.assertEqual(metrics["first_divergence_position"], 1)
        self.assertEqual(metrics["first_divergence_is_bf16_tie"], 1.0)

    def test_mean_rows_excludes_metadata_and_missing_values(self) -> None:
        summary = mean_numeric_rows(
            [
                {"prompt_idx": 0, "config": "k4v4", "score": 0.5, "margin": None},
                {"prompt_idx": 1, "config": "k4v4", "score": 1.0, "margin": 0.2},
            ],
            exclude=("prompt_idx",),
        )
        self.assertEqual(summary["score"], 0.75)
        self.assertEqual(summary["margin"], 0.2)
        self.assertNotIn("prompt_idx", summary)


if __name__ == "__main__":
    unittest.main()
