import random
import unittest

from aggregate_kv_quantized_generation import aggregate, bootstrap_mean_ci


def make_row(prompt, config, exact, token_match):
    return {
        "run_id": "run0",
        "prompt_key": f"run0:{prompt}",
        "model": "model",
        "prompt_len": "1024",
        "max_new_tokens": "64",
        "config": config,
        "exact_sequence_match": str(exact),
        "token_match_fraction": str(token_match),
        "prefix_retained_fraction": str(token_match),
        "first_divergence_is_bf16_tie": "0",
        "cache_saved_fraction": "0.5",
    }


class QuantizedGenerationAggregationTest(unittest.TestCase):
    def test_bootstrap_single_value_is_degenerate(self) -> None:
        self.assertEqual(
            bootstrap_mean_ci([0.25], rng=random.Random(1), samples=100),
            (0.25, 0.25, 0.25),
        )

    def test_aggregate_preserves_paired_contrast(self) -> None:
        rows = []
        for prompt in range(3):
            rows.append(make_row(prompt, "k8v4", 1, 0.9))
            rows.append(make_row(prompt, "k4v8", 0, 0.7))
        summaries, contrasts = aggregate(rows, bootstrap_samples=100, seed=2)
        self.assertEqual(len(summaries), 2)
        token_contrast = next(
            row for row in contrasts if row["metric"] == "token_match_fraction"
        )
        self.assertAlmostEqual(token_contrast["difference"], 0.2)
        self.assertEqual(token_contrast["num_paired_prompts"], 3)


if __name__ == "__main__":
    unittest.main()
