import random
import unittest
from pathlib import Path

from aggregate_kv_quantized_generation import (
    aggregate,
    bootstrap_macro_mean_ci,
    bootstrap_mean_ci,
    validate_prompt_count,
)


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
    def test_full_run_gate_rejects_underfilled_summary(self) -> None:
        payload = {"config": {"num_prompts": 16}, "observed_prompts": 11}
        with self.assertRaisesRegex(ValueError, "requested 16, observed 11"):
            validate_prompt_count(
                payload,
                path=Path("summary.json"),
                require_full_runs=True,
            )
        self.assertEqual(
            validate_prompt_count(
                payload,
                path=Path("summary.json"),
                require_full_runs=False,
            ),
            (16, 11),
        )

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
        summaries, contrasts, macro_summaries, macro_contrasts = aggregate(
            rows, bootstrap_samples=100, seed=2
        )
        self.assertEqual(len(summaries), 2)
        token_contrast = next(
            row for row in contrasts if row["metric"] == "token_match_fraction"
        )
        self.assertAlmostEqual(token_contrast["difference"], 0.2)
        self.assertEqual(token_contrast["num_paired_prompts"], 3)
        self.assertEqual(len(macro_summaries), 2)
        macro_token_contrast = next(
            row
            for row in macro_contrasts
            if row["metric"] == "token_match_fraction"
        )
        self.assertAlmostEqual(macro_token_contrast["difference"], 0.2)

    def test_macro_average_weights_models_equally(self) -> None:
        first = make_row(0, "k4v4", 0, 0.0)
        first["model"] = "model-a"
        second = make_row(0, "k4v4", 1, 1.0)
        second["model"] = "model-b"
        _, _, macro_summaries, _ = aggregate(
            [first, second], bootstrap_samples=0, seed=3
        )
        self.assertEqual(len(macro_summaries), 1)
        self.assertAlmostEqual(macro_summaries[0]["token_match_fraction"], 0.5)

    def test_macro_bootstrap_requires_non_empty_groups(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_macro_mean_ci([], rng=random.Random(4), samples=10)


if __name__ == "__main__":
    unittest.main()
