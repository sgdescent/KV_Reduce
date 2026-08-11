import random
import unittest

from aggregate_sequential_spec_runs import (
    aggregate,
    bootstrap_paired_ratio_difference,
    bootstrap_ratio_ci,
)


def row(prompt, config, accepted, proposed):
    return {
        "run_id": "run0",
        "prompt_key": f"run0:{prompt}",
        "big_model": "target",
        "small_model": "draft",
        "prompt_len": 1024,
        "max_new_tokens": 16,
        "draft_steps": 4,
        "config": config,
        "accepted_tokens": accepted,
        "proposed_tokens": proposed,
        "draft_cache_saved_fraction": 0.5,
        "total_cache_saved_fraction": 0.2,
    }


class SequentialSpecAggregationTest(unittest.TestCase):
    def test_ratio_uses_token_counts_not_mean_prompt_rates(self):
        mean, low, high = bootstrap_ratio_ci(
            [1, 9], [2, 18], rng=random.Random(1), samples=0
        )
        self.assertEqual((mean, low, high), (0.5, 0.5, 0.5))

    def test_paired_difference_preserves_pairing(self):
        result = bootstrap_paired_ratio_difference(
            [(8, 10), (4, 10)],
            [(6, 10), (2, 10)],
            rng=random.Random(2),
            samples=0,
        )
        for value in result:
            self.assertAlmostEqual(value, 0.2)

    def test_aggregate_reports_exact_paired_contrast(self):
        rows = []
        for prompt in range(3):
            rows.append(row(prompt, "k8v4", 4, 10))
            rows.append(row(prompt, "k4v8", 6, 10))
        summaries, contrasts, macro_summaries, macro_contrasts = aggregate(
            rows, bootstrap_samples=0, seed=3
        )
        self.assertEqual(len(summaries), 2)
        self.assertEqual(len(contrasts), 1)
        self.assertAlmostEqual(contrasts[0]["acceptance_difference"], -0.2)
        self.assertEqual(contrasts[0]["num_paired_prompts"], 3)
        self.assertEqual(len(macro_summaries), 2)
        self.assertAlmostEqual(macro_contrasts[0]["acceptance_difference"], -0.2)

    def test_aggregate_does_not_pool_proposal_lengths(self):
        rows = []
        for draft_steps in (2, 8):
            left = row(0, "k4v2", draft_steps, 10)
            right = row(0, "k2v4", 10 - draft_steps, 10)
            left["draft_steps"] = draft_steps
            right["draft_steps"] = draft_steps
            rows.extend((left, right))
        summaries, contrasts, macro_summaries, macro_contrasts = aggregate(
            rows, bootstrap_samples=0, seed=4
        )
        self.assertEqual({item["draft_steps"] for item in summaries}, {2, 8})
        self.assertEqual(len(contrasts), 2)
        differences = {
            item["draft_steps"]: item["acceptance_difference"] for item in contrasts
        }
        self.assertAlmostEqual(differences[2], -0.6)
        self.assertAlmostEqual(differences[8], 0.6)
        self.assertEqual({item["draft_steps"] for item in macro_summaries}, {2, 8})
        self.assertEqual(len(macro_contrasts), 2)

    def test_macro_average_weights_model_pairs_equally(self):
        first = row(0, "k4v4", 0, 10)
        first["big_model"] = "target-a"
        first["small_model"] = "draft-a"
        second = row(0, "k4v4", 90, 90)
        second["big_model"] = "target-b"
        second["small_model"] = "draft-b"
        _, _, macro_summaries, _ = aggregate(
            [first, second], bootstrap_samples=0, seed=5
        )
        self.assertEqual(len(macro_summaries), 1)
        self.assertAlmostEqual(macro_summaries[0]["acceptance_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
