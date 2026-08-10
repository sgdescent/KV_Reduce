import unittest

from aggregate_value_precision_sweep import aggregate_prompt_effects, parse_config_bits


class ValuePrecisionSweepTest(unittest.TestCase):
    def test_parses_compact_bit_configs(self):
        self.assertEqual(parse_config_bits("none"), (16, 16))
        self.assertEqual(parse_config_bits("k8v2"), (8, 2))
        self.assertEqual(parse_config_bits("k16v3"), (16, 3))

    def test_prompt_effects_keep_ties_and_drop_non_ties(self):
        rows = [
            {"prompt_idx": "0", "config": "none", "accepted_tokens": "5", "proposed_tokens": "10", "matches_target_greedy": "1"},
            {"prompt_idx": "0", "config": "k8v2", "accepted_tokens": "4", "proposed_tokens": "10", "matches_target_greedy": "1"},
            {"prompt_idx": "1", "config": "none", "accepted_tokens": "5", "proposed_tokens": "10", "matches_target_greedy": "0", "mismatch_min_top1_margin": "0"},
            {"prompt_idx": "1", "config": "k8v2", "accepted_tokens": "6", "proposed_tokens": "10", "matches_target_greedy": "0", "mismatch_min_top1_margin": "0"},
            {"prompt_idx": "2", "config": "none", "accepted_tokens": "5", "proposed_tokens": "10", "matches_target_greedy": "0", "mismatch_min_top1_margin": "0.2"},
            {"prompt_idx": "2", "config": "k8v2", "accepted_tokens": "9", "proposed_tokens": "10", "matches_target_greedy": "0", "mismatch_min_top1_margin": "0.2"},
        ]
        effects, counts, invalid = aggregate_prompt_effects(rows, configs=["k8v2"], tie_margin=1e-3)
        self.assertEqual(effects["k8v2"][0][0]["accepted_tokens"], "4")
        self.assertEqual(effects["k8v2"][1][0]["accepted_tokens"], "6")
        self.assertEqual(counts["exact"], 2)
        self.assertEqual(counts["numerical_tie"], 2)
        self.assertEqual(counts["non_tie_or_unknown"], 2)
        self.assertEqual(invalid, 1)


if __name__ == "__main__":
    unittest.main()
