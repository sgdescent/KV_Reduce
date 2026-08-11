import unittest

from pathlib import Path

from aggregate_value_precision_sweep import (
    aggregate_prompt_effects,
    parse_config_bits,
    parse_config_set,
    parse_int_set,
    underfilled_run_record,
    validate_prompt_config_coverage,
)


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

    def test_underfilled_run_record_only_reports_shortfalls(self):
        record = underfilled_run_record(
            run_dir=Path("ctx_1024/seed_0"),
            requested=512,
            actual=255,
            unit="prompts",
            context=1024,
            seed=0,
        )
        self.assertEqual(record["shortfall"], 257)
        self.assertEqual(record["actual"], 255)
        self.assertIsNone(
            underfilled_run_record(
                run_dir=Path("ctx_1024/seed_1"),
                requested=512,
                actual=512,
                unit="prompts",
                context=1024,
                seed=1,
            )
        )

    def test_strict_argument_parsers(self):
        self.assertEqual(parse_int_set("1024, 4096"), {1024, 4096})
        self.assertEqual(
            parse_config_set("none;k8v4;k4v8"), {"none", "k8v4", "k4v8"}
        )

    def test_prompt_config_coverage_is_fail_closed(self):
        rows = [
            {"prompt_idx": "0", "config": "none"},
            {"prompt_idx": "0", "config": "k8v4"},
            {"prompt_idx": "1", "config": "none"},
        ]
        self.assertEqual(
            validate_prompt_config_coverage(rows, {"none", "k8v4"}),
            ["1"],
        )


if __name__ == "__main__":
    unittest.main()
