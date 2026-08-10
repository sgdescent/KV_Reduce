import unittest

from aggregate_joint_target_draft_quantization import collect_prompt_effects


class JointTargetDraftAggregationTest(unittest.TestCase):
    def test_target_quantization_mismatch_is_an_outcome_not_an_exclusion(self) -> None:
        baseline = "target_none__draft_none"
        candidate = "target_k4v4__draft_k4v4"
        rows = [
            {
                "prompt_idx": "0",
                "config": baseline,
                "matches_target_greedy": "1",
                "mismatch_min_top1_margin": "nan",
                "accepted_tokens": "8",
                "proposed_tokens": "16",
            },
            {
                "prompt_idx": "0",
                "config": candidate,
                "matches_target_greedy": "0",
                "mismatch_min_top1_margin": "0.125",
                "accepted_tokens": "9",
                "proposed_tokens": "16",
            },
        ]
        effects, exactness, invalid = collect_prompt_effects(
            rows,
            config_names=[baseline, candidate],
            baseline_name=baseline,
            tie_margin=1e-3,
        )
        self.assertEqual(invalid, 0)
        self.assertEqual(len(effects[candidate]), 1)
        self.assertEqual(exactness[candidate]["non_tie_or_unknown"], 1)

    def test_non_tie_bf16_baseline_invalidates_only_that_prompt(self) -> None:
        baseline = "target_none__draft_none"
        candidate = "target_k8v8__draft_none"
        rows = [
            {
                "prompt_idx": "0",
                "config": baseline,
                "matches_target_greedy": "0",
                "mismatch_min_top1_margin": "0.1",
                "accepted_tokens": "8",
                "proposed_tokens": "16",
            },
            {
                "prompt_idx": "0",
                "config": candidate,
                "matches_target_greedy": "1",
                "mismatch_min_top1_margin": "nan",
                "accepted_tokens": "8",
                "proposed_tokens": "16",
            },
        ]
        effects, _, invalid = collect_prompt_effects(
            rows,
            config_names=[baseline, candidate],
            baseline_name=baseline,
            tie_margin=1e-3,
        )
        self.assertEqual(invalid, 1)
        self.assertEqual(effects[candidate], [])


if __name__ == "__main__":
    unittest.main()
