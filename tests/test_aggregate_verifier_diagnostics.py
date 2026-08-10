import unittest

from aggregate_verifier_diagnostics import (
    classify_speculative_mismatches,
    is_numerical_tie,
)


class VerifierMismatchClassificationTest(unittest.TestCase):
    def test_tie_rule_rejects_missing_and_nonfinite_margins(self):
        self.assertTrue(is_numerical_tie(0.001, 0.001))
        self.assertTrue(is_numerical_tie(-0.0005, 0.001))
        self.assertFalse(is_numerical_tie(0.002, 0.001))
        self.assertFalse(is_numerical_tie(None, 0.001))
        self.assertFalse(is_numerical_tie(float("nan"), 0.001))

    def test_classifies_decisions_and_independent_sequences(self):
        payload = {
            "speculative_audits": [
                {
                    "matches_independent_target_greedy": False,
                    "first_independent_reference_mismatch_margin": 0.0,
                    "decisions": [
                        {"top1_match": 0.0, "reference_margin": 0.0},
                        {"top1_match": 1.0, "reference_margin": 1.0},
                    ],
                },
                {
                    "matches_independent_target_greedy": False,
                    "first_independent_reference_mismatch_margin": 0.125,
                    "decisions": [
                        {"top1_match": 0.0, "reference_margin": 0.125},
                        {"top1_match": 0.0},
                    ],
                },
                {
                    "matches_independent_target_greedy": True,
                    "decisions": [],
                },
            ]
        }

        self.assertEqual(
            classify_speculative_mismatches(payload, tie_margin=1e-3),
            {
                "speculative_tie_decisions": 1,
                "speculative_non_tie_or_unknown_decisions": 2,
                "speculative_prompts_with_only_tie_mismatches": 1,
                "speculative_prompts_with_non_tie_or_unknown_mismatches": 1,
                "independent_greedy_tie_prompts": 1,
                "independent_greedy_non_tie_or_unknown_prompts": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
