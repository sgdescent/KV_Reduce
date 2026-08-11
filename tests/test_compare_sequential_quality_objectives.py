import unittest

from compare_sequential_quality_objectives import matched_rows, paired_objective_contrasts


def spec_payload():
    return {
        "runtime": {
            "source_evaluator_version": "cached_dynamic_v6_sequential_target",
            "exactness_gate": "all_rows_match_independent_target_greedy",
            "full_run_gate": True,
        },
        "macro_summaries": [
            {
                "prompt_len": 1024,
                "config": "k4v4",
                "acceptance_rate": 0.55,
                "draft_cache_saved_fraction": 0.66,
                "total_cache_saved_fraction": 0.29,
            }
        ],
        "macro_contrasts": [
            {
                "prompt_len": 1024,
                "left_config": "k4v4",
                "right_config": "none",
                "acceptance_difference": -0.01,
                "ci_low": -0.02,
                "ci_high": 0.0,
            },
            {
                "prompt_len": 1024,
                "left_config": "k8v4",
                "right_config": "k4v8",
                "acceptance_difference": 0.02,
                "ci_low": 0.01,
                "ci_high": 0.03,
            },
        ],
    }


def quality_payload():
    return {
        "runtime": {
            "source_evaluator_version": "teacher_forced_cached_v1",
            "full_run_gate": True,
        },
        "grouped": [
            {
                "context": 1024,
                "config": "k4v4",
                "k_bits": 4,
                "v_bits": 4,
                "kl_p_to_q_mean": 0.005,
                "kl_p_to_q_ci_low": 0.004,
                "kl_p_to_q_ci_high": 0.006,
                "delta_nll_mean": 0.001,
                "top1_match_mean": 0.98,
            }
        ],
        "paired_precision_contrasts": [
            {
                "context": 1024,
                "config_a": "k8v4",
                "config_b": "k4v8",
                "kl_contrast_mean": 0.004,
                "kl_contrast_ci_low": 0.002,
                "kl_contrast_ci_high": 0.006,
            }
        ],
    }


class SequentialQualityObjectiveComparisonTest(unittest.TestCase):
    def test_matches_exact_acceptance_with_quality(self) -> None:
        rows = matched_rows(spec_payload(), quality_payload())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["config"], "k4v4")
        self.assertAlmostEqual(rows[0]["acceptance_delta_mean"], -0.01)

    def test_reports_resolved_preference_reversal(self) -> None:
        rows = paired_objective_contrasts(spec_payload(), quality_payload())
        self.assertEqual(rows[0]["spec_preference"], "k8v4")
        self.assertEqual(rows[0]["quality_preference"], "k4v8")
        self.assertTrue(rows[0]["resolved_preference_reversal"])

    def test_rejects_missing_full_run_gate(self) -> None:
        spec = spec_payload()
        spec["runtime"]["full_run_gate"] = False
        with self.assertRaisesRegex(ValueError, "full-run"):
            matched_rows(spec, quality_payload())


if __name__ == "__main__":
    unittest.main()
