import unittest

from select_joint_kv_quantization_policy import evaluate_candidates, select_by_context


class JointKVPolicySelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.joint = {
            "grouped": [
                {
                    "context": 1024,
                    "config": "target_none__draft_k4v4",
                    "target_config": "none",
                    "draft_config": "k4v4",
                    "paired_acceptance_delta_mean": -0.005,
                    "paired_acceptance_delta_ci_low": -0.015,
                    "total_cache_saved_fraction": 0.30,
                },
                {
                    "context": 1024,
                    "config": "target_k4v4__draft_k4v4",
                    "target_config": "k4v4",
                    "draft_config": "k4v4",
                    "paired_acceptance_delta_mean": -0.01,
                    "paired_acceptance_delta_ci_low": -0.018,
                    "total_cache_saved_fraction": 0.62,
                },
                {
                    "context": 1024,
                    "config": "target_k2v2__draft_k2v2",
                    "target_config": "k2v2",
                    "draft_config": "k2v2",
                    "paired_acceptance_delta_mean": -0.04,
                    "paired_acceptance_delta_ci_low": -0.06,
                    "total_cache_saved_fraction": 0.80,
                },
            ]
        }
        self.quality = {
            "grouped": [
                {
                    "context": 1024,
                    "config": "k4v4",
                    "kl_p_to_q_mean": 0.006,
                    "kl_p_to_q_ci_high": 0.008,
                    "delta_nll_mean": 0.004,
                    "delta_nll_ci_high": 0.009,
                    "top1_match_mean": 0.97,
                    "accept_mass_mean": 0.98,
                },
                {
                    "context": 1024,
                    "config": "k2v2",
                    "kl_p_to_q_mean": 0.08,
                    "kl_p_to_q_ci_high": 0.09,
                    "delta_nll_mean": 0.07,
                    "delta_nll_ci_high": 0.08,
                    "top1_match_mean": 0.80,
                    "accept_mass_mean": 0.85,
                },
            ]
        }

    def test_selects_highest_saving_feasible_joint_policy(self) -> None:
        rows = evaluate_candidates(
            self.joint,
            self.quality,
            target_kl_max=0.01,
            target_delta_nll_max=0.02,
            target_top1_min=0.95,
            acceptance_drop_max=0.02,
        )
        selected = select_by_context(rows)
        self.assertEqual(
            selected[1024]["config"], "target_k4v4__draft_k4v4"
        )

    def test_reports_each_failed_constraint(self) -> None:
        rows = evaluate_candidates(
            self.joint,
            self.quality,
            target_kl_max=0.01,
            target_delta_nll_max=0.02,
            target_top1_min=0.95,
            acceptance_drop_max=0.02,
        )
        bad = next(row for row in rows if row["target_config"] == "k2v2")
        self.assertFalse(bad["feasible"])
        self.assertEqual(
            set(bad["constraint_failures"].split(";")),
            {"target_kl", "target_delta_nll", "target_top1", "acceptance"},
        )


if __name__ == "__main__":
    unittest.main()
