import unittest

from aggregate_kivi_objective_preferences import aggregate_preferences


def spec_row(accept_rate: float, proposed_tokens: int = 100):
    return {
        "accept_rate": str(accept_rate),
        "accepted_tokens": str(accept_rate * proposed_tokens),
        "proposed_tokens": str(proposed_tokens),
        "matches_target_greedy": "1",
        "mismatch_min_top1_margin": "nan",
    }


def quality_row(kl: float, delta_nll: float):
    return {"kl_p_to_q": str(kl), "delta_nll": str(delta_nll)}


class KiviObjectivePreferenceTest(unittest.TestCase):
    def test_detects_opposite_objective_preferences(self) -> None:
        spec = {
            (1024, 0, "0"): {"k8v4": spec_row(0.8), "k4v8": spec_row(0.6)},
            (1024, 0, "1"): {"k8v4": spec_row(0.7), "k4v8": spec_row(0.6)},
        }
        quality = {
            (1024, 0, "0"): {"k8v4": quality_row(0.02, 0.03), "k4v8": quality_row(0.01, 0.01)},
            (1024, 0, "1"): {"k8v4": quality_row(0.03, 0.04), "k4v8": quality_row(0.01, 0.02)},
        }
        memory = {(1024, "k8v4"): [0.25], (1024, "k4v8"): [0.24]}

        rows = aggregate_preferences(
            spec_rows=spec,
            quality_rows=quality,
            memory=memory,
            pairs=[("k8v4", "k4v8")],
            tie_margin=1e-3,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["spec_preference"], "k8v4")
        self.assertEqual(rows[0]["quality_preference"], "k4v8")
        self.assertTrue(rows[0]["preference_reversal"])
        self.assertTrue(rows[0]["resolved_preference_reversal"])
        self.assertEqual(rows[0]["spec_preference_resolved"], "k8v4")
        self.assertEqual(rows[0]["quality_preference_resolved"], "k4v8")
        self.assertFalse(rows[0]["memory_matched"])
        self.assertAlmostEqual(rows[0]["absolute_total_saved_fraction_gap"], 0.01)
        self.assertAlmostEqual(rows[0]["spec_acceptance_a_minus_b_mean"], 0.15)

    def test_flags_an_equal_memory_comparison(self) -> None:
        spec = {(1024, 0, "0"): {"k8v4": spec_row(0.8), "k4v8": spec_row(0.7)}}
        quality = {
            (1024, 0, "0"): {
                "k8v4": quality_row(0.02, 0.03),
                "k4v8": quality_row(0.01, 0.02),
            }
        }
        memory = {(1024, "k8v4"): [0.25], (1024, "k4v8"): [0.249]}

        rows = aggregate_preferences(
            spec_rows=spec,
            quality_rows=quality,
            memory=memory,
            pairs=[("k8v4", "k4v8")],
            tie_margin=1e-3,
        )

        self.assertTrue(rows[0]["memory_matched"])

    def test_distinguishes_raw_from_unresolved_reversal(self) -> None:
        spec = {
            (1024, 0, "0"): {"k4v3": spec_row(0.52), "k3v4": spec_row(0.50)},
            (1024, 0, "1"): {"k4v3": spec_row(0.49), "k3v4": spec_row(0.50)},
        }
        quality = {
            (1024, 0, "0"): {"k4v3": quality_row(0.03, 0.03), "k3v4": quality_row(0.01, 0.01)},
            (1024, 0, "1"): {"k4v3": quality_row(0.04, 0.04), "k3v4": quality_row(0.01, 0.01)},
        }
        memory = {(1024, "k4v3"): [0.25], (1024, "k3v4"): [0.25]}

        rows = aggregate_preferences(
            spec_rows=spec,
            quality_rows=quality,
            memory=memory,
            pairs=[("k4v3", "k3v4")],
            tie_margin=1e-3,
        )

        self.assertTrue(rows[0]["preference_reversal"])
        self.assertEqual(rows[0]["spec_preference_resolved"], "unresolved")
        self.assertEqual(rows[0]["quality_preference_resolved"], "k3v4")
        self.assertFalse(rows[0]["resolved_preference_reversal"])

    def test_acceptance_delta_weights_by_proposed_tokens(self) -> None:
        spec = {
            (1024, 0, "0"): {
                "k8v4": spec_row(1.0, proposed_tokens=10),
                "k4v8": spec_row(0.0, proposed_tokens=10),
            },
            (1024, 0, "1"): {
                "k8v4": spec_row(0.0, proposed_tokens=90),
                "k4v8": spec_row(0.5, proposed_tokens=90),
            },
        }
        quality = {
            (1024, 0, "0"): {"k8v4": quality_row(0.02, 0.03), "k4v8": quality_row(0.01, 0.02)},
            (1024, 0, "1"): {"k8v4": quality_row(0.02, 0.03), "k4v8": quality_row(0.01, 0.02)},
        }
        memory = {(1024, "k8v4"): [0.25], (1024, "k4v8"): [0.25]}

        rows = aggregate_preferences(
            spec_rows=spec,
            quality_rows=quality,
            memory=memory,
            pairs=[("k8v4", "k4v8")],
            tie_margin=1e-3,
        )

        self.assertAlmostEqual(rows[0]["spec_acceptance_a_minus_b_mean"], -0.35)


if __name__ == "__main__":
    unittest.main()
