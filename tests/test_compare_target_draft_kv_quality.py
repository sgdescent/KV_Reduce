import json
import tempfile
import unittest
from pathlib import Path

from compare_target_draft_kv_quality import (
    aggregate_role_differences,
    collect_role_rows,
)


def write_quality_summary(path: Path, *, kl: float, top1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "num_complete_runs": 2,
                "grouped": [
                    {
                        "context": 1024,
                        "config": "k4v4",
                        "k_bits": 4,
                        "v_bits": 4,
                        "kl_p_to_q_mean": kl,
                        "kl_p_to_q_ci_low": kl - 0.001,
                        "kl_p_to_q_ci_high": kl + 0.001,
                        "delta_nll_mean": kl / 2,
                        "top1_match_mean": top1,
                        "cache_saved_fraction": 0.66,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class TargetDraftRoleQualityTest(unittest.TestCase):
    def test_collects_matched_role_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pair = "pair_a"
            write_quality_summary(
                root / "quality" / pair / "aggregate" / "summary.json",
                kl=0.006,
                top1=0.97,
            )
            write_quality_summary(
                root / "target_quality" / pair / "aggregate" / "summary.json",
                kl=0.009,
                top1=0.95,
            )

            rows, audit = collect_role_rows(root, [pair])

            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["target_minus_draft_kl"], 0.003)
            self.assertEqual(audit["run_counts"][pair], {"draft": 2, "target": 2})

    def test_macro_difference_uses_one_observation_per_pair(self) -> None:
        rows = [
            {
                "pair": "a",
                "context": 1024,
                "config": "k4v4",
                "cache_saved_fraction": 0.66,
                "target_kl_mean": 0.01,
                "draft_kl_mean": 0.006,
                "target_minus_draft_kl": 0.004,
                "target_top1_match": 0.95,
                "draft_top1_match": 0.97,
            },
            {
                "pair": "b",
                "context": 1024,
                "config": "k4v4",
                "cache_saved_fraction": 0.66,
                "target_kl_mean": 0.004,
                "draft_kl_mean": 0.006,
                "target_minus_draft_kl": -0.002,
                "target_top1_match": 0.98,
                "draft_top1_match": 0.97,
            },
        ]

        grouped = aggregate_role_differences(rows)

        self.assertEqual(len(grouped), 1)
        self.assertAlmostEqual(grouped[0]["target_kl_macro_mean"], 0.007)
        self.assertAlmostEqual(
            grouped[0]["target_minus_draft_kl_macro_mean"], 0.001
        )
        self.assertEqual(grouped[0]["num_pairs_target_more_sensitive"], 1)


if __name__ == "__main__":
    unittest.main()
