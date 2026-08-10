import csv
import json
import tempfile
import unittest
from pathlib import Path

from aggregate_kivi_cross_family_meta import (
    aggregate_configs,
    collect_pair_rows,
    summarize_preferences,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CrossFamilyMetaAggregationTest(unittest.TestCase):
    def test_macro_aggregation_uses_model_pairs_as_units(self) -> None:
        rows = [
            {
                "pair": "a",
                "context": 1024,
                "config": "k4v4",
                "total_cache_saved_fraction": 0.2,
                "draft_cache_saved_fraction": 0.66,
                "acceptance_delta_mean": -0.01,
                "acceptance_delta_ci_low": -0.015,
                "quality_kl_mean": 0.006,
                "quality_kl_ci_high": 0.008,
                "quality_top1_match_mean": 0.97,
            },
            {
                "pair": "b",
                "context": 1024,
                "config": "k4v4",
                "total_cache_saved_fraction": 0.4,
                "draft_cache_saved_fraction": 0.66,
                "acceptance_delta_mean": 0.01,
                "acceptance_delta_ci_low": -0.01,
                "quality_kl_mean": 0.008,
                "quality_kl_ci_high": 0.009,
                "quality_top1_match_mean": 0.96,
            },
        ]
        grouped = aggregate_configs(
            rows, acceptance_drop_budget=0.02, quality_kl_budget=0.01
        )
        self.assertEqual(len(grouped), 1)
        self.assertAlmostEqual(grouped[0]["acceptance_delta_macro_mean"], 0.0)
        self.assertAlmostEqual(
            grouped[0]["total_cache_saved_fraction_macro_mean"], 0.3
        )
        self.assertEqual(grouped[0]["num_pairs_jointly_conservative"], 2)

    def test_collects_exactness_and_requires_pair_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pair = "pair_a"
            spec = root / "spec" / pair / "aggregate" / "summary.json"
            quality = root / "quality" / pair / "aggregate" / "summary.json"
            spec.parent.mkdir(parents=True)
            quality.parent.mkdir(parents=True)
            spec.write_text(
                json.dumps(
                    {
                        "num_complete_runs": 2,
                        "exactness": {
                            "exact": 100,
                            "numerical_tie": 4,
                            "non_tie_or_unknown": 2,
                        },
                        "invalid_prompt_occurrences": 1,
                    }
                ),
                encoding="utf-8",
            )
            quality.write_text(
                json.dumps({"num_complete_runs": 2}), encoding="utf-8"
            )
            comparison = root / "comparison" / pair
            row = {
                "context": 1024,
                "config": "k4v4",
                "total_cache_saved_fraction": 0.3,
                "draft_cache_saved_fraction": 0.66,
                "acceptance_delta_mean": 0.0,
                "acceptance_delta_ci_low": -0.01,
                "acceptance_delta_ci_high": 0.01,
                "quality_kl_mean": 0.006,
                "quality_kl_ci_low": 0.005,
                "quality_kl_ci_high": 0.007,
                "quality_top1_match_mean": 0.97,
            }
            write_csv(comparison / "matched_objectives.csv", [row])
            pref = {
                "context": 1024,
                "memory_matched": True,
                "preference_reversal": False,
            }
            write_csv(comparison / "paired_preferences.csv", [pref])

            matched, preferences, audit = collect_pair_rows(root, [pair, "missing"])

            self.assertEqual(len(matched), 1)
            self.assertEqual(len(preferences), 1)
            self.assertEqual(audit["exactness"]["exact"], 100)
            self.assertEqual(audit["exactness"]["invalid_prompt_occurrences"], 1)
            self.assertTrue(audit["missing_artifacts"])

    def test_counts_only_memory_matched_reversals_in_primary_count(self) -> None:
        summary = summarize_preferences(
            [
                {"memory_matched": "True", "preference_reversal": "True"},
                {"memory_matched": "False", "preference_reversal": "True"},
                {"memory_matched": "True", "preference_reversal": "False"},
            ]
        )
        self.assertEqual(summary["num_preference_reversals"], 2)
        self.assertEqual(summary["num_memory_matched_preference_reversals"], 1)

    def test_supports_explicit_partial_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pair = "pair_a"
            for role in ("spec", "quality"):
                path = root / role / pair / "aggregate_partial" / "summary.json"
                path.parent.mkdir(parents=True)
                payload = {"num_complete_runs": 1}
                if role == "spec":
                    payload.update({"exactness": {"exact": 1}, "invalid_prompt_occurrences": 0})
                path.write_text(json.dumps(payload), encoding="utf-8")
            comparison = root / "comparison" / pair / "partial"
            write_csv(
                comparison / "matched_objectives.csv",
                [{"context": 1024, "config": "k4v4"}],
            )
            write_csv(
                comparison / "paired_preferences.csv",
                [{"memory_matched": True, "preference_reversal": False}],
            )

            matched, preferences, audit = collect_pair_rows(
                root,
                [pair],
                aggregate_name="aggregate_partial",
                comparison_name="partial",
            )

            self.assertEqual(len(matched), 1)
            self.assertEqual(len(preferences), 1)
            self.assertFalse(audit["missing_artifacts"])


if __name__ == "__main__":
    unittest.main()
