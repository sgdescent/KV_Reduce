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


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path: Path, rows) -> None:
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
            write_json(
                spec,
                {
                    "num_complete_runs": 2,
                    "exactness": {
                        "exact": 100,
                        "numerical_tie": 4,
                        "non_tie_or_unknown": 2,
                    },
                    "invalid_prompt_occurrences": 1,
                },
            )
            write_json(quality, {"num_complete_runs": 2})
            comparison = root / "comparison" / pair
            write_csv(
                comparison / "matched_objectives.csv",
                [
                    {
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
                ],
            )
            write_csv(
                comparison / "paired_preferences.csv",
                [
                    {
                        "context": 1024,
                        "memory_matched": True,
                        "preference_reversal": False,
                    }
                ],
            )

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
                payload = {"num_complete_runs": 1}
                if role == "spec":
                    payload.update(
                        {"exactness": {"exact": 1}, "invalid_prompt_occurrences": 0}
                    )
                write_json(path, payload)
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


class KiviCrossFamilyPerPairTest(unittest.TestCase):
    def make_pair(self, root: Path, pair: str, *, exact_target: bool = True) -> None:
        pair_root = root / pair
        write_json(
            pair_root / "spec_aggregate" / "summary.json",
            {
                "num_complete_runs": 6,
                "integrity_gates": {
                    "complete": True,
                    "config_coverage": True,
                    "full_runs": True,
                    "exact_target": exact_target,
                },
                "exactness": {"exact": 100},
                "invalid_prompt_occurrences": 0,
            },
        )
        write_json(
            pair_root / "quality_aggregate" / "summary.json",
            {"num_complete_runs": 6, "full_run_gate": True},
        )
        comparison = pair_root / "objective_comparison"
        write_json(comparison / "preference_summary.json", {"all_runs_filled": True})
        write_csv(
            comparison / "matched_objectives.csv",
            [
                {
                    "context": 1024,
                    "config": "k4v2",
                    "acceptance_delta_mean": 0.01,
                    "quality_kl_mean": 0.02,
                }
            ],
        )
        write_csv(
            comparison / "paired_preferences.csv",
            [
                {
                    "context": 1024,
                    "config_a": "k4v2",
                    "config_b": "k2v4",
                    "memory_matched": True,
                    "preference_reversal": True,
                    "resolved_preference_reversal": False,
                }
            ],
        )

    def test_collects_strict_per_pair_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_pair(root, "smol")

            matched, preferences, audit = collect_pair_rows(
                root, ["smol"], layout="per_pair", require_integrity=True
            )

            self.assertEqual(len(matched), 1)
            self.assertEqual(len(preferences), 1)
            self.assertEqual(audit["complete_pairs"], ["smol"])
            self.assertEqual(audit["integrity_failures"], [])
            self.assertEqual(audit["exactness"], {"exact": 100, "invalid_prompt_occurrences": 0})

    def test_rejects_failed_strict_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_pair(root, "smol", exact_target=False)

            matched, preferences, audit = collect_pair_rows(
                root, ["smol"], layout="per_pair", require_integrity=True
            )

            self.assertEqual(matched, [])
            self.assertEqual(preferences, [])
            self.assertEqual(
                audit["integrity_failures"],
                [{"pair": "smol", "failures": ["spec:exact_target"]}],
            )


if __name__ == "__main__":
    unittest.main()
