import csv
import json
import tempfile
import unittest
from pathlib import Path

from aggregate_kivi_cross_family_meta import collect_pair_rows


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
