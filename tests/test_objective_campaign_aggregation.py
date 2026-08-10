import json
import tempfile
import unittest
from pathlib import Path

from paper.aggregate_objective_campaign import collect_rows, discover_aggregates


def write_summary(root: Path, name: str, *, missing: int = 0) -> None:
    out_dir = root / name / "aggregate"
    out_dir.mkdir(parents=True)
    payload = {
        "num_missing_pairs": missing,
        "num_rejected_pairs": 0,
        "grouped": [
            {
                "budget": 6,
                "context": 1024,
                "allocation_objective": "quality",
                "total_cache_saved_fraction": 0.25,
            }
        ],
        "cross_context_effects": [
            {
                "budget": 6,
                "paired_acceptance_mean": 0.01,
                "paired_acceptance_ci_low": 0.0,
                "paired_acceptance_ci_high": 0.02,
            }
        ],
        "kv_priority_cross_context_effects": [
            {
                "budget": 6,
                "paired_acceptance_mean": 0.02,
                "paired_quality_kl_mean": 0.03,
            }
        ],
        "exactness_audit": {
            "totals": {
                "exact": 90,
                "numerical_tie": 9,
                "non_tie_or_unknown": 1,
                "invalid_prompts": 1,
            }
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


class ObjectiveCampaignAggregationTest(unittest.TestCase):
    def test_discovers_only_complete_non_smoke_matrices_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "complete")
            write_summary(root, "incomplete", missing=1)
            write_summary(root, "matrix_smoke")

            records, rejected = discover_aggregates(
                root,
                include_incomplete=False,
                include_smoke=False,
            )

            self.assertEqual([record["matrix"] for record in records], ["complete"])
            self.assertEqual([record["matrix"] for record in rejected], ["incomplete"])

    def test_collects_effects_savings_and_exactness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "complete")
            records, _ = discover_aggregates(root, include_incomplete=False, include_smoke=False)

            rows = collect_rows(records)

            self.assertEqual(rows["objective"][0]["total_cache_saved_fraction"], 0.25)
            self.assertEqual(rows["kv"][0]["paired_quality_kl_mean"], 0.03)
            self.assertAlmostEqual(rows["exactness"][0]["exact_or_tie_fraction"], 0.99)


if __name__ == "__main__":
    unittest.main()
