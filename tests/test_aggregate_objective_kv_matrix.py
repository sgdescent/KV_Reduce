import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aggregate_objective_kv_matrix import (
    audit_cell_coverage,
    hierarchical_bootstrap_mean_ci,
    load_manifest_cells,
)


class ObjectiveMatrixAggregationTest(unittest.TestCase):
    def test_hierarchical_bootstrap_tracks_run_clusters(self) -> None:
        estimate, low, high, clusters = hierarchical_bootstrap_mean_ci(
            {(1024, 0): [0.1, 0.2], (4096, 0): [0.3, 0.4]},
            seed=7,
            samples=500,
        )

        self.assertEqual(clusters, 2)
        self.assertAlmostEqual(estimate, 0.25)
        self.assertLessEqual(low, estimate)
        self.assertGreaterEqual(high, estimate)

    def test_manifest_gate_rejects_incomplete_objective_pair(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "objective",
                        "budget",
                        "context",
                        "seed",
                        "num_eval",
                        "skip_blocks",
                    ],
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "objective": "quality",
                        "budget": 6,
                        "context": 1024,
                        "seed": 0,
                        "num_eval": 32,
                        "skip_blocks": 100,
                    }
                )

            with self.assertRaisesRegex(ValueError, "incomplete objective pairs"):
                load_manifest_cells(root)

    def test_cell_coverage_gate_detects_underfilled_artifacts(self) -> None:
        names = {"none": {}, "quality_b6": {}, "acceptance_b6": {}}
        issues = audit_cell_coverage(
            quality_eval={"num_sequences": 1, "summaries": names},
            acceptance_eval={
                "num_prompts": 2,
                "summaries": names,
                "target_quant_configs": ["none"],
            },
            quality_rows=[{"candidate": name, "sequence_idx": "0"} for name in names],
            acceptance_rows=[
                {"config": name, "prompt_idx": str(prompt_idx)}
                for name in names
                for prompt_idx in range(2)
            ],
            expected_num_eval=2,
            tracked_names=["quality_b6", "acceptance_b6"],
        )

        self.assertTrue(any("num_sequences" in issue for issue in issues))
        self.assertTrue(any("quality row count" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
