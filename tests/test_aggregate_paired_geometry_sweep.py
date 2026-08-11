import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aggregate_paired_geometry_sweep import allocation_contrast, collect, paired_values


class PairedGeometryAggregationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"seed": 0, "prompt_idx": 0, "treatment": 16, "config": "k8v4", "accept_rate": 0.8},
            {"seed": 0, "prompt_idx": 0, "treatment": 16, "config": "k4v8", "accept_rate": 0.5},
            {"seed": 0, "prompt_idx": 0, "treatment": 32, "config": "k8v4", "accept_rate": 0.6},
            {"seed": 0, "prompt_idx": 0, "treatment": 32, "config": "k4v8", "accept_rate": 0.7},
        ]

    def test_treatment_difference_is_prompt_paired(self) -> None:
        values = paired_values(
            self.rows,
            treatment=16,
            reference=32,
            config="k8v4",
            metric="accept_rate",
        )
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(values[0], 0.2)

    def test_allocation_contrast_is_prompt_paired(self) -> None:
        values = allocation_contrast(
            self.rows,
            treatment=16,
            config_a="k8v4",
            config_b="k4v8",
        )
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(values[0], 0.3)

    def test_rejects_different_prompt_offsets_within_seed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for treatment, offset in ((16, 100), (32, 200)):
                run_dir = root / f"group_{treatment}" / "seed_0"
                run_dir.mkdir(parents=True)
                summary = {
                    "runtime": {
                        "evaluator_version": "cached_dynamic_v6_sequential_target",
                        "target_verification_mode": "sequential",
                        "key_group_size": treatment,
                    },
                    "config": {"seed": 0, "skip_prompts": offset},
                    "num_prompts": 1,
                    "target_quant_configs": ["none"],
                    "draft_quant_configs": ["none"],
                    "summaries": {"none": {}},
                }
                (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
                with (run_dir / "benchmark_rows.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "config",
                            "prompt_idx",
                            "accept_rate",
                            "matches_target_greedy",
                            "first_target_mismatch",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "config": "none",
                            "prompt_idx": 0,
                            "accept_rate": 1.0,
                            "matches_target_greedy": 1.0,
                            "first_target_mismatch": -1,
                        }
                    )
            with self.assertRaisesRegex(ValueError, "do not share prompts"):
                collect(
                    root=root,
                    treatment_prefix="group",
                    treatment_config_key="key_group_size",
                    treatments=[16, 32],
                    seeds=[0],
                    expected_prompts=1,
                )


if __name__ == "__main__":
    unittest.main()
