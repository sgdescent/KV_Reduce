import csv
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from aggregate_objective_kv_matrix import (
    audit_cell_coverage,
    hierarchical_bootstrap_mean_ci,
    load_manifest_cells,
)
from prepare_objective_kv_matrix import matched_byte_target


REPO_ROOT = Path(__file__).resolve().parents[1]


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

    def test_byte_target_uses_less_compressed_nominal_baseline(self) -> None:
        with TemporaryDirectory() as tmp:
            profile = Path(tmp) / "profile.csv"
            rows = []
            savings = {
                ("k", 8): 1.0,
                ("k", 4): 2.0,
                ("v", 8): 1.5,
                ("v", 4): 3.0,
            }
            for layer in (0, 1):
                for (component, bits), saved_mib in savings.items():
                    rows.append(
                        {
                            "candidate": f"layer{layer}_{component}{bits}",
                            "layer": layer,
                            "component": component,
                            "bits": bits,
                            "cache_mib_saved": saved_mib,
                        }
                    )
            with profile.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            target, baselines = matched_byte_target(
                str(profile),
                profiled_layers=[0, 1],
                budget=6,
            )

        self.assertEqual(target, 7.0 * 1024.0**2)
        self.assertEqual(baselines["k_priority"], 8.0 * 1024.0**2)
        self.assertEqual(baselines["v_priority"], 7.0 * 1024.0**2)

    @unittest.skipUnless(
        importlib.util.find_spec("transformers") is not None,
        "allocator imports the experiment dependency stack",
    )
    def test_allocator_matches_metadata_aware_byte_target(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = root / "profile.csv"
            out_dir = root / "allocation"
            rows = [
                {
                    "candidate": "layer0_k8",
                    "layer": 0,
                    "component": "k",
                    "bits": 8,
                    "risk": 0.10,
                    "cache_mib_saved": 100.0 / 1024.0**2,
                },
                {
                    "candidate": "layer0_k4",
                    "layer": 0,
                    "component": "k",
                    "bits": 4,
                    "risk": 0.50,
                    "cache_mib_saved": 200.0 / 1024.0**2,
                },
                {
                    "candidate": "layer0_v8",
                    "layer": 0,
                    "component": "v",
                    "bits": 8,
                    "risk": 0.10,
                    "cache_mib_saved": 100.0 / 1024.0**2,
                },
                {
                    "candidate": "layer0_v4",
                    "layer": 0,
                    "component": "v",
                    "bits": 4,
                    "risk": 0.05,
                    "cache_mib_saved": 200.0 / 1024.0**2,
                },
            ]
            with profile.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "search_kv_bit_allocation.py"),
                    "--profile_csv",
                    str(profile),
                    "--risk_field",
                    "risk",
                    "--target_profiled_saved_bytes",
                    "300",
                    "--out_dir",
                    str(out_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            allocation = json.loads((out_dir / "allocation.json").read_text())

        self.assertEqual(allocation["achieved_profiled_saved_bytes"], 300.0)
        self.assertEqual(allocation["k_bits"], [8])
        self.assertEqual(allocation["v_bits"], [4])

    @unittest.skipUnless(
        importlib.util.find_spec("transformers") is not None,
        "preparation invokes the experiment dependency stack",
    )
    def test_aggressive_preparation_matches_bytes_end_to_end(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            quality_profile = root / "quality.csv"
            acceptance_profile = root / "acceptance.csv"
            out_dir = root / "matrix"
            rows = []
            saved_bytes = {
                ("k", 8): 80,
                ("k", 4): 120,
                ("k", 2): 140,
                ("v", 8): 70,
                ("v", 4): 110,
                ("v", 2): 130,
            }
            for (component, bits), saved in saved_bytes.items():
                rows.append(
                    {
                        "candidate": f"layer0_{component}{bits}",
                        "layer": 0,
                        "component": component,
                        "bits": bits,
                        "quality_risk": (16 - bits) * (1.0 if component == "k" else 0.5),
                        "acceptance_risk": (16 - bits) * (0.5 if component == "k" else 1.0),
                        "cache_mib_saved": saved / 1024.0**2,
                    }
                )
            for path in (quality_profile, acceptance_profile):
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "prepare_objective_kv_matrix.py"),
                    "--quality_profile_csv",
                    str(quality_profile),
                    "--acceptance_profile_csv",
                    str(acceptance_profile),
                    "--quality_risk_field",
                    "quality_risk",
                    "--acceptance_risk_field",
                    "acceptance_risk",
                    "--num_layers",
                    "1",
                    "--budgets",
                    "3,5",
                    "--contexts",
                    "128",
                    "--seeds",
                    "0",
                    "--num_eval",
                    "1",
                    "--allowed_bits",
                    "2,4,8",
                    "--out_dir",
                    str(out_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for budget in (3, 5):
                budget_root = out_dir / f"budget_{budget}"
                quality = json.loads(
                    (budget_root / "quality_allocation" / "allocation.json").read_text()
                )
                acceptance = json.loads(
                    (budget_root / "acceptance_allocation" / "allocation.json").read_text()
                )
                self.assertEqual(
                    quality["achieved_profiled_saved_bytes"],
                    acceptance["achieved_profiled_saved_bytes"],
                )
                self.assertIn(2, quality["allowed_bits"])
            with (out_dir / "manifest.tsv").open(encoding="utf-8") as handle:
                manifest_rows = list(csv.DictReader(handle, delimiter="\t"))

        self.assertEqual(len(manifest_rows), 4)


if __name__ == "__main__":
    unittest.main()
