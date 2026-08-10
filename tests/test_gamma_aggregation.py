import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class GammaAggregationTest(unittest.TestCase):
    def test_reports_paired_native_effects_and_exactness_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "matrix"
            result_dir = root / "budget_6" / "ctx_1024" / "gamma_4" / "seed_0"
            result_dir.mkdir(parents=True)
            roles = {
                "none": 0.50,
                "quality_b6": 0.48,
                "acceptance_b6": 0.52,
                "k_priority_b6": 0.51,
                "v_priority_b6": 0.47,
            }
            summary = {
                "runtime": {"evaluator_version": "cached_dynamic_v4"},
                "summaries": {
                    name: {
                        "overall_accept_rate": value,
                        "accepted_per_round": value * 4,
                        "full_accept_round_fraction": value,
                        "round_js": 0.1,
                        "draft_cache_saved_fraction": 0.5 if name != "none" else 0.0,
                        "total_cache_saved_fraction": 0.25 if name != "none" else 0.0,
                    }
                    for name, value in roles.items()
                },
            }
            (result_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with (result_dir / "benchmark_rows.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "config",
                        "prompt_idx",
                        "accept_rate",
                        "matches_target_greedy",
                        "mismatch_min_top1_margin",
                        "mismatch_source",
                    ],
                )
                writer.writeheader()
                for name, value in roles.items():
                    writer.writerow(
                        {
                            "config": name,
                            "prompt_idx": 0,
                            "accept_rate": value,
                            "matches_target_greedy": 1,
                            "mismatch_min_top1_margin": "nan",
                            "mismatch_source": "",
                        }
                    )
                writer.writerow(
                    {
                        "config": "none",
                        "prompt_idx": 1,
                        "accept_rate": 0.5,
                        "matches_target_greedy": 0,
                        "mismatch_min_top1_margin": 0.1,
                        "mismatch_source": "target_correction",
                    }
                )

            out_dir = Path(tmp) / "aggregate"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "aggregate_spec_kv_gamma_matrix.py"),
                    "--matrix_dir",
                    str(root),
                    "--out_dir",
                    str(out_dir),
                ],
                check=True,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )
            payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            native = {row["config_role"]: row for row in payload["native_effects"]}

            self.assertAlmostEqual(native["quality"]["acceptance_delta_vs_native_mean"], -0.02)
            self.assertAlmostEqual(native["acceptance"]["acceptance_delta_vs_native_mean"], 0.02)
            self.assertEqual(payload["exactness"]["non_tie_or_unknown"], 1)
            self.assertEqual(payload["exactness_examples"][0]["prompt_idx"], 1)


if __name__ == "__main__":
    unittest.main()
