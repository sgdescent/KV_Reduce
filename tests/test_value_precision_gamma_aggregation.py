import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_run(root: Path, evaluator_version: str) -> None:
    run_dir = root / "gamma_4" / "seed_0"
    run_dir.mkdir(parents=True)
    summary = {
        "runtime": {"evaluator_version": evaluator_version},
        "config": {"draft_steps": 4, "seed": 0},
        "quant_configs": ["none", "k4v4"],
        "summaries": {
            "none": {
                "overall_accept_rate": 0.5,
                "accepted_per_round": 2.0,
                "total_cache_saved_fraction": 0.0,
            },
            "k4v4": {
                "overall_accept_rate": 0.49,
                "accepted_per_round": 1.96,
                "total_cache_saved_fraction": 0.25,
            },
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (run_dir / "benchmark_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config",
                "prompt_idx",
                "accept_rate",
                "accepted_tokens",
                "proposed_tokens",
                "matches_target_greedy",
                "mismatch_min_top1_margin",
                "mismatch_source",
            ],
        )
        writer.writeheader()
        for name, rate, accepted in (("none", 0.5, 50), ("k4v4", 0.49, 49)):
            writer.writerow(
                {
                    "config": name,
                    "prompt_idx": 0,
                    "accept_rate": rate,
                    "accepted_tokens": accepted,
                    "proposed_tokens": 100,
                    "matches_target_greedy": 1,
                    "mismatch_min_top1_margin": "nan",
                    "mismatch_source": "",
                }
            )


class ValuePrecisionGammaAggregationTest(unittest.TestCase):
    def run_aggregate(self, sweep_dir: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "aggregate_value_precision_gamma_sweep.py"),
                "--sweep_dir",
                str(sweep_dir),
                "--out_dir",
                str(out_dir),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_accepts_only_current_evaluator_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sweep_dir = Path(tmp) / "sweep"
            out_dir = Path(tmp) / "aggregate"
            write_run(sweep_dir, "cached_dynamic_v4")

            result = self.run_aggregate(sweep_dir, out_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["evaluator_version"], "cached_dynamic_v4")
            self.assertEqual(payload["num_complete_runs"], 1)

    def test_rejects_older_evaluator_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sweep_dir = Path(tmp) / "sweep"
            out_dir = Path(tmp) / "aggregate"
            write_run(sweep_dir, "cached_dynamic_v3")

            result = self.run_aggregate(sweep_dir, out_dir)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Stale evaluator 'cached_dynamic_v3'", result.stderr)


if __name__ == "__main__":
    unittest.main()
