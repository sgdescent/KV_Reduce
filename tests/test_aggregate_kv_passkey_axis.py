import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "aggregate_kv_passkey_axis.py"
CONFIGS = ("none", "k4v4", "k4v2", "k2v4", "k2v2")


class AggregatePasskeyAxisTest(unittest.TestCase):
    def write_run(self, root: Path, axis: str, seed: int, *, bad_axis: bool = False) -> None:
        run_dir = root / "ctx_1024" / f"seed_{seed}"
        run_dir.mkdir(parents=True)
        summaries = {
            config: {"cache_saved_fraction": 0.7 if config != "none" else 0.0}
            for config in CONFIGS
        }
        summary = {
            "task": "passkey",
            "primary_metric": "normalized_accuracy",
            "num_examples": 2,
            "runtime": {
                "evaluator_version": "kv_multiple_choice_cached_v2",
                "task_generator_version": "synthetic_associative_passkey_v3",
                "key_quant_axis": "wrong" if bad_axis else axis,
            },
            "config": {
                "max_prompt_tokens": 1024,
                "passkey_num_choices": 16,
                "passkey_variant": "confusable_records",
                "passkey_score": "normalized",
            },
            "summaries": summaries,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        with (run_dir / "example_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("seed", "source_idx", "config", "normalized_correct"),
            )
            writer.writeheader()
            for example in range(2):
                for config in CONFIGS:
                    correct = 1.0
                    if axis == "per_channel" and config == "k2v4":
                        correct = 0.0
                    if axis == "per_token" and config == "k4v2":
                        correct = 0.0
                    writer.writerow(
                        {
                            "seed": seed,
                            "source_idx": seed * 2 + example,
                            "config": config,
                            "normalized_correct": correct,
                        }
                    )

    def run_aggregate(self, per_channel: Path, per_token: Path, out_dir: Path):
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--per_channel_root",
                str(per_channel),
                "--per_token_root",
                str(per_token),
                "--out_dir",
                str(out_dir),
                "--context",
                "1024",
                "--seeds",
                "0,1",
                "--expected_examples_per_run",
                "2",
                "--bootstrap_samples",
                "100",
                "--require_complete",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

    def test_reports_paired_axis_interaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_channel = root / "per_channel"
            per_token = root / "per_token"
            for seed in (0, 1):
                self.write_run(per_channel, "per_channel", seed)
                self.write_run(per_token, "per_token", seed)
            out_dir = root / "aggregate"
            result = self.run_aggregate(per_channel, per_token, out_dir)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["complete_run_gate"])
            self.assertEqual(summary["paired_count"], 4)
            contrasts = {row["axis"]: row for row in summary["contrasts"]}
            self.assertEqual(contrasts["per_channel"]["mean"], 1.0)
            self.assertEqual(contrasts["per_token"]["mean"], -1.0)
            self.assertEqual(contrasts["per_channel_minus_per_token"]["mean"], 2.0)

    def test_rejects_axis_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            per_channel = root / "per_channel"
            per_token = root / "per_token"
            for seed in (0, 1):
                self.write_run(per_channel, "per_channel", seed)
                self.write_run(per_token, "per_token", seed, bad_axis=(seed == 1))
            result = self.run_aggregate(per_channel, per_token, root / "aggregate")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Integrity check failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
