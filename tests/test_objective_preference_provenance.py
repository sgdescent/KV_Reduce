import csv
import json
import tempfile
import unittest
from pathlib import Path

from aggregate_kivi_objective_preferences import load_quality_rows, load_spec_rows


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ObjectivePreferenceProvenanceTest(unittest.TestCase):
    def test_loaders_report_underfilled_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = root / "spec" / "ctx_1024" / "seed_0"
            quality = root / "quality" / "ctx_1024" / "seed_0"
            spec.mkdir(parents=True)
            quality.mkdir(parents=True)
            (spec / "summary.json").write_text(
                json.dumps(
                    {
                        "runtime": {
                            "evaluator_version": "cached_dynamic_v6_sequential_target"
                        },
                        "config": {"prompt_len": 1024, "seed": 0, "num_prompts": 2},
                        "summaries": {"none": {"total_cache_saved_fraction": 0.0}},
                    }
                ),
                encoding="utf-8",
            )
            write_csv(
                spec / "benchmark_rows.csv",
                [{"prompt_idx": "0", "config": "none"}],
            )
            (quality / "summary.json").write_text(
                json.dumps(
                    {
                        "runtime": {"evaluator_version": "teacher_forced_cached_v1"},
                        "config": {"prompt_len": 1024, "seed": 0, "num_sequences": 2},
                    }
                ),
                encoding="utf-8",
            )
            write_csv(
                quality / "raw_sequence_rows.csv",
                [{"sequence_idx": "0", "candidate": "none"}],
            )

            _spec_rows, _memory, spec_underfilled = load_spec_rows(root / "spec")
            _quality_rows, quality_underfilled = load_quality_rows(root / "quality")

            self.assertEqual(spec_underfilled[0]["shortfall"], 1)
            self.assertEqual(quality_underfilled[0]["shortfall"], 1)


if __name__ == "__main__":
    unittest.main()
