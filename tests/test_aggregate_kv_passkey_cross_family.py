import json
import tempfile
import unittest
from pathlib import Path

from aggregate_kv_passkey_cross_family import (
    aggregate_contrasts,
    collect_rows,
    make_plot,
    parse_sources,
)


def make_summary(path: Path, *, complete: bool = True) -> None:
    payload = {
        "evaluator_version": "kv_multiple_choice_cached_v2",
        "task_generator_version": "synthetic_associative_passkey_v3",
        "expected_contexts": [8192],
        "expected_seeds": [0],
        "expected_num_choices": 16,
        "expected_passkey_variant": "confusable_records",
        "expected_passkey_score": "normalized",
        "num_complete_runs": 1,
        "missing_runs": [],
        "underfilled_runs": [],
        "complete_run_gate": complete,
        "grouped": [
            {"context": 8192, "depth": "all", "config": "k4v2", "accuracy_mean": 1.0}
        ],
        "comparisons": [
            {
                "context": 8192,
                "depth": "all",
                "config_a": "k4v2",
                "config_b": "k2v4",
                "accuracy_a_minus_b_mean": 0.25,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class PasskeyCrossFamilyTest(unittest.TestCase):
    def test_parse_sources_rejects_duplicates(self) -> None:
        with self.assertRaises(ValueError):
            parse_sources(["qwen=a.json", "qwen=b.json"])

    def test_collects_only_complete_strict_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            make_summary(path)
            accuracy, contrasts = collect_rows(
                {"qwen": path}, expected_contexts=[8192], expected_runs=1
            )
            self.assertEqual(len(accuracy), 1)
            self.assertEqual(len(contrasts), 1)

            make_summary(path, complete=False)
            with self.assertRaises(ValueError):
                collect_rows({"qwen": path}, expected_contexts=[8192], expected_runs=1)

    def test_macro_uses_models_as_units(self) -> None:
        rows = [
            {"model": "a", "context": 8192, "accuracy_a_minus_b_mean": 0.1},
            {"model": "b", "context": 8192, "accuracy_a_minus_b_mean": 0.3},
        ]
        output = aggregate_contrasts(rows, bootstrap_samples=1000, seed=1)
        self.assertEqual(output[0]["num_models"], 2)
        self.assertAlmostEqual(output[0]["k4v2_minus_k2v4_macro_mean"], 0.2)

    def test_writes_cross_family_figure(self) -> None:
        model_rows = [
            {"model": "a", "context": 8192, "accuracy_a_minus_b_mean": 0.1},
            {"model": "a", "context": 16384, "accuracy_a_minus_b_mean": 0.2},
        ]
        macro_rows = [
            {"context": 8192, "k4v2_minus_k2v4_macro_mean": 0.1},
            {"context": 16384, "k4v2_minus_k2v4_macro_mean": 0.2},
        ]
        with tempfile.TemporaryDirectory() as directory:
            paths = make_plot(model_rows, macro_rows, Path(directory))
            if paths:
                self.assertEqual(len(paths), 2)
                self.assertTrue(all(Path(path).exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
