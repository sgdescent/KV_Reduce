import unittest
from pathlib import Path

from aggregate_kv_multiple_choice import paired_metric_differences, underfilled_run_record


class MultipleChoiceAggregationTest(unittest.TestCase):
    def test_pairs_examples_within_task_and_seed(self):
        rows = [
            {"task": "arc", "seed": "0", "source_idx": "5", "config": "none", "raw_correct": "1"},
            {"task": "arc", "seed": "0", "source_idx": "5", "config": "k4v4", "raw_correct": "0"},
            {"task": "arc", "seed": "1", "source_idx": "5", "config": "none", "raw_correct": "0"},
            {"task": "arc", "seed": "1", "source_idx": "5", "config": "k4v4", "raw_correct": "1"},
        ]
        self.assertEqual(
            paired_metric_differences(
                rows,
                config="k4v4",
                baseline="none",
                metric="raw_correct",
            ),
            [-1.0, 1.0],
        )

    def test_ignores_unpaired_examples(self):
        rows = [
            {"task": "arc", "seed": "0", "source_idx": "5", "config": "none", "raw_correct": "1"},
            {"task": "arc", "seed": "0", "source_idx": "6", "config": "k4v4", "raw_correct": "0"},
        ]
        self.assertEqual(
            paired_metric_differences(
                rows,
                config="k4v4",
                baseline="none",
                metric="raw_correct",
            ),
            [],
        )

    def test_reports_dataset_exhaustion(self):
        summary = {
            "task": "arc_challenge",
            "num_examples": 148,
            "config": {"num_examples": 256, "seed": 4},
        }
        record = underfilled_run_record(Path("seed_4/summary.json"), summary)
        self.assertEqual(record["actual"], 148)
        self.assertEqual(record["shortfall"], 108)

        summary["num_examples"] = 256
        self.assertIsNone(
            underfilled_run_record(Path("seed_4/summary.json"), summary)
        )


if __name__ == "__main__":
    unittest.main()
