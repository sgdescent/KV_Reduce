import json
import tempfile
import unittest
from pathlib import Path

from aggregate_kv_multiple_choice_cross_family import (
    collect_summaries,
    complete_plot_models,
)


class CrossFamilyMultipleChoiceAggregationTest(unittest.TestCase):
    def test_collects_valid_models_and_propagates_underfill(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "summary.json"
            valid.write_text(
                json.dumps(
                    {
                        "evaluator_version": "kv_multiple_choice_cached_v2",
                        "grouped": [{"task": "hellaswag", "config": "k4v4"}],
                        "comparisons": [{"task": "hellaswag", "config_a": "k8v4"}],
                        "underfilled_runs": [{"actual": 7, "requested": 8}],
                    }
                ),
                encoding="utf-8",
            )
            missing = Path(tmp) / "missing.json"

            grouped, comparisons, underfilled, rejected = collect_summaries(
                [("llama32_3b", valid), ("olmo2_1b", missing)]
            )

            self.assertEqual(grouped[0]["model_label"], "Llama-3.2-3B")
            self.assertEqual(comparisons[0]["model"], "llama32_3b")
            self.assertEqual(underfilled[0]["model"], "llama32_3b")
            self.assertEqual(rejected[0]["reason"], "missing")

    def test_plot_skips_models_with_incomplete_config_grid(self):
        rows = [
            {
                "model": "llama32_3b",
                "task": "hellaswag",
                "config": config,
                "paired_delta_vs_bf16_mean": 0.0,
                "paired_delta_vs_bf16_ci_low": -0.01,
                "paired_delta_vs_bf16_ci_high": 0.01,
            }
            for config in ("k8v4", "k4v8", "k4v4")
        ]
        rows.append(
            {
                "model": "olmo2_1b",
                "task": "hellaswag",
                "config": "k4v4",
                "paired_delta_vs_bf16_mean": 0.0,
                "paired_delta_vs_bf16_ci_low": -0.01,
                "paired_delta_vs_bf16_ci_high": 0.01,
            }
        )
        complete = complete_plot_models(
            rows,
            "hellaswag",
            ("k8v4", "k4v8", "k4v4"),
            ("llama32_3b", "olmo2_1b"),
        )
        self.assertEqual(complete, ["llama32_3b"])


if __name__ == "__main__":
    unittest.main()
