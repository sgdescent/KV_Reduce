import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aggregate_kv_passkey import main as aggregate_main
from aggregate_kv_passkey import parse_comparison_pairs
from select_passkey_layer_policy import select_policy


class PasskeyLayerPolicyTest(unittest.TestCase):
    def test_selects_smallest_upper_confidence_harm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = {
                "name": "selected_k4v4",
                "k_bits": [16, 16, 4, 4],
                "v_bits": [16, 16, 4, 4],
            }
            (root / "baseline.json").write_text(json.dumps(baseline))
            manifest = {
                "selected_layers": [2, 3],
                "reduced_bits": 2,
                "baseline_path": str(root / "baseline.json"),
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            summary = {
                "complete_gate": True,
                "selected_layers": [2, 3],
                "source_index_ranges": [[192, 223]],
                "layer_results": [
                    {
                        "layer": 2,
                        "k_harm_mean": 0.1,
                        "k_harm_ci_low": 0.0,
                        "k_harm_ci_high": 0.2,
                        "v_harm_mean": 0.0,
                        "v_harm_ci_low": -0.1,
                        "v_harm_ci_high": 0.1,
                    },
                    {
                        "layer": 3,
                        "k_harm_mean": -0.1,
                        "k_harm_ci_low": -0.2,
                        "k_harm_ci_high": 0.0,
                        "v_harm_mean": 0.2,
                        "v_harm_ci_low": 0.1,
                        "v_harm_ci_high": 0.3,
                    },
                ],
            }
            (root / "summary.json").write_text(json.dumps(summary))
            result = select_policy(
                sensitivity_summary=root / "summary.json",
                sensitivity_manifest=root / "manifest.json",
                num_reductions=2,
                out_dir=root / "policy",
            )
            chosen = [(row["layer"], row["component"]) for row in result["chosen_reductions"]]
            self.assertEqual(chosen, [(3, "k"), (2, "v")])
            learned = json.loads((root / "policy" / "retrieval_aware_mixed.json").read_text())
            self.assertEqual(learned["k_bits"], [16, 16, 4, 2])
            self.assertEqual(learned["v_bits"], [16, 16, 2, 4])

    def test_custom_comparison_pair_parser(self) -> None:
        self.assertEqual(
            parse_comparison_pairs("learned:k2v4;learned:k4v2"),
            [("learned", "k2v4"), ("learned", "k4v2")],
        )
        with self.assertRaises(ValueError):
            parse_comparison_pairs("missing_separator")

    def test_aggregate_rejects_profile_split_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "ctx_16384" / "seed_0"
            run_dir.mkdir(parents=True)
            summary = {
                "runtime": {
                    "evaluator_version": "kv_multiple_choice_cached_v2",
                    "task_generator_version": "synthetic_associative_passkey_v3",
                },
                "task": "passkey",
                "primary_metric": "raw_accuracy",
                "num_examples": 1,
                "source_index_range": [287, 287],
                "config": {
                    "max_prompt_tokens": 16384,
                    "passkey_num_choices": 16,
                    "passkey_variant": "confusable_records",
                },
                "summaries": {"none": {"cache_saved_fraction": 0.0}},
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "example_rows.csv").write_text(
                "seed,source_idx,config,prompt_tokens,passkey_depth,raw_correct\n"
                "0,287,none,16384,0.5,1\n"
            )
            argv = [
                "aggregate_kv_passkey.py",
                "--root", str(root),
                "--out_dir", str(root / "aggregate"),
                "--expected_contexts", "16384",
                "--expected_seeds", "0",
                "--expected_examples_per_run", "1",
                "--expected_num_choices", "16",
                "--expected_generator_version", "synthetic_associative_passkey_v3",
                "--expected_passkey_variant", "confusable_records",
                "--minimum_source_index", "288",
                "--require_complete",
            ]
            with patch("sys.argv", argv), self.assertRaisesRegex(ValueError, "held-out boundary"):
                aggregate_main()


if __name__ == "__main__":
    unittest.main()
