import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from aggregate_fused_packed_systems import collect, geometric_mean, select_best_rows


class FusedPackedSystemsAggregationTest(unittest.TestCase):
    def test_selects_fastest_kernel_per_serving_cell(self) -> None:
        rows = [
            {
                "shape": "qwen",
                "batch_size": 1,
                "context": 4096,
                "config": "k4v4",
                "fused_decode_median_ms": 2.0,
            },
            {
                "shape": "qwen",
                "batch_size": 1,
                "context": 4096,
                "config": "k4v4",
                "fused_decode_median_ms": 1.0,
            },
        ]
        best = select_best_rows(rows)
        self.assertEqual(len(best), 1)
        self.assertEqual(best[0]["fused_decode_median_ms"], 1.0)

    def test_geometric_mean(self) -> None:
        self.assertAlmostEqual(geometric_mean([1.0, 4.0]), 2.0)

    def test_collect_preserves_kernel_error_schema(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "qwen25_15b" / "batch_1" / "summary.json"
            summary.parent.mkdir(parents=True)
            summary.write_text(
                json.dumps(
                    {
                        "runtime": {
                            "evaluator_version": "packed_fused_attention_v1",
                            "storage_mode": "actual_bit_packed_uint8_payloads",
                        },
                        "config": {
                            "batch_size": 1,
                            "query_heads": 12,
                            "kv_heads": 2,
                            "head_dim": 128,
                            "num_layers": 28,
                        },
                        "rows": [
                            {
                                "context": 1024,
                                "config": "k4v4",
                                "kernel_output_max_abs_error": 0.01,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows, sources = collect(root, require_complete=False)
            self.assertEqual(len(sources), 1)
            self.assertEqual(rows[0]["kernel_output_max_abs_error"], 0.01)


if __name__ == "__main__":
    unittest.main()
