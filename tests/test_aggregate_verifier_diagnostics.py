import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aggregate_verifier_diagnostics import COUNT_FIELDS, MAX_FIELDS, main


def diagnostic(dtype: str, speculative_mismatches: int, *, num_prompts: int = 1) -> dict:
    payload = {
        "config": {
            "dtype": dtype,
            "attn_implementation": "sdpa",
            "seed": 0,
            "skip_prompts": 3,
        },
    }
    payload.update({field: 0 for field in COUNT_FIELDS})
    payload.update({field: 0.0 for field in MAX_FIELDS})
    payload["num_prompts"] = num_prompts
    payload["speculative_top1_mismatches"] = speculative_mismatches
    payload["speculative_independent_greedy_mismatches"] = speculative_mismatches
    payload["speculative_prompts_with_top1_mismatch"] = speculative_mismatches
    payload["speculative_prompts_with_independent_greedy_mismatch"] = speculative_mismatches
    return payload


class VerifierDiagnosticAggregationTest(unittest.TestCase):
    def run_aggregate(self, root: Path) -> dict:
        out_dir = root / "aggregate"
        with patch(
            "sys.argv",
            [
                "aggregate_verifier_diagnostics.py",
                "--inputs",
                str(root / "*.json"),
                "--out_dir",
                str(out_dir),
            ],
        ):
            main()
        return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    def test_groups_numerical_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dtype, mismatches in (("bf16", 1), ("float32", 0)):
                (root / f"{dtype}.json").write_text(
                    json.dumps(diagnostic(dtype, mismatches)), encoding="utf-8"
                )

            result = self.run_aggregate(root)

            grouped = {row["dtype"]: row for row in result["grouped"]}
            self.assertEqual(grouped["bf16"]["prompts_with_speculative_mismatch"], 1)
            self.assertEqual(grouped["float32"]["prompts_with_speculative_mismatch"], 0)

    def test_counts_prompts_within_batched_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bf16.json").write_text(
                json.dumps(diagnostic("bf16", 3, num_prompts=32)), encoding="utf-8"
            )

            result = self.run_aggregate(root)

            grouped = result["grouped"][0]
            self.assertEqual(grouped["num_runs"], 1)
            self.assertEqual(grouped["num_prompts"], 32)
            self.assertEqual(grouped["prompts_with_speculative_mismatch"], 3)


if __name__ == "__main__":
    unittest.main()
