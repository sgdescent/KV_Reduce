import unittest
from pathlib import Path

from aggregate_value_precision_quality_sweep import enforce_full_run


class ValuePrecisionQualityAggregationTest(unittest.TestCase):
    def test_full_run_gate_rejects_underfilled_sequences(self) -> None:
        underfilled = {"requested": 12, "actual": 7}
        with self.assertRaisesRegex(ValueError, "requested 12, observed 7"):
            enforce_full_run(
                Path("summary.json"),
                underfilled,
                enabled=True,
            )

    def test_full_run_gate_can_be_disabled_for_exploration(self) -> None:
        enforce_full_run(
            Path("summary.json"),
            {"requested": 12, "actual": 7},
            enabled=False,
        )


if __name__ == "__main__":
    unittest.main()
