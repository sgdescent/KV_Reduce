import json
import tempfile
import unittest
from pathlib import Path

from paper.aggregate_objective_campaign import (
    collect_final_rows,
    collect_rows,
    discover_aggregates,
    discover_final_results,
    matrix_label,
    strict_matrix_issues,
)


def write_summary(root: Path, name: str, *, missing: int = 0) -> None:
    out_dir = root / name / "aggregate"
    out_dir.mkdir(parents=True)
    payload = {
        "required_evaluator_versions": {
            "quality": "teacher_forced_cached_v1",
            "acceptance": "cached_dynamic_v6_sequential_target",
        },
        "num_missing_pairs": missing,
        "num_rejected_pairs": 0,
        "integrity_gates": {
            "require_complete": True,
            "complete_matrix_gate": True,
            "require_exact_target": True,
            "exact_target_gate": True,
            "paired_within_objective_gate": True,
            "objective_allocations_byte_matched_gate": True,
        },
        "grouped": [
            {
                "budget": 6,
                "context": 1024,
                "allocation_objective": "quality",
                "total_cache_saved_fraction": 0.25,
            }
        ],
        "cross_context_effects": [
            {
                "budget": 6,
                "paired_acceptance_mean": 0.01,
                "paired_acceptance_ci_low": 0.0,
                "paired_acceptance_ci_high": 0.02,
            }
        ],
        "kv_priority_cross_context_effects": [
            {
                "budget": 6,
                "paired_acceptance_mean": 0.02,
                "paired_quality_kl_mean": 0.03,
            }
        ],
        "native_acceptance_cross_context_effects": [
            {
                "budget": 6,
                "allocation_objective": "quality",
                "paired_acceptance_mean": -0.005,
                "paired_acceptance_ci_low": -0.01,
                "paired_acceptance_ci_high": 0.0,
            }
        ],
        "exactness_audit": {
            "totals": {
                "exact": 99,
                "numerical_tie": 1,
                "non_tie_or_unknown": 0,
                "invalid_prompts": 0,
            }
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def write_final_summary(root: Path, name: str) -> None:
    out_dir = root / name / "final_results"
    out_dir.mkdir(parents=True)
    payload = {
        "evaluator_versions": {
            "quality": "teacher_forced_cached_v1",
            "acceptance": "cached_dynamic_v6_sequential_target",
        },
        "baseline": {"quality_nll": 2.0, "spec_accept_rate": 0.5},
        "rows": [
            {
                "allocation": "quality_optimized",
                "all_component_mean_bits": 6.0,
                "total_cache_saved_fraction": 0.27,
                "spec_accept_rate_delta": 0.01,
            },
            {
                "allocation": "acceptance_optimized",
                "all_component_mean_bits": 6.0,
                "total_cache_saved_fraction": 0.27,
                "spec_accept_rate_delta": 0.02,
            }
        ],
        "acceptance_exactness_audit": {
            "valid_prompts": 62,
            "excluded_non_tie_prompts": 2,
            "effects": {
                "quality_vs_native": {"mean": 0.005, "ci_low": -0.01, "ci_high": 0.02},
                "acceptance_vs_native": {"mean": 0.015, "ci_low": -0.005, "ci_high": 0.03},
            },
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


class ObjectiveCampaignAggregationTest(unittest.TestCase):
    def test_long_context_matrix_labels_identify_dataset(self) -> None:
        self.assertEqual(
            matrix_label("qwen25_all_layers_long_context_v1"),
            "Qwen / WikiText / all-layer / 8K-16K",
        )
        self.assertEqual(
            matrix_label("qwen25_all_layers_pg19_long_v1"),
            "Qwen / PG19 / all-layer / 16K-32K",
        )

    def test_powered_matrix_labels_report_held_out_scale(self) -> None:
        self.assertEqual(
            matrix_label("qwen25_accept_mass_powered_1k_v1"),
            "Qwen / WikiText / top-8 / mass-UCB / 384 held-out",
        )

    def test_exact_matrix_labels_identify_family_and_regime(self) -> None:
        self.assertEqual(
            matrix_label("olmo2_exact_aggressive_objective_matrix_v2_bytes"),
            "OLMo-2 7B/1B / aggressive / exact-byte",
        )
        self.assertEqual(
            matrix_label("smollm2_exact_objective_matrix_v2_bytes"),
            "SmolLM2 1.7B/360M / mild / exact-byte",
        )
        self.assertEqual(
            matrix_label("llama_all_layers_powered_b6_v1"),
            "Llama / WikiText / all-layer / b6 / 384 held-out",
        )

    def test_discovers_only_complete_non_smoke_matrices_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "complete")
            write_summary(root, "incomplete", missing=1)
            write_summary(root, "matrix_smoke")

            records, rejected = discover_aggregates(
                root,
                include_incomplete=False,
                include_smoke=False,
            )

            self.assertEqual([record["matrix"] for record in records], ["complete"])
            self.assertEqual([record["matrix"] for record in rejected], ["incomplete"])

    def test_rejects_legacy_nonsequential_objective_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "legacy")
            summary_path = root / "legacy" / "aggregate" / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["required_evaluator_versions"]["acceptance"] = "cached_dynamic_v4"
            summary_path.write_text(json.dumps(payload), encoding="utf-8")

            records, rejected = discover_aggregates(
                root,
                include_incomplete=False,
                include_smoke=False,
            )

            self.assertEqual(records, [])
            self.assertEqual([record["matrix"] for record in rejected], ["legacy"])

    def test_rejects_matrix_without_explicit_integrity_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "legacy_integrity")
            summary_path = root / "legacy_integrity" / "aggregate" / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            del payload["integrity_gates"]
            summary_path.write_text(json.dumps(payload), encoding="utf-8")

            records, rejected = discover_aggregates(
                root,
                include_incomplete=False,
                include_smoke=False,
            )

            self.assertEqual(records, [])
            self.assertIn("integrity gates", rejected[0]["validation_issues"])

    def test_rejects_non_tie_target_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "non_tie")
            summary_path = root / "non_tie" / "aggregate" / "summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["exactness_audit"]["totals"]["non_tie_or_unknown"] = 1
            summary_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertIn("non-tie or unknown target mismatches", strict_matrix_issues(payload))

    def test_collects_effects_savings_and_exactness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_summary(root, "complete")
            records, _ = discover_aggregates(root, include_incomplete=False, include_smoke=False)

            rows = collect_rows(records)

            self.assertEqual(rows["objective"][0]["total_cache_saved_fraction"], 0.25)
            self.assertEqual(rows["kv"][0]["paired_quality_kl_mean"], 0.03)
            self.assertEqual(rows["native"][0]["paired_acceptance_mean"], -0.005)
            self.assertAlmostEqual(rows["exactness"][0]["exact_or_tie_fraction"], 1.0)

    def test_discovers_and_flattens_final_result_campaigns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_final_summary(root, "all_layer")

            records = discover_final_results(root, include_smoke=False)
            rows = collect_final_rows(records)

            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["valid_evaluators"])
            self.assertEqual(rows[0]["baseline_spec_accept_rate"], 0.5)
            self.assertEqual(rows[0]["spec_accept_rate_delta_raw"], 0.01)
            self.assertEqual(rows[0]["spec_accept_rate_delta"], 0.005)
            self.assertEqual(rows[0]["spec_accept_rate_delta_ci_low"], -0.01)
            self.assertEqual(rows[0]["exactness_valid_prompts"], 62)
            self.assertEqual(rows[0]["total_cache_saved_fraction"], 0.27)


if __name__ == "__main__":
    unittest.main()
